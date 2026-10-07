"""Find TensorBoard runs in a folder, parse them, and index them into SQLite.

A *run* is the directory ``depth`` levels below the root (1 by default) that
contains at least one ``events.out.tfevents.*`` file somewhere inside it. Event
files in subfolders of a run are merged into it and their tags are prefixed
with the subfolder, so Keras-style ``train/`` and ``validation/`` folders
become ``train/epoch_loss`` and ``validation/epoch_loss``. PyTorch's
``add_hparams`` timestamp subfolders are merged without a prefix.

Scanning is incremental: a run is only re-parsed when the size or modification
time of one of its event or config files changes.

Runs are identified by a fingerprint of their earliest event file rather than
by path, so renaming or moving a run folder keeps its tags, notes and groups.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import sqlite3
import struct
import time
import warnings
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import yaml

from tensorboard_book import db

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", db.DATA_DIR}
OWN_FILES = {
    ".tbook.db",
    ".tbook.db-wal",
    ".tbook.db-shm",
    "tbook_annotations.json",
    "tbook_annotations.json.tmp",
}
CONFIG_NAMES = {
    f"{stem}.{ext}"
    for stem in ("config", "hparams", "args", "params", "cfg", "opts")
    for ext in ("yaml", "yml", "json")
}
PARSER_VERSION = 2  # Bump to re-parse every run after changing the parser.
MAX_CONFIG_BYTES = 2_000_000
MAX_TEXT_CHARS = 20_000
TIMESTAMP_DIR = re.compile(r"^\d{9,}(\.\d+)?$")
EVENT_TIME = re.compile(r"tfevents\.(\d+)")


def is_event_file(name: str) -> bool:
    """Tell whether a file name is a TensorBoard event file.

    Args:
        name: Base name of the file.

    Returns:
        True for ``events.out.tfevents.*`` files.
    """
    return "tfevents" in name and name.startswith("events")


def discover_runs(root: Path, depth: int = 1) -> dict[str, dict]:
    """Walk the root once and group every file into its run.

    Args:
        root: Folder that holds the runs.
        depth: How many folder levels below ``root`` a run lives. Folders with
            event files that are shallower than ``depth`` become runs
            themselves.

    Returns:
        Mapping of run path (POSIX, relative to root, ``"."`` for the root
        itself) to a dict with ``event_files`` and ``files``. Both are lists of
        ``(path relative to the run, size, mtime_ns)``. ``files`` excludes
        event files.
    """
    root = Path(root)
    all_files = []
    run_keys = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        parts = Path(dirpath).relative_to(root).parts
        if any(is_event_file(f) for f in filenames):
            run_keys.add(parts[:depth])
        for name in filenames:
            if not parts and name in OWN_FILES:
                continue
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue  # Broken symlink or file removed mid-scan.
            all_files.append((parts, name, st.st_size, st.st_mtime_ns))

    runs = {
        key: {"event_files": [], "files": []}
        for key in sorted(run_keys, key=lambda k: (len(k), k))
    }
    for parts, name, size, mtime in all_files:
        # A file belongs to the deepest run folder that contains it.
        owner = next(
            (
                parts[:n]
                for n in range(len(parts), -1, -1)
                if parts[:n] in runs
            ),
            None,
        )
        if owner is None:
            continue
        rel = Path(*parts[len(owner) :], name).as_posix()
        bucket = "event_files" if is_event_file(name) else "files"
        runs[owner][bucket].append((rel, size, mtime))
    return {
        (Path(*key).as_posix() if key else "."): info
        for key, info in runs.items()
    }


def event_file_order(rel: str) -> tuple:
    """Sort key that orders event files by the timestamp in their name.

    Args:
        rel: Event file path relative to its run.

    Returns:
        A tuple usable as a sort key.
    """
    match = EVENT_TIME.search(Path(rel).name)
    return (int(match.group(1)) if match else math.inf, rel)


def fingerprint(run_dir: Path, event_files: list[tuple]) -> str:
    """Identify a run independently of its path.

    Uses the name and first 4 KiB of the earliest event file. Those bytes hold
    the file's creation timestamp and never change as training appends data.

    Args:
        run_dir: Absolute run folder.
        event_files: ``(rel, size, mtime_ns)`` tuples of its event files.

    Returns:
        A hex digest.
    """
    first = min((f[0] for f in event_files), key=event_file_order)
    digest = hashlib.sha1(Path(first).name.encode())
    try:
        with open(run_dir / first, "rb") as fh:
            digest.update(fh.read(4096))
    except OSError:
        pass
    return digest.hexdigest()


def config_files(files: list[tuple]) -> list[str]:
    """Pick the config files inside a run whose values become hyperparameters.

    Args:
        files: Non-event ``(rel, size, mtime_ns)`` tuples of the run.

    Returns:
        Relative paths of config files at the top level, one folder down, or
        in ``.hydra/``.
    """
    picked = []
    for rel, size, _ in files:
        path = Path(rel)
        if size > MAX_CONFIG_BYTES or path.name not in CONFIG_NAMES:
            continue
        if len(path.parts) <= 2 or path.parts[0] == ".hydra":
            picked.append(rel)
    return sorted(picked, key=lambda p: (len(Path(p).parts), p))


def signature(
    event_files: list[tuple], configs: list[str], files: list[tuple]
) -> str:
    """Hash the file stats that decide whether a run needs re-parsing.

    Args:
        event_files: Event file stats.
        configs: Relative paths of config files.
        files: Non-event file stats (only the config ones are used).

    Returns:
        A hex digest.
    """
    config_stats = [f for f in files if f[0] in set(configs)]
    payload = json.dumps(sorted(event_files) + sorted(config_stats))
    return hashlib.sha1(payload.encode()).hexdigest()


class _LenientLoader(yaml.SafeLoader):
    """Safe YAML loader that reads unknown tags as plain data.

    Lightning's ``hparams.yaml`` often contains ``!!python/object`` tags that
    ``yaml.safe_load`` rejects. Nothing is ever instantiated.
    """


def _construct_unknown(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_LenientLoader.add_multi_constructor("", _construct_unknown)


def flatten(data: dict, prefix: str = "") -> dict:
    """Flatten nested dictionaries into dotted keys.

    Args:
        data: Nested mapping, e.g. a parsed config.
        prefix: Prefix for every key (used by the recursion).

    Returns:
        Flat mapping. Numbers, bools and None are kept, lists become short JSON
        strings and anything else becomes a string of at most 200 characters.
    """
    out = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, name + "."))
        elif isinstance(value, bool | int | float) or value is None:
            out[name] = value
        elif isinstance(value, list | tuple):
            out[name] = json.dumps(value, default=str)[:200]
        else:
            out[name] = str(value)[:200]
    return out


def read_config(path: Path) -> dict:
    """Parse a YAML or JSON config file into flat hyperparameters.

    Args:
        path: Config file.

    Returns:
        Flat mapping, or an empty dict if the file is not a mapping or cannot
        be parsed.
    """
    try:
        text = path.read_text(errors="replace")
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            # A SafeLoader subclass: unknown tags become plain data.
            data = yaml.load(text, Loader=_LenientLoader)  # noqa: S506
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return flatten(data) if isinstance(data, dict) else {}


def read_records(path: Path):
    """Yield the raw payloads of a TFRecord file.

    CRCs are not verified, which makes this much faster than TensorBoard's
    pure-Python reader. A truncated last record (a file still being written)
    ends the iteration quietly.

    Args:
        path: Event file.

    Yields:
        Serialized ``Event`` protos as bytes.
    """
    with open(path, "rb") as fh:
        while True:
            header = fh.read(12)
            if len(header) < 12:
                return
            (length,) = struct.unpack("<Q", header[:8])
            if length > 1 << 31:
                return  # Corrupt length field.
            data = fh.read(length)
            if len(data) < length or len(fh.read(4)) < 4:
                return
            yield data


def _proto_value(value) -> object:
    kind = value.WhichOneof("kind")
    if kind == "number_value" and value.number_value.is_integer():
        return int(value.number_value)  # hparams store every number as float.
    if kind in ("number_value", "string_value", "bool_value"):
        return getattr(value, kind)
    return None


def compute_segments(
    walls: np.ndarray, gap_seconds: float
) -> list[list[float]]:
    """Split sorted wall times into activity segments at long silences.

    Args:
        walls: Sorted wall-clock timestamps.
        gap_seconds: Silence longer than this starts a new segment.

    Returns:
        List of ``[start, end]`` pairs.
    """
    if len(walls) == 0:
        return []
    breaks = np.nonzero(np.diff(walls) > gap_seconds)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(walls) - 1]])
    return [[float(walls[s]), float(walls[e])] for s, e in zip(starts, ends)]


def downsample_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    """Choose which points of a long series to keep.

    Keeps an even spread plus the first, last, minimum and maximum points.

    Args:
        values: The series values.
        max_points: Target number of points (0 keeps everything).

    Returns:
        Sorted indices to keep.
    """
    n = len(values)
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    keep = set(np.linspace(0, n - 1, max_points).astype(int).tolist())
    finite = np.isfinite(values)
    if finite.any():
        idx = np.nonzero(finite)[0]
        keep.add(int(idx[np.argmin(values[finite])]))
        keep.add(int(idx[np.argmax(values[finite])]))
    keep.add(n - 1)
    return np.array(sorted(keep))


def parse_run(
    run_dir: str,
    event_files: list[str],
    configs: list[str],
    gap_seconds: float,
    max_points: int,
) -> dict:
    """Read every event and config file of one run.

    This is a plain top-level function so it can run in a worker process.

    Args:
        run_dir: Absolute run folder.
        event_files: Event file paths relative to ``run_dir``.
        configs: Config file paths relative to ``run_dir``.
        gap_seconds: Silence that splits the run into segments.
        max_points: Max stored points per scalar series (0 keeps all).

    Returns:
        Dictionary with metric summaries, compressed series, hyperparameters,
        text summaries, timing, and any parse errors.
    """
    from tensorboard.compat.proto import event_pb2
    from tensorboard.plugins.hparams import plugin_data_pb2
    from tensorboard.util import tensor_util

    run_dir = Path(run_dir)
    scalars: dict[str, tuple[list, list, list]] = {}
    walls: list[float] = []
    hparams: dict = {}
    sources: list[str] = []
    texts: dict[str, str] = {}
    other: dict[str, set] = {}
    errors = []
    n_events = 0

    for rel in sorted(event_files, key=event_file_order):
        sub = Path(rel).parent
        merge = sub == Path(".") or TIMESTAMP_DIR.match(sub.name)
        prefix = "" if merge else sub.as_posix() + "/"
        plugin_by_tag: dict[str, str] = {}
        try:
            for record in read_records(run_dir / rel):
                event = event_pb2.Event.FromString(record)
                n_events += 1
                walls.append(event.wall_time)
                if not event.HasField("summary"):
                    continue
                for value in event.summary.value:
                    # TF2 only writes plugin metadata on a tag's first value.
                    plugin = value.metadata.plugin_data.plugin_name
                    if plugin:
                        plugin_by_tag[value.tag] = plugin
                    else:
                        plugin = plugin_by_tag.get(value.tag, "")
                    kind = value.WhichOneof("value")
                    tag = prefix + value.tag
                    if plugin == "hparams":
                        content = value.metadata.plugin_data.content
                        info = plugin_data_pb2.HParamsPluginData.FromString(
                            content
                        )
                        if info.HasField("session_start_info"):
                            for (
                                k,
                                v,
                            ) in info.session_start_info.hparams.items():
                                hparams[k] = _proto_value(v)
                            if "event hparams" not in sources:
                                sources.append("event hparams")
                    elif kind == "simple_value":
                        series = scalars.setdefault(tag, ([], [], []))
                        series[0].append(event.step)
                        series[1].append(event.wall_time)
                        series[2].append(value.simple_value)
                    elif kind == "tensor" and plugin == "scalars":
                        series = scalars.setdefault(tag, ([], [], []))
                        series[0].append(event.step)
                        series[1].append(event.wall_time)
                        series[2].append(
                            float(tensor_util.make_ndarray(value.tensor))
                        )
                    elif kind == "tensor" and plugin == "text":
                        arr = tensor_util.make_ndarray(value.tensor).flatten()
                        text = "\n".join(
                            x.decode(errors="replace")
                            if isinstance(x, bytes)
                            else str(x)
                            for x in arr
                        )
                        texts[tag] = text[:MAX_TEXT_CHARS]
                    else:
                        other.setdefault(
                            plugin or kind or "unknown", set()
                        ).add(tag)
        except Exception as exc:  # noqa: BLE001 - keep whatever was read.
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")

    config_values: dict = {}
    for rel in configs:
        values = read_config(run_dir / rel)
        if values:
            config_values.update(values)
            sources.append(rel)
    hparams = {**config_values, **hparams}

    metrics, series_blobs = [], []
    has_nonfinite = False
    for tag, (steps, tag_walls, vals) in scalars.items():
        s = np.asarray(steps, dtype=np.int64)
        w = np.asarray(tag_walls, dtype=np.float64)
        v = np.asarray(vals, dtype=np.float64)
        order = np.lexsort((w, s))
        s, w, v = s[order], w[order], v[order]
        finite = np.isfinite(v)
        nonfinite = int((~finite).sum())
        has_nonfinite |= nonfinite > 0
        row = {
            "tag": tag,
            "n": len(v),
            "first_step": int(s[0]),
            "last_step": int(s[-1]),
            "last": float(v[-1]) if finite[-1] else None,
            "nonfinite": nonfinite,
            "min": None,
            "max": None,
            "step_min": None,
            "step_max": None,
        }
        if finite.any():
            idx = np.nonzero(finite)[0]
            imin = idx[np.argmin(v[finite])]
            imax = idx[np.argmax(v[finite])]
            row.update(
                min=float(v[imin]),
                max=float(v[imax]),
                step_min=int(s[imin]),
                step_max=int(s[imax]),
            )
        metrics.append(row)
        keep = downsample_indices(v, max_points)
        series_blobs.append((tag, db.encode_series(s[keep], w[keep], v[keep])))

    wall_arr = np.sort(np.asarray(walls, dtype=np.float64))
    segments = compute_segments(wall_arr, gap_seconds)
    all_steps = [m["first_step"] for m in metrics] + [
        m["last_step"] for m in metrics
    ]
    return {
        "metrics": metrics,
        "series": series_blobs,
        "hparams": hparams,
        "hparam_sources": sources,
        "texts": texts,
        "other_tags": {k: len(v) for k, v in sorted(other.items())},
        "start_time": float(wall_arr[0]) if len(wall_arr) else None,
        "end_time": float(wall_arr[-1]) if len(wall_arr) else None,
        "wall_span": float(wall_arr[-1] - wall_arr[0])
        if len(wall_arr)
        else 0.0,
        "compute_time": float(sum(e - s for s, e in segments)),
        "segments": segments,
        "min_step": min(all_steps) if all_steps else None,
        "max_step": max(all_steps) if all_steps else None,
        "n_events": n_events,
        "has_nonfinite": has_nonfinite,
        "parse_error": "\n".join(errors) or None,
    }


def _save_parsed(
    conn: sqlite3.Connection, run_id: int, sig: str, res: dict
) -> None:
    conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM series WHERE run_id = ?", (run_id,))
    conn.executemany(
        "INSERT INTO metrics(run_id, tag, n, first_step, last_step, "
        "min, max, last, step_min, step_max, nonfinite) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                run_id,
                m["tag"],
                m["n"],
                m["first_step"],
                m["last_step"],
                m["min"],
                m["max"],
                m["last"],
                m["step_min"],
                m["step_max"],
                m["nonfinite"],
            )
            for m in res["metrics"]
        ],
    )
    conn.executemany(
        "INSERT INTO series(run_id, tag, data) VALUES (?, ?, ?)",
        [(run_id, tag, blob) for tag, blob in res["series"]],
    )
    conn.execute(
        "UPDATE runs SET signature = ?, last_scanned = ?, start_time = ?, "
        "end_time = ?, wall_span = ?, compute_time = ?, n_segments = ?, "
        "segments = ?, min_step = ?, max_step = ?, n_events = ?, hparams = ?, "
        "hparam_sources = ?, texts = ?, other_tags = ?, has_nonfinite = ?, "
        "parse_error = ? WHERE id = ?",
        (
            sig,
            time.time(),
            res["start_time"],
            res["end_time"],
            res["wall_span"],
            res["compute_time"],
            len(res["segments"]),
            json.dumps(res["segments"][:500]),
            res["min_step"],
            res["max_step"],
            res["n_events"],
            json.dumps(res["hparams"], default=str),
            json.dumps(res["hparam_sources"]),
            json.dumps(res["texts"]),
            json.dumps(res["other_tags"]),
            int(res["has_nonfinite"]),
            res["parse_error"],
            run_id,
        ),
    )


def scan(
    conn: sqlite3.Connection,
    root: str | Path,
    depth: int = 1,
    workers: int = 1,
    force: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Bring the database in sync with the runs folder.

    New runs are added, changed runs re-parsed, renamed runs followed by
    fingerprint, and runs whose folder disappeared marked as missing (their
    annotations are kept).

    Args:
        conn: Open database connection.
        root: Folder that holds the runs.
        depth: Folder depth of runs below ``root``.
        workers: Worker processes for parsing (1 parses in this process).
            Only use more than 1 from a guarded entry point (the CLI), not
            from inside Streamlit.
        force: Re-parse every run even if unchanged. This also happens
            automatically when the segment gap, point limit or parser version
            changed since the last scan.
        progress: Optional ``callback(done, total, run_path)``.

    Returns:
        Counts: ``total``, ``new``, ``parsed``, ``renamed``, ``missing``.
    """
    root = Path(root).resolve()
    settings = db.get_settings(conn)
    gap_seconds = float(settings["gap_minutes"]) * 60
    max_points = int(settings["max_points"])
    # Settings that change parse results: when they differ from the last scan,
    # every run is re-parsed so all runs use the same rules.
    parse_key = json.dumps([PARSER_VERSION, gap_seconds, max_points])
    if db.get_meta(conn, "parse_key") != parse_key:
        force = True
    discovered = discover_runs(root, depth)
    rows = conn.execute(
        "SELECT id, fingerprint, path, signature FROM runs"
    ).fetchall()
    by_fp = {r["fingerprint"]: r for r in rows}
    by_path = {r["path"]: r for r in rows}
    fps = {
        path: fingerprint(root / path, info["event_files"])
        for path, info in discovered.items()
    }
    # A copied run folder has the same fingerprint as its original. The copy
    # that already owns the record keeps it, the others get a path suffix.
    for fp in [f for f, n in Counter(fps.values()).items() if n > 1]:
        paths = sorted(p for p, f in fps.items() if f == fp)
        owner = (
            by_fp[fp]["path"]
            if fp in by_fp and by_fp[fp]["path"] in paths
            else paths[0]
        )
        for path in paths:
            if path != owner:
                fps[path] = (
                    f"{fp}-{hashlib.sha1(path.encode()).hexdigest()[:8]}"
                )
    live_fps = set(fps.values())
    now = time.time()
    counts = {
        "total": len(discovered),
        "new": 0,
        "parsed": 0,
        "renamed": 0,
        "missing": 0,
    }
    seen, jobs = set(), []

    for path, info in discovered.items():
        fp = fps[path]
        row = by_fp.get(fp)
        if (
            row is None
            and path in by_path
            and by_path[path]["fingerprint"] not in live_fps
        ):
            # Same folder, but its first event file changed: keep the record.
            row = by_path[path]
            conn.execute(
                "UPDATE runs SET fingerprint = ? WHERE id = ?", (fp, row["id"])
            )
        if row is None:
            cur = conn.execute(
                (
                    "INSERT INTO runs(fingerprint, path, first_seen) VALUES "
                    "(?, ?, ?)"
                ),
                (fp, path, now),
            )
            run_id, old_sig = cur.lastrowid, None
            counts["new"] += 1
        else:
            run_id, old_sig = row["id"], row["signature"]
            if row["path"] != path:
                counts["renamed"] += 1
        seen.add(run_id)
        configs = config_files(info["files"])
        sig = signature(info["event_files"], configs, info["files"])
        mtimes = [f[2] for f in info["event_files"]]
        artifact_bytes = sum(f[1] for f in info["files"])
        event_bytes = sum(f[1] for f in info["event_files"])
        conn.execute(
            "UPDATE runs SET path = ?, missing = 0, last_file_mtime = ?, "
            "total_bytes = ?, artifact_bytes = ?, n_artifacts = ? "
            "WHERE id = ?",
            (
                path,
                max(mtimes) / 1e9,
                artifact_bytes + event_bytes,
                artifact_bytes,
                len(info["files"]),
                run_id,
            ),
        )
        if force or sig != old_sig:
            event_rel = [f[0] for f in info["event_files"]]
            jobs.append(
                (run_id, path, sig, (str(root / path), event_rel, configs))
            )

    gone = [r["id"] for r in rows if r["id"] not in seen]
    conn.executemany(
        "UPDATE runs SET missing = 1 WHERE id = ?", [(i,) for i in gone]
    )
    counts["missing"] = len(gone)
    conn.commit()

    def job_args(job):
        return (*job[3], gap_seconds, max_points)

    done = 0

    def save(job, result):
        nonlocal done
        _save_parsed(conn, job[0], job[2], result)
        conn.commit()
        done += 1
        if progress:
            progress(done, len(jobs), job[1])

    pending = list(jobs)
    if workers > 1 and len(jobs) > 1:
        # "spawn" behaves the same on Linux, macOS and Windows. Workers must
        # only be started from a guarded entry point such as the CLI, never
        # from inside the Streamlit app: spawned workers re-import the main
        # script, and there that script is the app itself.
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as pool:
                futures = [
                    (job, pool.submit(parse_run, *job_args(job)))
                    for job in jobs
                ]
                for job, future in futures:
                    save(job, future.result())
                    pending.remove(job)
        except BrokenProcessPool:
            warnings.warn(
                (
                    "Parallel parsing failed; parsing the remaining runs in "
                    "one process."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
    for job in pending:
        save(job, parse_run(*job_args(job)))

    counts["parsed"] = len(jobs)
    db.set_meta(conn, "parse_key", parse_key)
    db.set_meta(conn, "last_scan", str(time.time()))
    db.bump_version(conn)
    conn.commit()
    return counts
