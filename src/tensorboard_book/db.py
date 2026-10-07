"""SQLite storage for runs, metrics, and your annotations.

There are two kinds of data in the database:

* Derived data (``runs`` scan fields, ``metrics``, ``series``) that the scanner
  can always rebuild from the event files.
* Annotations (tags, notes, stars, archive flags, groups, properties,
  metric directions, settings) that exist only here. Every edit also
  rewrites a JSON backup, so deleting the database never loses them.

The database runs in WAL mode so several browser sessions and a CLI scan can
use it at the same time. Both files live in a hidden ``.tensorboard-book``
folder inside the runs folder (see :func:`storage_dir`), so they move with
the runs but stay out of the way.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl

SCHEMA_VERSION = 1

# Everything the app writes goes into this hidden folder of the runs folder.
DATA_DIR = ".tensorboard-book"
DB_NAME = "index.db"
ANNOTATIONS_NAME = "annotations.json"
# Where versions before 0.6 kept their files, directly in the runs folder.
LEGACY_FILES = {
    ".tbook.db": DB_NAME,
    ".tbook.db-wal": DB_NAME + "-wal",
    ".tbook.db-shm": DB_NAME + "-shm",
    "tbook_annotations.json": ANNOTATIONS_NAME,
}

DEFAULT_SETTINGS = {
    "active_minutes": 300,
    "gap_minutes": 360,
    "seed_regex": r"[_-]?seed[_-]?\d+",
    "max_points": 5000,
    "max_zip_mb": 1024,
    "default_metrics": [],
    "tensorboard_args": "",
    "max_tensorboards": 5,
}


def storage_dir(root: Path) -> Path:
    """Return the folder for the database and annotations, creating it.

    It is ``ROOT/.tensorboard-book``, so it is copied along when the runs
    folder is moved. When the runs folder is read-only, a per-folder
    directory in ``~/.cache/tensorboard-book`` is used instead. Files left
    in the runs folder by older versions are moved in on first use.

    Args:
        root: Runs folder.

    Returns:
        The storage folder.
    """
    root = Path(root).resolve()
    if os.access(root, os.W_OK):
        folder = root / DATA_DIR
    else:
        digest = hashlib.sha1(str(root).encode()).hexdigest()[:12]
        folder = (
            Path.home() / ".cache" / "tensorboard-book" / f"{root.name}-"
            f"{digest}"
        )
    folder.mkdir(parents=True, exist_ok=True)
    for old, new in LEGACY_FILES.items():
        source, target = root / old, folder / new
        if source.is_file() and not target.exists():
            try:
                source.rename(target)
            except OSError:
                pass  # Read-only root: the old files are left in place.
    return folder


def default_path(root: Path) -> Path:
    """Return the default database path for a runs folder.

    Args:
        root: Runs folder.

    Returns:
        ``ROOT/.tensorboard-book/index.db`` (see :func:`storage_dir`).
    """
    return storage_dir(root) / DB_NAME


def annotations_file(db_path: Path) -> Path:
    """Return the JSON backup that belongs to a database.

    Args:
        db_path: Database file.

    Returns:
        ``annotations.json`` next to the database.
    """
    return Path(db_path).parent / ANNOTATIONS_NAME


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    fingerprint TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    missing INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    starred INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    signature TEXT,
    first_seen REAL,
    last_scanned REAL,
    last_file_mtime REAL,
    start_time REAL,
    end_time REAL,
    wall_span REAL,
    compute_time REAL,
    n_segments INTEGER,
    segments TEXT,
    min_step INTEGER,
    max_step INTEGER,
    n_events INTEGER,
    total_bytes INTEGER,
    artifact_bytes INTEGER,
    n_artifacts INTEGER,
    hparams TEXT,
    hparam_sources TEXT,
    texts TEXT,
    other_tags TEXT,
    has_nonfinite INTEGER NOT NULL DEFAULT 0,
    parse_error TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    n INTEGER,
    first_step INTEGER,
    last_step INTEGER,
    min REAL,
    max REAL,
    last REAL,
    step_min INTEGER,
    step_max INTEGER,
    nonfinite INTEGER,
    PRIMARY KEY (run_id, tag)
);
CREATE TABLE IF NOT EXISTS series (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (run_id, tag)
);
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    selection_metric TEXT,
    columns TEXT,
    created REAL
);
CREATE TABLE IF NOT EXISTS group_members (
    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    PRIMARY KEY (group_id, run_id)
);
CREATE TABLE IF NOT EXISTS run_tags (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (run_id, tag)
);
CREATE TABLE IF NOT EXISTS metric_directions (
    tag TEXT PRIMARY KEY,
    direction TEXT NOT NULL CHECK (direction IN ('max', 'min'))
);
CREATE TABLE IF NOT EXISTS properties (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('number', 'category')),
    description TEXT NOT NULL DEFAULT '',
    created REAL
);
CREATE TABLE IF NOT EXISTS run_properties (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL
        REFERENCES properties(name) ON DELETE CASCADE ON UPDATE CASCADE,
    value NOT NULL,
    PRIMARY KEY (run_id, name)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the database, creating the schema if needed.

    Args:
        db_path: Path of the SQLite file.

    Returns:
        A connection with ``sqlite3.Row`` rows, WAL mode and foreign keys on.
    """
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None):
    """Read one value from the ``meta`` table.

    Args:
        conn: Open connection.
        key: Meta key.
        default: Value returned when the key is absent.

    Returns:
        The stored string, or ``default``.
    """
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Write one value to the ``meta`` table (no commit).

    Args:
        conn: Open connection.
        key: Meta key.
        value: String value.
    """
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def data_version(conn: sqlite3.Connection) -> int:
    """Return a counter that increases on every write, used as a cache key.

    Args:
        conn: Open connection.

    Returns:
        The current data version.
    """
    return int(get_meta(conn, "data_version", "0"))


def bump_version(conn: sqlite3.Connection) -> None:
    """Increase the data version (no commit).

    Args:
        conn: Open connection.
    """
    set_meta(conn, "data_version", str(data_version(conn) + 1))


def get_settings(conn: sqlite3.Connection) -> dict:
    """Return app settings merged over the defaults.

    Args:
        conn: Open connection.

    Returns:
        Settings dictionary with every key from ``DEFAULT_SETTINGS``.
    """
    stored = json.loads(get_meta(conn, "settings", "{}"))
    return {**DEFAULT_SETTINGS, **stored}


def save_settings(conn: sqlite3.Connection, values: dict) -> None:
    """Update some settings and commit the edit.

    Args:
        conn: Open connection.
        values: Settings to overwrite.
    """
    settings = {**get_settings(conn), **values}
    set_meta(conn, "settings", json.dumps(settings))
    commit_edit(conn)


def commit_edit(conn: sqlite3.Connection) -> None:
    """Commit an annotation edit and refresh the JSON backup.

    Args:
        conn: Open connection.
    """
    bump_version(conn)
    conn.commit()
    path = get_meta(conn, "annotations_path")
    if path:
        try:
            export_annotations(conn, path)
        except OSError:
            pass  # A read-only folder must not break editing.


def encode_series(
    steps: np.ndarray, walls: np.ndarray, values: np.ndarray
) -> bytes:
    """Pack a scalar series into a compressed blob.

    Args:
        steps: Step numbers.
        walls: Wall-clock timestamps in seconds.
        values: Scalar values.

    Returns:
        zlib-compressed float64 array of shape ``(3, n)``.
    """
    stacked = np.vstack([steps, walls, values]).astype(np.float64)
    return zlib.compress(stacked.tobytes(), 6)


RUN_COLUMNS = {
    "id": pl.Int64,
    "fingerprint": pl.String,
    "path": pl.String,
    "missing": pl.Int64,
    "archived": pl.Int64,
    "starred": pl.Int64,
    "notes": pl.String,
    "first_seen": pl.Float64,
    "last_scanned": pl.Float64,
    "last_file_mtime": pl.Float64,
    "start_time": pl.Float64,
    "end_time": pl.Float64,
    "wall_span": pl.Float64,
    "compute_time": pl.Float64,
    "n_segments": pl.Int64,
    "min_step": pl.Int64,
    "max_step": pl.Int64,
    "n_events": pl.Int64,
    "total_bytes": pl.Int64,
    "artifact_bytes": pl.Int64,
    "n_artifacts": pl.Int64,
    "has_nonfinite": pl.Int64,
    "parse_error": pl.String,
}
METRIC_COLUMNS = {
    "run_id": pl.Int64,
    "tag": pl.String,
    "n": pl.Int64,
    "first_step": pl.Int64,
    "last_step": pl.Int64,
    "min": pl.Float64,
    "max": pl.Float64,
    "last": pl.Float64,
    "step_min": pl.Int64,
    "step_max": pl.Int64,
    "nonfinite": pl.Int64,
}
DETAIL_COLUMNS = {
    "hparams": {},
    "hparam_sources": [],
    "texts": {},
    "other_tags": {},
    "segments": [],
}


def query(
    conn: sqlite3.Connection, table: str, schema: dict, where: str = ""
) -> pl.DataFrame:
    """Read columns of a table into a polars DataFrame with a fixed schema.

    A fixed schema keeps column types stable even when every value of a
    column is NULL.

    Args:
        conn: Open connection.
        table: Table name.
        schema: Mapping of column name to polars type, in output order.
        where: Optional SQL after ``FROM table`` (e.g. ``ORDER BY name``).

    Returns:
        The rows as a DataFrame.
    """
    sql = f"SELECT {', '.join(schema)} FROM {table} {where}"
    rows = [tuple(r) for r in conn.execute(sql)]
    return pl.DataFrame(rows, schema=schema, orient="row")


def load_series(
    conn: sqlite3.Connection, run_id: int, tag: str
) -> pl.DataFrame:
    """Load one stored scalar series.

    Args:
        conn: Open connection.
        run_id: Run id.
        tag: Scalar tag.

    Returns:
        DataFrame with ``step``, ``wall_time`` and ``value`` columns (empty
        if the run did not log that tag).
    """
    row = conn.execute(
        "SELECT data FROM series WHERE run_id = ? AND tag = ?", (run_id, tag)
    ).fetchone()
    if row is None:
        arr = np.empty((3, 0))
    else:
        arr = np.frombuffer(zlib.decompress(row["data"]), dtype=np.float64)
        arr = arr.reshape(3, -1)
    return pl.DataFrame({"step": arr[0], "wall_time": arr[1], "value": arr[2]})


def load_runs(conn: sqlite3.Connection) -> pl.DataFrame:
    """Load every run with its tags and group names.

    Args:
        conn: Open connection.

    Returns:
        DataFrame with the columns of ``RUN_COLUMNS``, plus ``name`` and
        the sorted list columns ``tags`` and ``groups``. The JSON fields
        are loaded separately by :func:`load_run_details`.
    """
    runs = query(conn, "runs", RUN_COLUMNS, "ORDER BY id")
    tags: dict[int, list[str]] = {}
    for r in conn.execute("SELECT run_id, tag FROM run_tags ORDER BY tag"):
        tags.setdefault(r["run_id"], []).append(r["tag"])
    groups: dict[int, list[str]] = {}
    for r in conn.execute(
        "SELECT m.run_id, g.name FROM group_members m "
        "JOIN groups g ON g.id = m.group_id ORDER BY g.name"
    ):
        groups.setdefault(r["run_id"], []).append(r["name"])
    ids = runs["id"].to_list()
    return runs.with_columns(
        pl.col("path").alias("name"),
        pl.Series("tags", [tags.get(i, []) for i in ids], pl.List(pl.String)),
        pl.Series(
            "groups", [groups.get(i, []) for i in ids], pl.List(pl.String)
        ),
    )


def load_run_details(conn: sqlite3.Connection) -> dict[int, dict]:
    """Load the JSON fields of every run, decoded.

    Args:
        conn: Open connection.

    Returns:
        Mapping of run id to a dict with ``hparams``, ``hparam_sources``,
        ``texts``, ``other_tags`` and ``segments``.
    """
    sql = f"SELECT id, {', '.join(DETAIL_COLUMNS)} FROM runs"
    return {
        r["id"]: {
            col: json.loads(r[col]) if r[col] else type(empty)()
            for col, empty in DETAIL_COLUMNS.items()
        }
        for r in conn.execute(sql)
    }


def load_metrics(conn: sqlite3.Connection) -> pl.DataFrame:
    """Load per-run metric summaries in long format.

    Args:
        conn: Open connection.

    Returns:
        DataFrame with one row per (run, tag).
    """
    return query(conn, "metrics", METRIC_COLUMNS)


def load_groups(conn: sqlite3.Connection) -> list[dict]:
    """Load groups with their member run ids.

    Args:
        conn: Open connection.

    Returns:
        One dict per group, sorted by name, with ``id``, ``name``,
        ``description``, ``selection_metric``, ``columns`` (a list) and
        ``members`` (a list of run ids).
    """
    members: dict[int, list[int]] = {}
    for r in conn.execute("SELECT group_id, run_id FROM group_members"):
        members.setdefault(r["group_id"], []).append(r["run_id"])
    return [
        {
            "id": g["id"],
            "name": g["name"],
            "description": g["description"],
            "selection_metric": g["selection_metric"],
            "columns": json.loads(g["columns"]) if g["columns"] else [],
            "members": members.get(g["id"], []),
        }
        for g in conn.execute("SELECT * FROM groups ORDER BY name")
    ]


def load_directions(conn: sqlite3.Connection) -> dict[str, str]:
    """Return the user-set metric directions.

    Args:
        conn: Open connection.

    Returns:
        Mapping of tag to ``"max"`` or ``"min"``.
    """
    rows = conn.execute(
        "SELECT tag, direction FROM metric_directions"
    ).fetchall()
    return {r["tag"]: r["direction"] for r in rows}


def set_run_flag(
    conn: sqlite3.Connection, run_ids: list[int], field: str, value
) -> None:
    """Set ``starred``, ``archived`` or ``notes`` on some runs.

    Args:
        conn: Open connection.
        run_ids: Runs to update.
        field: One of ``starred``, ``archived``, ``notes``.
        value: New value.

    Raises:
        ValueError: If ``field`` is not editable.
    """
    if field not in ("starred", "archived", "notes"):
        raise ValueError(f"Field {field!r} is not editable")
    conn.executemany(
        f"UPDATE runs SET {field} = ? WHERE id = ?",
        [(value, i) for i in run_ids],
    )
    commit_edit(conn)


def add_tags(
    conn: sqlite3.Connection, run_ids: list[int], tags: list[str]
) -> None:
    """Attach tags to runs.

    Args:
        conn: Open connection.
        run_ids: Runs to tag.
        tags: Tag names; surrounding whitespace and empty names are dropped.
    """
    tags = [t.strip() for t in tags if t.strip()]
    conn.executemany(
        "INSERT OR IGNORE INTO run_tags(run_id, tag) VALUES (?, ?)",
        [(r, t) for r in run_ids for t in tags],
    )
    commit_edit(conn)


def remove_tags(
    conn: sqlite3.Connection, run_ids: list[int], tags: list[str]
) -> None:
    """Detach tags from runs.

    Args:
        conn: Open connection.
        run_ids: Runs to untag.
        tags: Tag names.
    """
    conn.executemany(
        "DELETE FROM run_tags WHERE run_id = ? AND tag = ?",
        [(r, t) for r in run_ids for t in tags],
    )
    commit_edit(conn)


def set_run_tags(
    conn: sqlite3.Connection, run_id: int, tags: list[str]
) -> None:
    """Replace all tags of one run.

    Args:
        conn: Open connection.
        run_id: Run to update.
        tags: The complete new tag list.
    """
    conn.execute("DELETE FROM run_tags WHERE run_id = ?", (run_id,))
    add_tags(conn, [run_id], tags)


def rename_tag(conn: sqlite3.Connection, old: str, new: str) -> None:
    """Rename a tag everywhere, merging into ``new`` if it already exists.

    Args:
        conn: Open connection.
        old: Current tag name.
        new: New tag name.
    """
    conn.execute(
        "INSERT OR IGNORE INTO run_tags(run_id, tag) "
        "SELECT run_id, ? FROM run_tags WHERE tag = ?",
        (new.strip(), old),
    )
    conn.execute("DELETE FROM run_tags WHERE tag = ?", (old,))
    commit_edit(conn)


def delete_tag(conn: sqlite3.Connection, tag: str) -> None:
    """Remove a tag from every run.

    Args:
        conn: Open connection.
        tag: Tag name.
    """
    conn.execute("DELETE FROM run_tags WHERE tag = ?", (tag,))
    commit_edit(conn)


def create_group(
    conn: sqlite3.Connection, name: str, description: str = ""
) -> int:
    """Create a group, or return the existing one with that name.

    Args:
        conn: Open connection.
        name: Unique group name.
        description: Free text.

    Returns:
        The group id.
    """
    conn.execute(
        (
            "INSERT OR IGNORE INTO groups(name, description, created) VALUES "
            "(?, ?, ?)"
        ),
        (name.strip(), description, time.time()),
    )
    commit_edit(conn)
    row = conn.execute("SELECT id FROM groups WHERE name = ?", (name.strip(),))
    return row.fetchone()["id"]


def update_group(conn: sqlite3.Connection, group_id: int, **fields) -> None:
    """Update group fields.

    Args:
        conn: Open connection.
        group_id: Group to update.
        **fields: Any of ``name``, ``description``, ``selection_metric``,
            ``columns`` (a list, stored as JSON).

    Raises:
        ValueError: If an unknown field is given.
    """
    allowed = {"name", "description", "selection_metric", "columns"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown group fields: {sorted(unknown)}")
    if "columns" in fields:
        fields["columns"] = json.dumps(fields["columns"])
    for key, value in fields.items():
        conn.execute(
            f"UPDATE groups SET {key} = ? WHERE id = ?", (value, group_id)
        )
    commit_edit(conn)


def delete_group(conn: sqlite3.Connection, group_id: int) -> None:
    """Delete a group. Its runs are untouched.

    Args:
        conn: Open connection.
        group_id: Group to delete.
    """
    conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
    commit_edit(conn)


def add_to_group(
    conn: sqlite3.Connection, group_id: int, run_ids: list[int]
) -> None:
    """Add runs to a group.

    Args:
        conn: Open connection.
        group_id: Target group.
        run_ids: Runs to add.
    """
    conn.executemany(
        "INSERT OR IGNORE INTO group_members(group_id, run_id) VALUES (?, ?)",
        [(group_id, r) for r in run_ids],
    )
    commit_edit(conn)


def remove_from_group(
    conn: sqlite3.Connection, group_id: int, run_ids: list[int]
) -> None:
    """Remove runs from a group.

    Args:
        conn: Open connection.
        group_id: Group to edit.
        run_ids: Runs to remove.
    """
    conn.executemany(
        "DELETE FROM group_members WHERE group_id = ? AND run_id = ?",
        [(group_id, r) for r in run_ids],
    )
    commit_edit(conn)


def set_run_groups(
    conn: sqlite3.Connection, run_id: int, group_ids: list[int]
) -> None:
    """Replace the group memberships of one run.

    Args:
        conn: Open connection.
        run_id: Run to update.
        group_ids: The complete new list of groups.
    """
    conn.execute("DELETE FROM group_members WHERE run_id = ?", (run_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO group_members(group_id, run_id) VALUES (?, ?)",
        [(g, run_id) for g in group_ids],
    )
    commit_edit(conn)


def set_directions(
    conn: sqlite3.Connection, directions: dict[str, str]
) -> None:
    """Store "higher/lower is better" choices for metrics.

    Args:
        conn: Open connection.
        directions: Mapping of tag to ``"max"`` or ``"min"``.
    """
    conn.executemany(
        "INSERT INTO metric_directions(tag, direction) VALUES (?, ?) "
        "ON CONFLICT(tag) DO UPDATE SET direction = excluded.direction",
        list(directions.items()),
    )
    commit_edit(conn)


def clear_directions(
    conn: sqlite3.Connection, tags: list[str] | None = None
) -> None:
    """Forget user-set directions, so those metrics use the guess again.

    Args:
        conn: Open connection.
        tags: Metrics to reset; all of them when None.
    """
    if tags is None:
        conn.execute("DELETE FROM metric_directions")
    else:
        conn.executemany(
            "DELETE FROM metric_directions WHERE tag = ?",
            [(t,) for t in tags],
        )
    commit_edit(conn)


PROPERTY_KINDS = ("number", "category")


def create_property(
    conn: sqlite3.Connection, name: str, kind: str, description: str = ""
) -> None:
    """Define a property that runs can be given a value for.

    Args:
        conn: Open connection.
        name: Unique property name, e.g. ``dataset`` or ``gpu_hours``.
        kind: ``"number"`` or ``"category"``.
        description: What the property means.

    Raises:
        ValueError: If the name is empty or taken, or the kind is unknown.
    """
    name = name.strip()
    if not name:
        raise ValueError("Give the property a name.")
    if kind not in PROPERTY_KINDS:
        raise ValueError(f"Kind must be one of {PROPERTY_KINDS}.")
    exists = conn.execute(
        "SELECT 1 FROM properties WHERE name = ?", (name,)
    ).fetchone()
    if exists:
        raise ValueError(f"A property named {name!r} already exists.")
    conn.execute(
        "INSERT INTO properties(name, kind, description, created) "
        "VALUES (?, ?, ?, ?)",
        (name, kind, description.strip(), time.time()),
    )
    commit_edit(conn)


def update_property(
    conn: sqlite3.Connection,
    name: str,
    new_name: str | None = None,
    description: str | None = None,
) -> None:
    """Rename a property or change its description. Values are kept.

    Args:
        conn: Open connection.
        name: Current name.
        new_name: New name, if renaming.
        description: New description, if changing it.

    Raises:
        ValueError: If the new name is empty or taken.
    """
    if description is not None:
        conn.execute(
            "UPDATE properties SET description = ? WHERE name = ?",
            (description.strip(), name),
        )
    if new_name is not None and new_name.strip() != name:
        new_name = new_name.strip()
        taken = conn.execute(
            "SELECT 1 FROM properties WHERE name = ?", (new_name,)
        ).fetchone()
        if not new_name or taken:
            raise ValueError(f"Can't rename to {new_name!r}.")
        # ON UPDATE CASCADE carries the values over to the new name.
        conn.execute(
            "UPDATE properties SET name = ? WHERE name = ?", (new_name, name)
        )
    commit_edit(conn)


def delete_property(conn: sqlite3.Connection, name: str) -> None:
    """Delete a property and every run's value for it.

    Args:
        conn: Open connection.
        name: Property name.
    """
    conn.execute("DELETE FROM properties WHERE name = ?", (name,))
    commit_edit(conn)


def property_value(kind: str, value):
    """Check and convert a value for a property of the given kind.

    Args:
        kind: ``"number"`` or ``"category"``.
        value: The value typed in (None or "" clears it).

    Returns:
        A float for numbers, a stripped string for categories, or None.

    Raises:
        ValueError: If a number property gets something that isn't one.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if kind == "number":
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{value!r} is not a number.") from None
        if number != number:  # NaN
            raise ValueError("NaN is not a value.")
        return number
    return str(value).strip()


def set_property(
    conn: sqlite3.Connection, run_ids: list[int], name: str, value
) -> None:
    """Set (or clear, with None) one property on some runs.

    Args:
        conn: Open connection.
        run_ids: Runs to change.
        name: Property name.
        value: The value; None or an empty string clears it.

    Raises:
        ValueError: If the property doesn't exist or the value doesn't fit.
    """
    row = conn.execute(
        "SELECT kind FROM properties WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No property named {name!r}.")
    value = property_value(row["kind"], value)
    if value is None:
        conn.executemany(
            "DELETE FROM run_properties WHERE run_id = ? AND name = ?",
            [(r, name) for r in run_ids],
        )
    else:
        conn.executemany(
            "INSERT INTO run_properties(run_id, name, value) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, name) DO UPDATE SET value = excluded.value",
            [(r, name, value) for r in run_ids],
        )
    commit_edit(conn)


def load_properties(conn: sqlite3.Connection) -> list[dict]:
    """Load property definitions with how they are used.

    Args:
        conn: Open connection.

    Returns:
        One dict per property, sorted by name, with ``name``, ``kind``,
        ``description``, ``runs`` (how many runs have a value) and
        ``values`` (the distinct values, sorted).
    """
    props = []
    for p in conn.execute("SELECT * FROM properties ORDER BY name"):
        values = [
            r["value"]
            for r in conn.execute(
                "SELECT value FROM run_properties WHERE name = ?",
                (p["name"],),
            )
        ]
        props.append(
            {
                "name": p["name"],
                "kind": p["kind"],
                "description": p["description"],
                "runs": len(values),
                "values": sorted(set(values), key=str),
            }
        )
    return props


def load_run_properties(conn: sqlite3.Connection) -> dict[int, dict]:
    """Load every run's property values.

    Args:
        conn: Open connection.

    Returns:
        Mapping of run id to ``{property name: value}``.
    """
    values: dict[int, dict] = {}
    for r in conn.execute("SELECT run_id, name, value FROM run_properties"):
        values.setdefault(r["run_id"], {})[r["name"]] = r["value"]
    return values


def forget_missing_runs(conn: sqlite3.Connection) -> int:
    """Delete database rows (and annotations) of runs whose folder is gone.

    Files on disk are never touched.

    Args:
        conn: Open connection.

    Returns:
        Number of runs forgotten.
    """
    cur = conn.execute("DELETE FROM runs WHERE missing = 1")
    commit_edit(conn)
    return cur.rowcount


def export_annotations(
    conn: sqlite3.Connection, path: str | Path | None = None
):
    """Serialize every annotation to JSON.

    Runs are identified by fingerprint and path, so the file can be imported
    into a fresh database even after folders were renamed.

    Args:
        conn: Open connection.
        path: If given, the JSON is written there atomically.

    Returns:
        The annotations as a dictionary.
    """
    runs = {}
    for r in conn.execute(
        "SELECT id, fingerprint, path, starred, archived, notes FROM runs"
    ):
        runs[r["id"]] = {
            "fingerprint": r["fingerprint"],
            "path": r["path"],
            "starred": bool(r["starred"]),
            "archived": bool(r["archived"]),
            "notes": r["notes"],
            "tags": [],
            "properties": {},
        }
    for r in conn.execute("SELECT run_id, tag FROM run_tags ORDER BY tag"):
        runs[r["run_id"]]["tags"].append(r["tag"])
    for run_id, values in load_run_properties(conn).items():
        runs[run_id]["properties"] = values
    groups = []
    for g in conn.execute("SELECT * FROM groups ORDER BY name").fetchall():
        members = conn.execute(
            "SELECT r.fingerprint FROM group_members m JOIN runs r ON "
            "r.id = m.run_id "
            "WHERE m.group_id = ?",
            (g["id"],),
        ).fetchall()
        groups.append(
            {
                "name": g["name"],
                "description": g["description"],
                "selection_metric": g["selection_metric"],
                "columns": json.loads(g["columns"]) if g["columns"] else [],
                "members": [m["fingerprint"] for m in members],
            }
        )
    data = {
        "format": "tensorboard-book-annotations",
        "version": 1,
        "exported_at": time.time(),
        "settings": get_settings(conn),
        "metric_directions": load_directions(conn),
        "properties": [
            {k: p[k] for k in ("name", "kind", "description")}
            for p in load_properties(conn)
        ],
        "groups": groups,
        "runs": [
            r
            for r in runs.values()
            if r["tags"]
            or r["notes"]
            or r["starred"]
            or r["archived"]
            or r["properties"]
        ],
    }
    if path is not None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, path)
    return data


def import_annotations(conn: sqlite3.Connection, data: dict) -> dict:
    """Merge annotations from :func:`export_annotations` into the database.

    Runs are matched by fingerprint first, then by path. Runs that are not in
    the database yet are skipped, so scan before importing.

    Args:
        conn: Open connection.
        data: The decoded JSON.

    Returns:
        Counts of matched and skipped runs.

    Raises:
        ValueError: If the data is not an annotations export.
    """
    if data.get("format") != "tensorboard-book-annotations":
        raise ValueError("Not a tensorboard-book annotations file")
    by_fp = {
        r["fingerprint"]: r["id"] for r in conn.execute("SELECT * FROM runs")
    }
    by_path = {r["path"]: r["id"] for r in conn.execute("SELECT * FROM runs")}
    for p in data.get("properties", []):
        conn.execute(
            "INSERT INTO properties(name, kind, description, created) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "description = excluded.description",
            (p["name"], p["kind"], p.get("description", ""), time.time()),
        )
    kinds = {
        r["name"]: r["kind"]
        for r in conn.execute("SELECT name, kind FROM properties")
    }
    matched = skipped = 0
    for run in data.get("runs", []):
        run_id = by_fp.get(run["fingerprint"]) or by_path.get(run["path"])
        if run_id is None:
            skipped += 1
            continue
        matched += 1
        conn.execute(
            (
                "UPDATE runs SET starred = ?, archived = ?, notes = ? WHERE "
                "id = ?"
            ),
            (int(run["starred"]), int(run["archived"]), run["notes"], run_id),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO run_tags(run_id, tag) VALUES (?, ?)",
            [(run_id, t) for t in run["tags"]],
        )
        for name, value in run.get("properties", {}).items():
            if name in kinds:
                conn.execute(
                    "INSERT INTO run_properties(run_id, name, value) "
                    "VALUES (?, ?, ?) ON CONFLICT(run_id, name) "
                    "DO UPDATE SET value = excluded.value",
                    (run_id, name, property_value(kinds[name], value)),
                )
    for g in data.get("groups", []):
        conn.execute(
            "INSERT INTO groups(name, description, selection_metric, "
            "columns, created) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "description = excluded.description, "
            "selection_metric = excluded.selection_metric, "
            "columns = excluded.columns",
            (
                g["name"],
                g.get("description", ""),
                g.get("selection_metric"),
                json.dumps(g.get("columns", [])),
                time.time(),
            ),
        )
        gid = conn.execute(
            "SELECT id FROM groups WHERE name = ?", (g["name"],)
        )
        gid = gid.fetchone()["id"]
        conn.executemany(
            (
                "INSERT OR IGNORE INTO group_members(group_id, run_id) VALUES "
                "(?, ?)"
            ),
            [(gid, by_fp[fp]) for fp in g.get("members", []) if fp in by_fp],
        )
    conn.executemany(
        "INSERT INTO metric_directions(tag, direction) VALUES (?, ?) "
        "ON CONFLICT(tag) DO UPDATE SET direction = excluded.direction",
        list(data.get("metric_directions", {}).items()),
    )
    if data.get("settings"):
        merged = {**get_settings(conn), **data["settings"]}
        set_meta(conn, "settings", json.dumps(merged))
    commit_edit(conn)
    return {"matched": matched, "skipped": skipped}
