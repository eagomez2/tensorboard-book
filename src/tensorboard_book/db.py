"""SQLite storage for runs, metrics, and your annotations.

There are two kinds of data in the database:

* Derived data (``runs`` scan fields, ``metrics``, ``series``) that the scanner
  can always rebuild from the event files.
* Annotations (tags, notes, stars, archive flags, collections, fields,
  metric directions and display, settings) that exist only here. Every
  edit also rewrites a JSON backup, so deleting the database never loses
  them.

The database runs in WAL mode so several browser sessions and a CLI scan can
use it at the same time. Both files live in a hidden ``.tensorboard-book``
folder inside the runs folder (see :func:`storage_dir`), so they move with
the runs but stay out of the way.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from tensorboard_book import fields
from tensorboard_book.fields import Field

# 2: properties became fields (types, groups, units, options), with the
# order of the tag options and the display of metrics.
SCHEMA_VERSION = 2
ANNOTATIONS_FORMAT = "tensorboard-book-annotations"
ANNOTATIONS_VERSION = 2
FIELDS_FORMAT = "tensorboard-book-fields"

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
    grp TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    unit TEXT NOT NULL DEFAULT '',
    scale TEXT NOT NULL DEFAULT '',
    decimals INTEGER,
    direction TEXT,
    options TEXT NOT NULL DEFAULT '[]',
    visible INTEGER NOT NULL DEFAULT 1,
    position INTEGER NOT NULL DEFAULT 0,
    created REAL
);
CREATE TABLE IF NOT EXISTS run_properties (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL
        REFERENCES properties(name) ON DELETE CASCADE ON UPDATE CASCADE,
    value NOT NULL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS metric_display (
    tag TEXT PRIMARY KEY,
    grp TEXT,
    unit TEXT NOT NULL DEFAULT '',
    scale TEXT NOT NULL DEFAULT '',
    decimals INTEGER
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
    upgrade(conn)
    conn.commit()
    return conn


def upgrade(conn: sqlite3.Connection) -> None:
    """Bring a database from an older version up to date (no commit).

    Version 1 had properties with a ``kind`` of ``number`` or
    ``category``. They become fields: a category becomes a choice field
    whose options are its values in alphabetical order, so pill colors
    start out the same. The annotations file is written first, so the
    previous state can be restored with ``import-annotations``.

    Args:
        conn: Open connection.
    """
    columns = {r[1] for r in conn.execute("PRAGMA table_info(properties)")}
    if "kind" not in columns:
        return
    path = get_meta(conn, "annotations_path")
    if not path:
        # The annotations file is kept next to the database by default.
        db_file = conn.execute("PRAGMA database_list").fetchone()[2]
        path = db_file and str(Path(db_file).with_name("annotations.json"))
    if path and Path(path).exists():
        backup = Path(path).with_name("annotations-v1.json")
        if not backup.exists():
            backup.write_text(Path(path).read_text())
    rows = conn.execute("SELECT * FROM properties ORDER BY name").fetchall()
    conn.commit()
    # Rebuilding the table needs foreign keys off, or dropping it would
    # delete every value.
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN")
        # The new table is made beside the old one and renamed after the
        # old one is dropped, so run_properties keeps referring to
        # "properties".
        body = SCHEMA.split("CREATE TABLE IF NOT EXISTS properties (")[1]
        conn.execute(f"CREATE TABLE properties_v2 ({body.split(');')[0]})")
        for position, row in enumerate(rows):
            kind = "choice" if row["kind"] == "category" else "number"
            options = []
            if kind == "choice":
                options = sorted(
                    {
                        str(v[0])
                        for v in conn.execute(
                            "SELECT value FROM run_properties WHERE name = ?",
                            (row["name"],),
                        )
                    }
                )
            conn.execute(
                "INSERT INTO properties_v2(name, type, description, options, "
                "position, created) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["name"],
                    kind,
                    row["description"],
                    json.dumps(options),
                    position,
                    row["created"],
                ),
            )
        conn.execute("DROP TABLE properties")
        conn.execute("ALTER TABLE properties_v2 RENAME TO properties")
        set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


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


FIELD_COLUMNS = (
    "name",
    "grp",
    "type",
    "description",
    "unit",
    "scale",
    "decimals",
    "direction",
    "options",
    "visible",
    "position",
)


def _field_from_row(row, runs: int = 0) -> Field:
    """Build a :class:`Field` from a row of the ``properties`` table."""
    return Field(
        name=row["name"],
        group=row["grp"] or "",
        type=row["type"],
        description=row["description"] or "",
        unit=row["unit"] or "",
        scale=row["scale"] or "",
        decimals=row["decimals"],
        direction=row["direction"] or None,
        options=tuple(json.loads(row["options"] or "[]")),
        visible=bool(row["visible"]),
        position=row["position"] or 0,
        runs=runs,
    )


def _check_field(fld: Field) -> None:
    """Raise ValueError if a field definition isn't valid."""
    if not fld.name:
        raise ValueError("Give the field a name.")
    if fld.name.casefold() == fields.TAGS.casefold():
        raise ValueError(f"{fields.TAGS} is the built-in tags field.")
    if fld.type not in fields.TYPES:
        raise ValueError(
            f"{fld.name}: the type must be one of {', '.join(fields.TYPES)}."
        )
    if fld.direction not in (None, *fields.DIRECTIONS):
        raise ValueError(f"{fld.name}: better when is higher or lower.")
    if fld.direction and not fld.is_number:
        raise ValueError(f"{fld.name}: only numbers can be higher or lower.")
    if fld.scale not in fields.SCALES:
        raise ValueError(f"{fld.name}: unknown scale {fld.scale!r}.")
    if fld.decimals is not None and not 0 <= fld.decimals <= 10:
        raise ValueError(f"{fld.name}: decimals go from 0 to 10.")


def _write_field(
    conn: sqlite3.Connection, fld: Field, old_name: str | None = None
) -> None:
    """Insert a field, or update the one called ``old_name`` (no commit)."""
    values = (
        fld.name,
        fld.group,
        fld.type,
        fld.description,
        fld.unit if fld.is_number else "",
        fld.scale if fld.is_number else "",
        fld.decimals if fld.is_number else None,
        fld.direction if fld.is_number else None,
        json.dumps(list(fld.options) if fld.has_options else []),
        int(fld.visible),
        fld.position,
    )
    if old_name is None:
        conn.execute(
            f"INSERT INTO properties({', '.join(FIELD_COLUMNS)}, created) "
            f"VALUES ({', '.join('?' * len(FIELD_COLUMNS))}, ?)",
            (*values, time.time()),
        )
    else:
        sets = ", ".join(f"{c} = ?" for c in FIELD_COLUMNS)
        # ON UPDATE CASCADE carries the values over to a new name.
        conn.execute(
            f"UPDATE properties SET {sets} WHERE name = ?",
            (*values, old_name),
        )


def create_field(conn: sqlite3.Connection, fld: Field) -> None:
    """Define a field that runs can be given a value for.

    Args:
        conn: Open connection.
        fld: The definition. Names are unique across all fields.

    Raises:
        ValueError: If the definition isn't valid or the name is taken.
    """
    fld = replace(fld, name=fld.name.strip(), group=fld.group.strip())
    _check_field(fld)
    if get_field(conn, fld.name) is not None:
        raise ValueError(f"A field named {fld.name!r} already exists.")
    _write_field(conn, fld)
    commit_edit(conn)


def update_field(conn: sqlite3.Connection, name: str, fld: Field) -> None:
    """Change a field: rename, regroup, retype, set unit, options, ….

    Values are kept. A type change converts every value, and is refused
    with the values that don't convert.

    Args:
        conn: Open connection.
        name: The field's current name.
        fld: The new definition.

    Raises:
        ValueError: If the definition isn't valid, the new name is taken,
            or values don't convert to a new type.
    """
    old = get_field(conn, name)
    if old is None:
        raise ValueError(f"No field named {name!r}.")
    fld = replace(fld, name=fld.name.strip(), group=fld.group.strip())
    _check_field(fld)
    if fld.name != name and get_field(conn, fld.name) is not None:
        raise ValueError(f"A field named {fld.name!r} already exists.")
    converted = []
    if fld.type != old.type:
        problems = []
        for row in conn.execute(
            "SELECT r.path, v.run_id, v.value FROM run_properties v "
            "JOIN runs r ON r.id = v.run_id WHERE v.name = ?",
            (name,),
        ):
            try:
                value = fields.convert(
                    old, fld.type, fields.decode(old.type, row["value"])
                )
            except ValueError as exc:
                problems.append(f"{row['path']}: {exc}")
                continue
            converted.append((row["run_id"], value))
            fld = fields.with_options(fld, value)
        if problems:
            shown = "; ".join(problems[:5])
            more = (
                f" and {len(problems) - 5} more" if len(problems) > 5 else ""
            )
            raise ValueError(
                f"{name}: can't change to {fld.type}, these values don't "
                f"convert: {shown}{more}."
            )
    _write_field(conn, fld, old_name=name)
    for run_id, value in converted:
        if value is None:
            conn.execute(
                "DELETE FROM run_properties WHERE run_id = ? AND name = ?",
                (run_id, fld.name),
            )
        else:
            conn.execute(
                "UPDATE run_properties SET value = ? "
                "WHERE run_id = ? AND name = ?",
                (fields.encode(fld, value), run_id, fld.name),
            )
    commit_edit(conn)


def delete_field(conn: sqlite3.Connection, name: str) -> None:
    """Delete a field and every run's value for it.

    Args:
        conn: Open connection.
        name: Field name.
    """
    conn.execute("DELETE FROM properties WHERE name = ?", (name,))
    commit_edit(conn)


def get_field(conn: sqlite3.Connection, name: str) -> Field | None:
    """Return one field, the built-in ``tags`` field included.

    Args:
        conn: Open connection.
        name: Field name.

    Returns:
        The field, or None if there is none with that name.
    """
    if name == fields.TAGS:
        return tags_field(conn)
    row = conn.execute(
        "SELECT * FROM properties WHERE name = ?", (name,)
    ).fetchone()
    return None if row is None else _field_from_row(row)


def load_fields(conn: sqlite3.Connection) -> list[Field]:
    """Load the fields you defined, in display order.

    Args:
        conn: Open connection.

    Returns:
        Fields sorted by position, then group and name, each with ``runs``
        (how many runs have a value). The built-in ``tags`` field is not
        included; see :func:`tags_field`.
    """
    counts = dict(
        conn.execute(
            "SELECT name, COUNT(*) FROM run_properties GROUP BY name"
        ).fetchall()
    )
    rows = conn.execute(
        "SELECT * FROM properties ORDER BY position, grp, name"
    ).fetchall()
    return [_field_from_row(r, counts.get(r["name"], 0)) for r in rows]


def tags_field(conn: sqlite3.Connection) -> Field:
    """Return the built-in ``tags`` field with its options in order.

    The order is kept in the ``tag_options`` meta value. Tags that are in
    use but not in it yet (from an older version or an import) follow in
    alphabetical order, so the colors start out as before.

    Args:
        conn: Open connection.

    Returns:
        The field.
    """
    used = [
        r[0]
        for r in conn.execute("SELECT DISTINCT tag FROM run_tags ORDER BY tag")
    ]
    order = json.loads(get_meta(conn, "tag_options", "[]"))
    options = [*order, *(t for t in used if t not in order)]
    runs = conn.execute(
        "SELECT COUNT(DISTINCT run_id) FROM run_tags"
    ).fetchone()[0]
    return Field(
        name=fields.TAGS,
        type="tags",
        description="Tags of the run, used by the Tags filter.",
        options=tuple(options),
        builtin=True,
        runs=runs,
    )


def _save_tag_options(conn: sqlite3.Connection, options) -> None:
    """Store the order of the tag options (no commit)."""
    set_meta(conn, "tag_options", json.dumps(list(dict.fromkeys(options))))


def load_run_fields(
    conn: sqlite3.Connection, defined: list[Field] | None = None
) -> dict[int, dict]:
    """Load every run's field values, decoded by type.

    Args:
        conn: Open connection.
        defined: The fields (loaded when not given).

    Returns:
        Mapping of run id to ``{field name: value}``. Tags of the
        built-in field are not included (they are in ``load_runs``).
    """
    kinds = {f.name: f.type for f in (defined or load_fields(conn))}
    values: dict[int, dict] = {}
    for r in conn.execute("SELECT run_id, name, value FROM run_properties"):
        value = fields.decode(kinds.get(r["name"], "text"), r["value"])
        if value is not None:
            values.setdefault(r["run_id"], {})[r["name"]] = value
    return values


def set_field(
    conn: sqlite3.Connection, run_ids: list[int], name: str, value
) -> None:
    """Set (or clear, with None or "") one field on some runs.

    A choice or tags value that isn't an option yet becomes one, as in
    modelboard.

    Args:
        conn: Open connection.
        run_ids: Runs to change.
        name: Field name (``tags`` replaces the runs' tags).
        value: The value, in any form :func:`fields.coerce` reads.

    Raises:
        ValueError: If the field doesn't exist or the value doesn't fit.
    """
    fld = get_field(conn, name)
    if fld is None:
        raise ValueError(f"No field named {name!r}.")
    value = fields.coerce(fld, value)
    if fld.builtin:
        for run_id in run_ids:
            conn.execute("DELETE FROM run_tags WHERE run_id = ?", (run_id,))
        add_tags(conn, run_ids, value or [])
        return
    grown = fields.with_options(fld, value)
    if grown is not fld:
        _write_field(conn, grown, old_name=name)
    if value is None:
        conn.executemany(
            "DELETE FROM run_properties WHERE run_id = ? AND name = ?",
            [(r, name) for r in run_ids],
        )
    else:
        conn.executemany(
            "INSERT INTO run_properties(run_id, name, value) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, name) DO UPDATE SET value = excluded.value",
            [(r, name, fields.encode(fld, value)) for r in run_ids],
        )
    commit_edit(conn)


def option_counts(conn: sqlite3.Connection, name: str) -> dict[str, int]:
    """Count how many runs have each option of a choice or tags field.

    Args:
        conn: Open connection.
        name: Field name.

    Returns:
        Mapping of option to number of runs, for every option in order.
    """
    fld = get_field(conn, name)
    counts = dict.fromkeys(fld.options, 0)
    if fld.builtin:
        for tag, n in conn.execute(
            "SELECT tag, COUNT(*) FROM run_tags GROUP BY tag"
        ):
            counts[tag] = n
        return counts
    for (stored,) in conn.execute(
        "SELECT value FROM run_properties WHERE name = ?", (name,)
    ):
        value = fields.decode(fld.type, stored)
        for item in value if isinstance(value, list) else [value]:
            counts[item] = counts.get(item, 0) + 1
    return counts


def save_options(
    conn: sqlite3.Connection,
    name: str,
    options: list[str],
    renames: dict[str, str] | None = None,
    deletes: list[str] | None = None,
) -> None:
    """Reorder, rename, merge and delete the options of a field.

    Renaming changes the value on every run that has it; renaming to an
    option that already exists merges the two. Deleting removes the value
    from every run.

    Args:
        conn: Open connection.
        name: A choice or tags field, ``tags`` included.
        options: The options in their new order, under their new names.
        renames: Old name to new name.
        deletes: Options to remove from every run.

    Raises:
        ValueError: If the field has no options.
    """
    fld = get_field(conn, name)
    if fld is None or not fld.has_options:
        raise ValueError(f"{name!r} has no options.")
    renames = {o: n.strip() for o, n in (renames or {}).items() if n.strip()}
    deletes = set(deletes or [])
    options = list(dict.fromkeys(o.strip() for o in options if o.strip()))
    if fld.builtin:
        for old, new in renames.items():
            if old != new:
                conn.execute(
                    "INSERT OR IGNORE INTO run_tags(run_id, tag) "
                    "SELECT run_id, ? FROM run_tags WHERE tag = ?",
                    (new, old),
                )
                conn.execute("DELETE FROM run_tags WHERE tag = ?", (old,))
        conn.executemany(
            "DELETE FROM run_tags WHERE tag = ?", [(t,) for t in deletes]
        )
        _save_tag_options(conn, [o for o in options if o not in deletes])
        commit_edit(conn)
        return
    rows = conn.execute(
        "SELECT run_id, value FROM run_properties WHERE name = ?", (name,)
    ).fetchall()
    for row in rows:
        value = fields.decode(fld.type, row["value"])
        items = value if isinstance(value, list) else [value]
        items = [renames.get(i, i) for i in items if i not in deletes]
        items = list(dict.fromkeys(items))
        if not items:
            conn.execute(
                "DELETE FROM run_properties WHERE run_id = ? AND name = ?",
                (row["run_id"], name),
            )
            continue
        new = items if fld.type == "tags" else items[0]
        conn.execute(
            "UPDATE run_properties SET value = ? "
            "WHERE run_id = ? AND name = ?",
            (fields.encode(fld, new), row["run_id"], name),
        )
    kept = [o for o in options if o not in deletes]
    _write_field(conn, replace(fld, options=tuple(kept)), old_name=name)
    commit_edit(conn)


def load_metric_display(conn: sqlite3.Connection) -> dict[str, dict]:
    """Return how each metric is shown: group, unit, scale and decimals.

    Args:
        conn: Open connection.

    Returns:
        Mapping of tag to ``{"group", "unit", "scale", "decimals"}`` for
        metrics with settings. ``group`` is None when it follows the tag.
    """
    return {
        r["tag"]: {
            "group": r["grp"],
            "unit": r["unit"] or "",
            "scale": r["scale"] or "",
            "decimals": r["decimals"],
        }
        for r in conn.execute("SELECT * FROM metric_display")
    }


def set_metric_display(
    conn: sqlite3.Connection, display: dict[str, dict]
) -> None:
    """Store how metrics are shown; metrics with no settings are removed.

    Args:
        conn: Open connection.
        display: Mapping of tag to ``{"group", "unit", "scale",
            "decimals"}``.

    Raises:
        ValueError: For an unknown scale or decimals out of 0 to 10.
    """
    for tag, d in display.items():
        group = d.get("group")
        unit, scale = d.get("unit") or "", d.get("scale") or ""
        decimals = d.get("decimals")
        if scale not in fields.SCALES:
            raise ValueError(f"{tag}: unknown scale {scale!r}.")
        if decimals is not None and not 0 <= decimals <= 10:
            raise ValueError(f"{tag}: decimals go from 0 to 10.")
        if group is None and not unit and not scale and decimals is None:
            conn.execute("DELETE FROM metric_display WHERE tag = ?", (tag,))
            continue
        conn.execute(
            "INSERT INTO metric_display(tag, grp, unit, scale, decimals) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(tag) DO UPDATE SET "
            "grp = excluded.grp, unit = excluded.unit, "
            "scale = excluded.scale, decimals = excluded.decimals",
            (tag, group, unit, scale, decimals),
        )
    commit_edit(conn)


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


def field_dict(fld: Field) -> dict:
    """Describe a field for JSON files, in modelboard's words.

    Args:
        fld: The field.

    Returns:
        A dict with ``name``, ``group``, ``type``, ``description``,
        ``unit``, ``scale``, ``decimals``, ``direction``, ``options``,
        ``visible`` and ``position``.
    """
    return {
        "name": fld.name,
        "group": fld.group,
        "type": fld.type,
        "description": fld.description,
        "unit": fld.unit,
        "scale": fld.scale,
        "decimals": fld.decimals,
        "direction": fld.direction,
        "options": list(fld.options),
        "visible": fld.visible,
        "position": fld.position,
    }


def field_from_dict(data: dict) -> Field:
    """Read a field from :func:`field_dict` or a version 1 property.

    Version 1 files describe properties as ``{"name", "kind",
    "description"}``, where ``kind`` is ``number`` or ``category``; a
    category becomes a choice field. modelboard's fields files work too.

    Args:
        data: The dict.

    Returns:
        The field.
    """
    kind = data.get("type") or data.get("kind") or "text"
    if kind == "category":
        kind = "choice"
    decimals = data.get("decimals")
    return Field(
        name=str(data["name"]).strip(),
        group=str(data.get("group") or ""),
        type=kind,
        description=str(data.get("description") or ""),
        unit=str(data.get("unit") or ""),
        scale=str(data.get("scale") or ""),
        decimals=None if decimals is None else int(decimals),
        direction=data.get("direction") or None,
        options=tuple(str(o) for o in data.get("options") or ()),
        visible=data.get("visible", True) is not False,
        position=int(data.get("position") or 0),
    )


def export_fields(conn: sqlite3.Connection) -> dict:
    """Describe every field, to set up another runs folder the same way.

    Args:
        conn: Open connection.

    Returns:
        ``{"format": "tensorboard-book-fields", "version": 1, "fields":
        [...], "tag_options": [...]}``. Values are not included.
    """
    return {
        "format": FIELDS_FORMAT,
        "version": 1,
        "exported_at": now_iso(),
        "fields": [field_dict(f) for f in load_fields(conn)],
        "tag_options": list(tags_field(conn).options),
    }


def import_fields(
    conn: sqlite3.Connection, data: dict, update_existing: bool = False
) -> dict:
    """Add the fields of an exported file; values are never touched.

    Reads :func:`export_fields` files, annotation files (their fields) and
    modelboard's fields files.

    Args:
        conn: Open connection.
        data: The decoded JSON.
        update_existing: Update fields that exist (matched by name) from
            the file, instead of keeping them as they are.

    Returns:
        The names ``added``, ``updated`` and ``kept``.

    Raises:
        ValueError: If the file has no fields, or a field isn't valid.
    """
    items = data.get("fields")
    if items is None:
        items = data.get("properties")
    if not isinstance(items, list):
        raise ValueError("This file has no fields.")
    result = {"added": [], "updated": [], "kept": []}
    for item in items:
        fld = field_from_dict(item)
        if fld.name == fields.TAGS:
            continue
        old = get_field(conn, fld.name)
        if old is None:
            create_field(conn, fld)
            result["added"].append(fld.name)
        elif update_existing:
            update_field(conn, fld.name, fld)
            result["updated"].append(fld.name)
        else:
            result["kept"].append(fld.name)
    if data.get("tag_options"):
        current = tags_field(conn).options
        _save_tag_options(
            conn,
            [*data["tag_options"], *current]
            if update_existing
            else [*current, *data["tag_options"]],
        )
        commit_edit(conn)
    return result


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
    defined = load_fields(conn)
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
            "fields": {},
        }
    for r in conn.execute("SELECT run_id, tag FROM run_tags ORDER BY tag"):
        runs[r["run_id"]]["tags"].append(r["tag"])
    for run_id, values in load_run_fields(conn, defined).items():
        runs[run_id]["fields"] = values
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
        "format": ANNOTATIONS_FORMAT,
        "version": ANNOTATIONS_VERSION,
        "exported_at": now_iso(),
        "settings": get_settings(conn),
        "metric_directions": load_directions(conn),
        "metric_display": load_metric_display(conn),
        "fields": [field_dict(f) for f in defined],
        "tag_options": list(tags_field(conn).options),
        # Called groups inside the file; the app calls them collections.
        "groups": groups,
        "runs": [
            r
            for r in runs.values()
            if r["tags"]
            or r["notes"]
            or r["starred"]
            or r["archived"]
            or r["fields"]
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
    the database yet are skipped, so scan before importing. Version 1 files
    (properties with a ``kind``) are read too.

    Args:
        conn: Open connection.
        data: The decoded JSON.

    Returns:
        Counts of matched and skipped runs.

    Raises:
        ValueError: If the data is not an annotations export, or is from a
            newer version.
    """
    if data.get("format") != ANNOTATIONS_FORMAT:
        raise ValueError("Not a tensorboard-book annotations file")
    if int(data.get("version", 1)) > ANNOTATIONS_VERSION:
        raise ValueError(
            "This annotations file is from a newer tensorboard-book. "
            "Update tensorboard-book to import it."
        )
    by_fp = {
        r["fingerprint"]: r["id"] for r in conn.execute("SELECT * FROM runs")
    }
    by_path = {r["path"]: r["id"] for r in conn.execute("SELECT * FROM runs")}
    for item in data.get("fields", data.get("properties", [])):
        fld = field_from_dict(item)
        old = get_field(conn, fld.name)
        if fld.name == fields.TAGS:
            continue
        if old is None:
            _check_field(fld)
            _write_field(conn, fld)
        else:
            merged = fields.with_options(
                replace(fld, options=old.options), list(fld.options)
            )
            if fld.type == old.type:
                _write_field(conn, merged, old_name=fld.name)
    if data.get("tag_options"):
        _save_tag_options(
            conn, [*tags_field(conn).options, *data["tag_options"]]
        )
    defined = {f.name: f for f in load_fields(conn)}
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
        values = run.get("fields", run.get("properties", {}))
        for name, value in values.items():
            fld = defined.get(name)
            if fld is None:
                continue
            value = fields.coerce(fld, value)
            if value is None:
                continue
            grown = fields.with_options(fld, value)
            if grown is not fld:
                _write_field(conn, grown, old_name=name)
                defined[name] = grown
            conn.execute(
                "INSERT INTO run_properties(run_id, name, value) "
                "VALUES (?, ?, ?) ON CONFLICT(run_id, name) "
                "DO UPDATE SET value = excluded.value",
                (run_id, name, fields.encode(fld, value)),
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
    for tag, d in data.get("metric_display", {}).items():
        conn.execute(
            "INSERT INTO metric_display(tag, grp, unit, scale, decimals) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(tag) DO UPDATE SET "
            "grp = excluded.grp, unit = excluded.unit, "
            "scale = excluded.scale, decimals = excluded.decimals",
            (
                tag,
                d.get("group"),
                d.get("unit") or "",
                d.get("scale") or "",
                d.get("decimals"),
            ),
        )
    if data.get("settings"):
        merged = {**get_settings(conn), **data["settings"]}
        set_meta(conn, "settings", json.dumps(merged))
    commit_edit(conn)
    return {"matched": matched, "skipped": skipped}


def now_iso() -> str:
    """Return the current UTC time as ISO 8601 text, as modelboard does.

    Returns:
        For example ``2026-10-09T12:00:00+00:00``.
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
