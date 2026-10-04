"""Streamlit app for browsing and annotating a folder of TensorBoard runs.

Start it with the ``tensorboard-book`` command rather than
``streamlit run``: the command checks the password setup, runs the first
(parallel) scan, and passes the configuration through environment
variables (``TBOOK_ROOT``, ``TBOOK_DB``, ``TBOOK_DEPTH``).

Scans started from the app always parse in this process. Worker
processes would re-import this script as their main module and start the
app again.

Tables are polars DataFrames. Streamlit can only highlight single cells
through a pandas Styler, so :func:`styled` converts a finished table right
before it is drawn; pandas is not used anywhere else.

Each view is one function below; ``main`` handles login, scanning, the
sidebar filters and picks the view.
"""

from __future__ import annotations

import base64
import difflib
import html
import io
import json
import os
import shlex
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from tensorboard_book import __version__, auth, db, scanner, tbview
from tensorboard_book import analysis as an
from tensorboard_book.helptext import HELP

ROOT = Path(os.environ.get("TBOOK_ROOT", ".")).resolve()
DB_PATH = Path(os.environ.get("TBOOK_DB") or db.default_path(ROOT)).resolve()
DEPTH = int(os.environ.get("TBOOK_DEPTH", "1"))
ANNOTATIONS_PATH = db.annotations_file(DB_PATH)
ASSETS = Path(__file__).with_name("assets")
TENSORBOARD_HOST = os.environ.get("TBOOK_TENSORBOARD_HOST", tbview.LOCAL_HOST)
# "Show" options that are custom properties carry this prefix.
PROP = "prop:"

# View name -> one-color Material Symbols icon for the sidebar menu.
VIEWS = {
    "Experiments": ":material/science:",
    "Run details": ":material/description:",
    "Compare": ":material/leaderboard:",
    "Curves": ":material/show_chart:",
    "Timeline": ":material/view_timeline:",
    "Manage": ":material/tune:",
}
# One line under each view's title saying what the view is for.
PURPOSE = {
    "Experiments": (
        "Every run that matches the sidebar filters, with its best metrics."
    ),
    "Compare": (
        "Each run is read at the step where its selection metric is best, "
        "and every metric is reported at that step."
    ),
    "Curves": "Training curves of up to eight runs, overlaid.",
    "Timeline": "When runs were running, and how much compute they used.",
    "Run details": (
        "Everything about one run: timing, notes, metrics and files."
    ),
    "Manage": (
        "Groups, tags, properties, metric directions, disk usage, settings "
        "and backups."
    ),
}
# Fonts ship inside the package and are served by Streamlit's static file
# server (``--server.enableStaticServing``), so the app works offline. The
# rest adds what the theme options cannot express: tighter page padding, a
# see-through header, the side menu, summary tiles and pills. Colors are
# theme-neutral (opacity and translucent fills), so they work in both the
# light and the dark theme. The look follows modelboard's.
FONTS = "app/static/fonts"
STYLE = f"""
<style>
@font-face {{
    font-family: "IBM Plex Sans"; font-style: normal; font-display: swap;
    font-weight: 100 700;
    src: url("{FONTS}/IBMPlexSans.woff2") format("woff2");
}}
@font-face {{
    font-family: "Newsreader"; font-style: normal; font-display: swap;
    font-weight: 200 800;
    src: url("{FONTS}/Newsreader.woff2") format("woff2");
}}
@font-face {{
    font-family: "IBM Plex Mono"; font-style: normal; font-display: swap;
    font-weight: 400;
    src: url("{FONTS}/IBMPlexMono-Regular.woff2") format("woff2");
}}
@font-face {{
    font-family: "IBM Plex Mono"; font-style: normal; font-display: swap;
    font-weight: 500;
    src: url("{FONTS}/IBMPlexMono-Medium.woff2") format("woff2");
}}
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; }}
[data-testid="stHeader"] {{
    background: transparent; pointer-events: none;
}}
[data-testid="stHeader"] button, [data-testid="stHeader"] a,
[data-testid="stStatusWidget"] {{ pointer-events: auto; }}
/* With the sidebar open, the app menu (theme, settings) sits in its
   bottom-left corner, as in modelboard. */
@media (min-width: 768px) {{
    body:has([data-testid="stSidebar"][aria-expanded="true"])
        [data-testid="stHeader"] {{ z-index: 999992; }}
    body:has([data-testid="stSidebar"][aria-expanded="true"])
        [data-testid="stMainMenu"] {{
        position: fixed; left: 0.9rem; bottom: 0.9rem;
    }}
    /* A footer strip in the sidebar's own color (so it follows theme
       switches), under the menu button: the sidebar's content scrolls
       under it instead of showing through the button. */
    [data-testid="stSidebar"][aria-expanded="true"]::after {{
        content: ""; position: absolute; left: 0; right: 0; bottom: 0;
        height: 3.4rem; background: inherit; z-index: 1;
        pointer-events: none;
    }}
    [data-testid="stMainMenuButton"] {{ opacity: 0.6; }}
    [data-testid="stMainMenuButton"]:hover {{ opacity: 1; }}
    [data-testid="stSidebarUserContent"] {{ padding-bottom: 4rem; }}
}}
.tb-muted {{ opacity: 0.68; }}
.tb-brand {{
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.25rem;
}}
.tb-brand b {{
    font-family: Newsreader, serif; font-weight: 500; white-space: nowrap;
}}
.tb-brand small {{
    font-size: 0.75rem; opacity: 0.6; align-self: flex-end;
    padding-bottom: 0.25rem;
}}
.tb-section {{
    font-size: 0.875rem; font-weight: 500; margin: 0.2rem 0 0.2rem 0;
}}
.st-key-nav {{ gap: 0.1rem; }}
.st-key-nav button[kind] {{
    justify-content: flex-start; border: none; color: inherit;
    padding: 0.2rem 0.5rem; min-height: 2rem; border-radius: 0.5rem;
}}
.st-key-nav button[kind] > div {{
    justify-content: flex-start; width: 100%;
}}
.st-key-nav button[kind] > div > span {{ gap: 0.6rem; }}
.st-key-nav [data-testid="stIconMaterial"] {{
    font-size: 1.05rem; opacity: 0.75;
}}
.st-key-nav button[kind] p {{ font-size: 0.95rem; }}
.st-key-nav button[kind="primary"] {{
    background: color-mix(in srgb, currentColor 9%, transparent);
}}
.st-key-nav button[kind="primary"] p {{ font-weight: 600; }}
.st-key-nav button[kind="primary"] [data-testid="stIconMaterial"] {{
    opacity: 1;
}}
.st-key-nav button[kind]:hover {{
    background: color-mix(in srgb, currentColor 6%, transparent);
    color: inherit;
}}
.st-key-nav button[kind="primary"]:hover {{
    background: color-mix(in srgb, currentColor 12%, transparent);
}}
[data-testid="stMetric"] {{ gap: 3px; }}
[data-testid="stMetricLabel"] p {{
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; opacity: 0.68;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.15rem; font-weight: 600;
}}
.tb-pills {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.tb-run-pills {{ gap: 8px 6px; padding: 2px 0; }}
.tb-run-pills .tb-pill {{ padding: 2px 11px; }}
.tb-pill {{
    display: inline-block; padding: 1px 9px; border-radius: 999px;
    font-size: 0.82rem; font-weight: 500; line-height: 1.5;
    white-space: nowrap;
}}
.tb-pill.blue {{ background: rgba(59, 130, 246, 0.18); }}
.tb-pill.green {{ background: rgba(34, 160, 110, 0.20); }}
.tb-pill.orange {{ background: rgba(234, 140, 40, 0.22); }}
.tb-pill.violet {{ background: rgba(139, 92, 246, 0.20); }}
.tb-pill.red {{ background: rgba(220, 70, 60, 0.18); }}
.tb-pill.yellow {{ background: rgba(220, 180, 30, 0.24); }}
.tb-pill.gray {{ background: rgba(128, 128, 128, 0.20); }}
</style>
"""
# Views whose content the sidebar filters change.
FILTER_VIEWS = {"Experiments", "Timeline"}
FILTER_DEFAULTS = {
    "group": "All runs",
    "tags": [],
    "status": ["Active", "Stopped"],
    "search": "",
    "starred": False,
    "archived": False,
}
INFO_COLUMNS = [
    "status",
    "steps",
    "compute",
    "wall span",
    "start",
    "last event",
    "size",
    "tags",
    "groups",
    "notes",
]
# Headers of the columns the app itself adds, shown capitalized ("Wall
# span"). Metric, hyperparameter and property columns keep their names as
# logged, and so do the columns of previewed CSV files.
BUILTIN_COLUMNS = {
    "run",
    "runs",
    "status",
    "steps",
    "compute",
    "wall span",
    "start",
    "end",
    "last event",
    "size",
    "tags",
    "tag",
    "groups",
    "group",
    "notes",
    "config",
    "n",
    "best step",
    "best",
    "last",
    "last step",
    "min",
    "max",
    "points",
    "key",
    "value",
    "duration",
    "pause after",
    "path",
    "modified",
    "selection metric",
    "description",
    "property",
    "kind",
    "values",
    "metric",
    "better",
    "total",
    "artifacts",
    "files",
    "archived",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
TABLE_EXT = {".csv", ".tsv"}
AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac"}
VIDEO_EXT = {".mp4", ".webm", ".mov"}
CODE_LANG = {
    ".py": "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".md": "markdown",
    ".diff": "diff",
}
CHECKPOINT_PATTERN = r"\.(pt|pth|ckpt|safetensors|bin|h5|pkl)$"
MAX_PREVIEW_BYTES = 300_000

# Chart colors, the same as modelboard's: checked for color vision
# deficiencies, normal vision and contrast between neighbors. Teal, orange,
# violet, rose, blue, gold, brick, green; the dark set has the same hues in
# the same order.
PALETTE = {
    "light": [
        "#008C80",
        "#DB6B1F",
        "#6B52C8",
        "#C8457E",
        "#2C7DD3",
        "#A0832A",
        "#984646",
        "#659734",
    ],
    "dark": [
        "#22A596",
        "#D8702C",
        "#8F7BE0",
        "#DB6696",
        "#4A96E3",
        "#AF923C",
        "#B25E5C",
        "#7CA35A",
    ],
}
# Sequential teal ramp, light to dark.
SEQUENTIAL = [
    "#D3ECE8",
    "#A6D8D0",
    "#72BEB3",
    "#3FA294",
    "#1D8173",
    "#0F5F5A",
    "#0A403C",
]
OTHER_COLOR = "#9A968C"
STATUS_COLOR = {
    "Active": "#1F9A8C",
    "Stopped": "#9A968C",
    "Missing": "#C2553A",
}
# Named Streamlit colors for tag, group and property pills: drawn as soft,
# translucent fills with normal text, in tables and elsewhere.
PILL_COLORS = ("blue", "green", "orange", "violet", "red", "yellow", "gray")
# Cell highlights per theme: the best value (warm amber, as in modelboard)
# and values that differ between runs.
CELL_CSS = {
    "light": {
        "best": "background-color: #FBE7CF; color: #7A3F06; font-weight: 600",
        "differ": "background-color: #F1EEE6; font-weight: 600",
    },
    "dark": {
        "best": "background-color: #4A3417; color: #F6C68A; font-weight: 600",
        "differ": "background-color: #2C2A26; font-weight: 600",
    },
}


# ------------------------------------------------------------------------
# Data loading, scanning and small shared helpers
# ------------------------------------------------------------------------


@st.cache_data(max_entries=4, show_spinner=False)
def load_data(db_path: str, version: int) -> dict:
    """Load everything the views need, cached until the data changes.

    Args:
        db_path: Database path (part of the cache key).
        version: ``db.data_version``; any write invalidates the cache.

    Returns:
        Dict with ``runs`` and ``metrics`` (DataFrames), ``groups`` (a
        list of dicts), ``details`` (decoded JSON fields per run id),
        ``hparams`` (a wide DataFrame), ``directions`` and ``settings``.
    """
    conn = db.connect(db_path)
    details = db.load_run_details(conn)
    data = {
        "runs": db.load_runs(conn),
        "metrics": db.load_metrics(conn),
        "groups": db.load_groups(conn),
        "details": details,
        "hparams": an.hparam_frame(
            {i: d["hparams"] for i, d in details.items()}
        ),
        "directions": db.load_directions(conn),
        "settings": db.get_settings(conn),
        "properties": db.load_properties(conn),
        "run_properties": db.load_run_properties(conn),
    }
    conn.close()
    return data


@st.cache_resource
def scan_lock() -> threading.Lock:
    """Return the process-wide lock that keeps scans from overlapping."""
    return threading.Lock()


def run_scan(force: bool = False, quiet: bool = False) -> dict | None:
    """Scan the runs folder, showing progress in the page.

    Args:
        force: Re-parse every run.
        quiet: Skip the progress bar (used by auto-refresh).

    Returns:
        Scan counts, or None if another session is already scanning.
    """
    lock = scan_lock()
    if not lock.acquire(blocking=False):
        return None
    try:
        conn = db.connect(DB_PATH)
        bar = None if quiet else st.progress(0.0, text="Scanning runs…")

        def progress(done, total, path):
            if bar is not None:
                bar.progress(
                    done / total, text=f"Parsed {done}/{total}: {path}"
                )

        counts = scanner.scan(conn, ROOT, DEPTH, 1, force, progress)
        if bar is not None:
            bar.empty()
        conn.close()
        return counts
    finally:
        lock.release()


def viewer_tz():
    """Return the viewer's time zone, falling back to the server's."""
    try:
        return ZoneInfo(st.context.timezone)
    except Exception:  # noqa: BLE001 - missing or unknown time zone.
        return datetime.now().astimezone().tzinfo


def to_local(seconds, tz) -> list[datetime | None]:
    """Convert epoch seconds to naive local datetimes for display.

    Args:
        seconds: Iterable of epoch seconds (None or NaN allowed).
        tz: Target time zone.

    Returns:
        Timezone-naive datetimes, None where the input was missing.
    """
    return [
        None
        if an.is_missing(s)
        else datetime.fromtimestamp(s, tz).replace(tzinfo=None)
        for s in seconds
    ]


def is_dark() -> bool:
    """Tell whether the viewer uses the dark theme."""
    return (getattr(st.context.theme, "type", None) or "light") == "dark"


def palette() -> list[str]:
    """Return the categorical palette for the active light or dark theme."""
    return PALETTE["dark" if is_dark() else "light"]


def cell_css(kind: str) -> str:
    """Return the CSS of a highlighted table cell for the active theme.

    Args:
        kind: ``"best"`` or ``"differ"``.
    """
    return CELL_CSS["dark" if is_dark() else "light"][kind]


def brand_header(size: int = 30, title_size: str = "1.25rem") -> None:
    """Draw the logo, the app name and its version side by side.

    Args:
        size: Logo size in pixels.
        title_size: CSS font size of the name.
    """
    name = "logo-dark.svg" if is_dark() else "logo.svg"
    data = base64.b64encode((ASSETS / name).read_bytes()).decode()
    st.markdown(
        '<div class="tb-brand">'
        f'<img src="data:image/svg+xml;base64,{data}" width="{size}" '
        f'height="{size}" alt="">'
        f'<b style="font-size:{title_size}">tensorboard-book</b>'
        f"<small>v{__version__}</small></div>",
        unsafe_allow_html=True,
    )


def label_colors(labels: list[str], order: list[str]) -> dict[str, str]:
    """Give categories stable colors; past eight they share "Other" gray.

    Args:
        labels: Labels that appear in the chart.
        order: Stable ordering of all possible labels (e.g. every group
            name). A label keeps its color when others are filtered out.

    Returns:
        Mapping of label to hex color.
    """
    colors = palette()
    slot = {name: i for i, name in enumerate(order)}
    return {
        lab: colors[slot[lab]]
        if slot.get(lab, 99) < len(colors)
        else OTHER_COLOR
        for lab in labels
    }


def run_colors(run_ids: list[int]) -> dict[int, str]:
    """Give each plotted run a color slot that stays fixed in the session.

    Args:
        run_ids: Runs being plotted (at most eight).

    Returns:
        Mapping of run id to hex color.
    """
    slots = st.session_state.setdefault("color_slots", {})
    for rid in list(slots):
        if rid not in run_ids:
            del slots[rid]
    for rid in run_ids:
        if rid not in slots:
            used = set(slots.values())
            slots[rid] = next(i for i in range(8) if i not in used)
    colors = palette()
    return {rid: colors[slots[rid]] for rid in run_ids}


def short_path(path: Path, limit: int = 80) -> str:
    """Show a path as inline code, trimmed from the left if it's long.

    The end of a path is the part that tells folders apart, so the start
    is replaced by an ellipsis. Hovering shows the full path.

    Args:
        path: The path.
        limit: Longest text shown, in characters.

    Returns:
        HTML for ``st.caption(..., unsafe_allow_html=True)``.
    """
    full = str(path)
    shown = full if len(full) <= limit else "…" + full[-(limit - 1) :]
    return (
        f'<code title="{html.escape(full)}" style="display:inline-block;'
        "max-width:100%;overflow:hidden;text-overflow:ellipsis;"
        f'white-space:nowrap;vertical-align:bottom">{html.escape(shown)}'
        "</code>"
    )


def go_to(view: str, **state) -> None:
    """Switch view from a button callback, optionally setting state keys.

    Args:
        view: One of ``VIEWS``.
        **state: Extra ``st.session_state`` values, e.g. ``run_id``.
    """
    st.session_state.view = view
    for key, value in state.items():
        st.session_state[key] = value


def tag_counts(metrics: pl.DataFrame, run_ids: list[int]) -> dict[str, int]:
    """Count how many of the given runs logged each metric.

    Args:
        metrics: Long metric summaries.
        run_ids: Runs to consider.

    Returns:
        Mapping of tag to number of runs, most common first, then by name.
    """
    counts = (
        metrics.filter(pl.col("run_id").is_in(run_ids))
        .group_by("tag")
        .len()
        .sort(["len", "tag"], descending=[True, False])
    )
    return dict(counts.iter_rows())


def default_metrics(options: list[str], limit: int = 4) -> list[str]:
    """Pick default metric columns, preferring validation, then test.

    Args:
        options: Available tags, most common first.
        limit: Maximum number to return.

    Returns:
        A short list of tags.
    """

    def rank(tag: str) -> int:
        # Validation first (checkpoints are chosen on it), then test.
        name = tag.lower()
        if "val" in name or "eval" in name:
            return 0
        return 1 if "test" in name else 2

    usable = [t for t in options if "lr" not in t.lower().split("/")[-1]]
    return sorted(usable, key=rank)[:limit]


def widget_state(key: str, options, default) -> None:
    """Prepare the value of a keyed widget in session state.

    Keyed widgets get their starting value here instead of through
    ``default=`` or ``index=``: Streamlit warns when a widget has both a
    default and a value set through session state. This also drops values
    that stopped being options (e.g. a deleted tag), which would otherwise
    make the widget fail.

    Args:
        key: Widget key.
        options: The options the widget is about to get.
        default: Starting value (a list for multiselects).
    """
    allowed = set(options)
    if key not in st.session_state:
        st.session_state[key] = default
    value = st.session_state[key]
    if isinstance(value, list):
        st.session_state[key] = [v for v in value if v in allowed]
    elif value not in allowed:
        st.session_state[key] = default


def name_colors(lists: list[list[str]]) -> dict[str, str]:
    """Give every tag (or group) name a fixed color.

    Args:
        lists: The names of every run (a list of lists).

    Returns:
        Mapping of name to a Streamlit color name, in sorted name order.
    """
    names = sorted({name for items in lists for name in items})
    return {
        name: PILL_COLORS[i % len(PILL_COLORS)] for i, name in enumerate(names)
    }


def pill_column(label: str, lists: list[list[str]]):
    """Column config that shows a list column as colored pills.

    Every tag (or group) keeps the same color everywhere, because colors
    are assigned from the sorted names of all runs, not the rows shown.

    Args:
        label: Column label.
        lists: The names of every run (a list of lists).

    Returns:
        A ``MultiselectColumn`` config.
    """
    colors = name_colors(lists)
    return st.column_config.MultiselectColumn(
        label,
        options=list(colors),
        color=list(colors.values()),
        width="medium",
    )


def info_value(column: str, members: list[dict]):
    """Summarize an info column over one run or the runs of one config.

    Args:
        column: One of ``tags``, ``notes``, ``status``, ``groups``,
            ``steps``.
        members: The runs (as dicts) behind one table row.

    Returns:
        A list for tags and groups (shown as pills), otherwise text or a
        number.
    """
    if column in ("tags", "groups"):
        return sorted({name for m in members for name in m[column]})
    if column == "notes":
        return " · ".join(m["notes"] for m in members if m["notes"])
    if column == "status":
        return ", ".join(sorted({m["status"] for m in members}))
    steps = [m["max_step"] for m in members if m["max_step"] is not None]
    return max(steps) if steps else None


def info_column_config(lists: dict[str, list[list[str]]]) -> dict:
    """Column configs for tag, group and run-list columns.

    Args:
        lists: ``{"tags": ..., "groups": ...}`` with the names of every run,
            so pill colors match the rest of the app.

    Returns:
        A ``column_config`` mapping.
    """
    return {
        "tags": pill_column("tags", lists["tags"]),
        "groups": pill_column("groups", lists["groups"]),
        "runs": st.column_config.ListColumn("runs", width="medium"),
        "notes": st.column_config.TextColumn("notes", width="medium"),
    }


def property_cell(prop: dict, members: list[dict]):
    """Value of a custom property for one table row.

    Args:
        prop: The property definition (``name`` and ``kind``).
        members: The runs behind the row (one, or all seeds of a config).

    Returns:
        For a number, the value (the mean when several runs share the row)
        or None; for a category, the sorted distinct values as a list, so
        they show as pills.
    """
    values = [
        m["properties"][prop["name"]]
        for m in members
        if prop["name"] in m["properties"]
    ]
    if prop["kind"] == "category":
        return sorted({str(v) for v in values})
    return sum(values) / len(values) if values else None


def property_columns(
    options: list[str], properties: list[dict], taken
) -> dict[str, dict]:
    """Map the chosen property options to column labels.

    A property keeps its own name as the column label, unless another
    column already uses it; then ``(property)`` is appended.

    Args:
        options: Chosen "Show" options (only ``prop:`` ones are used).
        properties: All property definitions.
        taken: Column labels already in the table.

    Returns:
        Mapping of column label to property definition, in option order.
    """
    by_name = {p["name"]: p for p in properties}
    columns = {}
    for option in options:
        prop = by_name.get(option.removeprefix(PROP))
        if not option.startswith(PROP) or prop is None:
            continue
        label = prop["name"]
        if label in taken or label in columns:
            label = f"{label} (property)"
        columns[label] = prop
    return columns


def add_property_columns(
    columns: dict, prop_cols: dict[str, dict], members: list[list[dict]]
) -> None:
    """Add property columns to a table being built, one cell per row.

    Args:
        columns: The table's columns so far (changed in place).
        prop_cols: Output of :func:`property_columns`.
        members: For each row, the runs behind it.
    """
    for label, prop in prop_cols.items():
        dtype = pl.Float64 if prop["kind"] == "number" else pl.List(pl.String)
        cells = [property_cell(prop, ms) for ms in members]
        columns[label] = pl.Series(label, cells, dtype)


def number_formats(prop_cols: dict[str, dict]) -> dict:
    """Display formats for number properties (4 significant digits).

    Args:
        prop_cols: Output of :func:`property_columns`.

    Returns:
        Mapping of column label to format function.
    """
    return {
        label: an.format_value
        for label, prop in prop_cols.items()
        if prop["kind"] == "number"
    }


def property_config(columns: dict[str, dict]) -> dict:
    """Column configs that show category properties as colored pills.

    Args:
        columns: Output of :func:`property_columns`.

    Returns:
        A ``column_config`` mapping.
    """
    return {
        label: pill_column(label, [[str(v)] for v in prop["values"]])
        for label, prop in columns.items()
        if prop["kind"] == "category"
    }


def property_input(prop: dict, current, key: str, label: str | None = None):
    """Draw the input for one property value (number or category).

    Args:
        prop: The property definition.
        current: The current value, or None.
        key: Widget key.
        label: Widget label (defaults to the property name).

    Returns:
        The entered value, or None when left empty.
    """
    label = label or prop["name"]
    help_text = prop["description"] or None
    if prop["kind"] == "number":
        return st.number_input(
            label,
            value=None if current is None else float(current),
            format="%g",
            placeholder="empty",
            key=key,
            help=help_text,
        )
    options = [str(v) for v in prop["values"]]
    return st.selectbox(
        label,
        options,
        index=options.index(str(current)) if current in options else None,
        accept_new_options=True,
        placeholder="Choose or type a new value",
        key=key,
        help=help_text,
    )


def show_label(option: str) -> str:
    """Display text of a "Show" option: properties are marked as such."""
    if option.startswith(PROP):
        return f"{option.removeprefix(PROP)} · property"
    return option


def column_label(option: str) -> str:
    """Display text of a "Show" menu option.

    The app's own columns are capitalized, like their table headers;
    properties keep their logged names.
    """
    if option in BUILTIN_COLUMNS:
        return option[0].upper() + option[1:]
    return show_label(option)


def format_hparam(value) -> str:
    """Format a hyperparameter for display (``2`` rather than ``2.0``).

    Args:
        value: Any hyperparameter value.

    Returns:
        Display text; empty for missing values.
    """
    if an.is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def arrow(tag: str, directions: dict) -> str:
    """Label a metric with ↑ (higher is better) or ↓ (lower is better)."""
    down = an.direction_of(tag, directions) == "min"
    return f"{tag} {'↓' if down else '↑'}"


def best_rows(values: list, direction: str) -> set[int]:
    """Find the rows holding the best value of a column.

    Args:
        values: Column values; missing values are ignored.
        direction: ``"max"`` or ``"min"``.

    Returns:
        Positions of every row equal to the best value, or an empty set
        when fewer than two values exist (nothing to compare).
    """
    present = [v for v in values if not an.is_missing(v)]
    if len(present) < 2:
        return set()
    best = min(present) if direction == "min" else max(present)
    return {i for i, v in enumerate(values) if v == best}


def styled(
    frame: pl.DataFrame,
    highlight: dict[str, set[int]] | None = None,
    formats: dict | None = None,
    css: str | None = None,
):
    """Turn a finished polars table into a Styler for ``st.dataframe``.

    Streamlit only supports per-cell styling through pandas' Styler, so
    this converts the table at the last moment. List columns become plain
    lists so they still render as pills.

    Args:
        frame: The table to show.
        highlight: Column name -> row positions to highlight.
        formats: Column name -> function that formats a cell for display.
        css: The CSS applied to highlighted cells; the best-value
            highlight of the active theme by default.

    Returns:
        A pandas Styler.
    """
    table = frame.to_pandas()
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.List):
            table[name] = [[] if v is None else list(v) for v in table[name]]
    styler = table.style
    css = css or cell_css("best")
    for column, rows in (highlight or {}).items():
        marks = [css if i in rows else "" for i in range(len(table))]
        styler = styler.apply(lambda _, m=marks: m, subset=[column])
    for column, func in (formats or {}).items():
        styler = styler.format(func, subset=[column])
    return styler


def headers(columns, config: dict | None = None) -> dict:
    """Add capitalized headers for the app's own columns to a config.

    Args:
        columns: Column names of the table.
        config: The table's ``column_config``, if any.

    Returns:
        A ``column_config`` where every column in ``BUILTIN_COLUMNS`` is
        labelled with a capital first letter.
    """
    config = dict(config or {})
    for name in columns:
        if name not in BUILTIN_COLUMNS:
            continue
        label = name[0].upper() + name[1:]
        current = config.get(name)
        if isinstance(current, dict):
            config[name] = {**current, "label": label}
        elif current is None:
            config[name] = label
    return config


def show_table(data, **kwargs):
    """Draw ``st.dataframe`` with capitalized headers for built-in columns.

    Args:
        data: A polars DataFrame or a pandas Styler from :func:`styled`.
        **kwargs: Passed on to ``st.dataframe``.

    Returns:
        What ``st.dataframe`` returns (the selection, when enabled).
    """
    frame = getattr(data, "data", data)  # A Styler wraps its DataFrame.
    kwargs["column_config"] = headers(
        list(frame.columns), kwargs.get("column_config")
    )
    return st.dataframe(data, **kwargs)


def table_height(rows: int, limit: int = 740) -> int:
    """Height in pixels that fits a table's rows, up to a limit.

    Args:
        rows: Number of rows.
        limit: Maximum height.

    Returns:
        Pixels for ``st.dataframe(height=...)``.
    """
    return min(35 * (rows + 1) + 3, limit)


def list_artifacts(run_dir: Path) -> pl.DataFrame:
    """List the non-event files of a run.

    Args:
        run_dir: Absolute run folder.

    Returns:
        DataFrame with ``path`` (relative), ``size`` and ``modified``.
    """
    rows = []
    for dirpath, dirnames, filenames in os.walk(run_dir, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in scanner.SKIP_DIRS)
        for name in sorted(filenames):
            if scanner.is_event_file(name) or name in scanner.OWN_FILES:
                continue
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            rows.append(
                (
                    full.relative_to(run_dir).as_posix(),
                    stat.st_size,
                    stat.st_mtime,
                )
            )
    return pl.DataFrame(
        rows,
        schema={"path": pl.String, "size": pl.Int64, "modified": pl.Float64},
        orient="row",
    )


def safe_file(run_dir: Path, rel: str) -> Path | None:
    """Resolve a run-relative path, refusing anything outside the root.

    Symlinks that point outside the runs folder are refused, so the app
    never serves files it was not pointed at.

    Args:
        run_dir: Absolute run folder.
        rel: Path relative to the run.

    Returns:
        The resolved file, or None if it is outside the root or not a
        file.
    """
    path = (run_dir / rel).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        return None
    return path


def read_text(path: Path, limit: int = MAX_PREVIEW_BYTES) -> str | None:
    """Read the start of a file as text, or return None if it is binary.

    Args:
        path: File to read.
        limit: Maximum bytes to read.

    Returns:
        Decoded text, or None for binary content.
    """
    with open(path, "rb") as fh:
        chunk = fh.read(limit)
    if b"\x00" in chunk[:8192]:
        return None
    try:
        return chunk.decode("utf-8")
    except UnicodeDecodeError:
        return None


# ------------------------------------------------------------------------
# Page frame: help, header, toolbar and sidebar
# ------------------------------------------------------------------------


@st.dialog("How tensorboard-book works", width="large")
def help_dialog(topic: str) -> None:
    """Show the in-app documentation, opened on one topic.

    Args:
        topic: A key of ``HELP``; usually the current view.
    """
    topics = list(HELP)
    default = topic if topic in HELP else topics[0]
    widget_state("help_topic", topics, default)
    choice = st.pills(
        "Topic", topics, key="help_topic", label_visibility="collapsed"
    )
    st.markdown(HELP[choice or topic])


def open_help(topic: str) -> None:
    """Button callback: remember the topic, then open the help dialog."""
    st.session_state.help_topic = topic
    st.session_state.help_open = True


def page_header(view: str) -> None:
    """Draw the header every view shares: icon, name, purpose and help.

    Args:
        view: The current view name (also the help topic it opens).
    """
    row = st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="bottom",
    )
    row.title(view)
    row.button(
        "How this page works",
        icon=":material/help:",
        type="tertiary",
        key=f"help_{view}",
        on_click=open_help,
        args=(view,),
    )
    st.caption(PURPOSE[view])


def toolbar():
    """Return the bordered action bar used at the bottom of every view.

    Returns:
        A horizontal container; add buttons with ``type="tertiary"``.
    """
    return st.container(
        border=True, horizontal=True, gap="small", vertical_alignment="center"
    )


def section_label(text: str) -> None:
    """Draw a small section heading in the sidebar (e.g. "Views").

    Args:
        text: The heading.
    """
    st.markdown(
        f'<div class="tb-section">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def remember_filter(name: str) -> None:
    """Widget callback: copy a filter widget's value into ``filters``.

    Filter widgets are only drawn in the views they apply to, and
    Streamlit forgets the value of widgets that aren't drawn. Keeping the
    values in ``st.session_state.filters`` preserves them across views.

    Args:
        name: Filter name (the widget key is ``f_<name>``).
    """
    st.session_state.filters[name] = st.session_state[f"f_{name}"]


def filter_widget(draw, name: str, label: str, *args, **kwargs):
    """Draw one filter widget, starting from its remembered value.

    Args:
        draw: The Streamlit widget function, e.g. ``st.multiselect``.
        name: Filter name in ``st.session_state.filters``.
        label: Widget label.
        *args: Extra positional arguments (e.g. the options).
        **kwargs: Extra keyword arguments for the widget.
    """
    key = f"f_{name}"
    value = st.session_state.filters[name]
    if args:  # A widget with options: drop remembered values that vanished.
        widget_state(key, args[0], value)
    elif key not in st.session_state:
        st.session_state[key] = value
    draw(
        label,
        *args,
        key=key,
        on_change=remember_filter,
        args=(name,),
        **kwargs,
    )


def apply_filters(
    runs: pl.DataFrame, by_id: dict[int, dict], filters: dict
) -> pl.DataFrame:
    """Return the runs that pass the sidebar filters.

    Args:
        runs: All runs, with a ``status`` column.
        by_id: Every run as a dict, by id (for the text search).
        filters: ``st.session_state.filters``.

    Returns:
        The filtered runs.
    """
    visible = runs
    group = filters["group"]
    if group == "Ungrouped":
        visible = visible.filter(pl.col("groups").list.len() == 0)
    elif group != "All runs":
        visible = visible.filter(pl.col("groups").list.contains(group))
    if filters["tags"]:
        wanted = filters["tags"]
        visible = visible.filter(
            pl.col("tags").list.eval(pl.element().is_in(wanted)).list.any()
        )
    statuses = filters["status"] or ["Active", "Stopped", "Missing"]
    visible = visible.filter(pl.col("status").is_in(statuses))
    if filters["starred"]:
        visible = visible.filter(pl.col("starred") == 1)
    if not filters["archived"]:
        visible = visible.filter(pl.col("archived") == 0)
    query = filters["search"].strip().lower()
    if query:

        def haystack(run: dict) -> str:
            params = " ".join(f"{k}={v}" for k, v in run["hparams"].items())
            props = " ".join(f"{k}={v}" for k, v in run["properties"].items())
            parts = [run["name"], *run["tags"], run["notes"], params, props]
            return " ".join(parts).lower()

        hits = [i for i in visible["id"] if query in haystack(by_id[i])]
        visible = visible.filter(pl.col("id").is_in(hits))
    return visible


def sidebar(
    runs: pl.DataFrame, groups: list[dict], editor: bool, now: float
) -> None:
    """Draw the sidebar: the menu, filters where they apply, and status.

    As in modelboard, the menu comes first; the scan status and controls
    and the app version sit at the bottom.

    Args:
        runs: All runs, with a ``status`` column.
        groups: All groups.
        editor: Whether the visitor can edit.
        now: Current time.
    """
    view = st.session_state.view
    with st.sidebar:
        section_label("Views")
        with st.container(key="nav", gap=None):
            for name, icon in VIEWS.items():
                st.button(
                    name,
                    icon=icon,
                    key=f"nav_{name}",
                    type="primary" if view == name else "tertiary",
                    width="stretch",
                    on_click=go_to,
                    args=(name,),
                )
            # Looks like a menu item, but opens the help dialog.
            st.button(
                "Help",
                icon=":material/menu_book:",
                key="nav_help",
                type="tertiary",
                width="stretch",
                on_click=open_help,
                args=(view,),
            )

        if view in FILTER_VIEWS:
            section_label("Filters")
            filter_widget(
                st.selectbox,
                "group",
                "Group",
                ["All runs", "Ungrouped", *(g["name"] for g in groups)],
            )
            tags = sorted({t for ts in runs["tags"].to_list() for t in ts})
            filter_widget(st.multiselect, "tags", "Tags (any)", tags)
            filter_widget(
                st.multiselect,
                "status",
                "Status",
                ["Active", "Stopped", "Missing"],
            )
            filter_widget(
                st.text_input,
                "search",
                "Search",
                placeholder="name, tag, note or config value",
            )
            filter_widget(st.toggle, "starred", "Starred only")
            filter_widget(st.toggle, "archived", "Show archived")
        last_scan = st.session_state.get("last_scan", now)
        n_active = runs.filter(pl.col("status") == "Active").height
        st.caption(
            f"{runs.height} runs · {n_active} active · scanned "
            f"{an.format_ago(last_scan, now)}"
        )
        if not editor:
            st.badge("Read-only", color="gray")
        left, right = st.columns(2)
        if left.button("Rescan", icon=":material/refresh:", width="stretch"):
            counts = run_scan()
            if counts is None:
                st.toast("Another scan is running, try again in a moment.")
            else:
                st.session_state.last_scan = time.time()
                st.toast(
                    f"{counts['total']} runs · {counts['new']} new · "
                    f"{counts['parsed']} parsed · "
                    f"{counts['missing']} missing"
                )
            st.rerun()
        right.toggle("Auto", key="auto_refresh", help="Rescan every 60 s")

        if os.environ.get(auth.NO_AUTH_ENV) != "1":
            if st.button("Log out", icon=":material/logout:", type="tertiary"):
                st.session_state.clear()
                st.rerun()
        st.caption(f"tensorboard-book {__version__}")

    if st.session_state.get("auto_refresh"):
        auto_refresh()


@st.fragment(run_every=60)
def auto_refresh() -> None:
    """Rescan in the background every minute; refresh the page on changes."""
    before = st.session_state.get("auto_version")
    counts = run_scan(quiet=True)
    conn = db.connect(DB_PATH)
    version = db.data_version(conn)
    conn.close()
    st.session_state.auto_version = version
    changed = counts and (
        counts["parsed"] or counts["new"] or counts["missing"]
    )
    if before is not None and changed:
        st.session_state.last_scan = time.time()
        st.rerun(scope="app")


# ------------------------------------------------------------------------
# Views
# ------------------------------------------------------------------------


def view_experiments(
    data: dict, visible: pl.DataFrame, editor: bool, now: float
) -> None:
    """Main table of runs with metric, hyperparameter and info columns.

    Args:
        data: Output of :func:`load_data`, plus ``by_id``.
        visible: Runs that pass the sidebar filters.
        editor: Whether the visitor can edit.
        now: Current time.
    """
    metrics, directions = data["metrics"], data["directions"]
    settings, by_id = data["settings"], data["by_id"]
    tz = viewer_tz()
    group_name = st.session_state.filters["group"]
    group = next((g for g in data["groups"] if g["name"] == group_name), None)

    tiles = st.columns(4)

    def summary(shown: pl.DataFrame) -> None:
        """Fill the four summary tiles for the runs in the table."""
        tiles[0].metric("Runs shown", shown.height, border=True)
        active_for = an.format_duration(settings["active_minutes"] * 60)
        tiles[1].metric(
            "Active now",
            shown.filter(pl.col("status") == "Active").height,
            help="A run is **active** if its event files were written in "
            f"the last {active_for}. TensorBoard logs have no 'finished' "
            "marker, so a run that has been quiet for longer counts as "
            "stopped. If one epoch can take longer than that, raise the "
            "threshold in Manage → Settings.",
            border=True,
        )
        tiles[2].metric(
            "Compute time",
            an.format_duration(shown["compute_time"].sum()) or "0s",
            border=True,
        )
        tiles[3].metric(
            "Disk usage",
            an.format_bytes(shown["total_bytes"].sum()) or "0 B",
            border=True,
        )

    if visible.is_empty():
        summary(visible)
        st.info("No runs match the filters. Check the sidebar, or Rescan.")
        return

    ids = visible["id"].to_list()
    counts = tag_counts(metrics, ids)
    options = list(counts)
    saved = group["columns"] if group and group["columns"] else None
    saved = saved or settings["default_metrics"]
    saved = [t for t in saved if t in options] or default_metrics(options)
    metric_key = f"metrics_{group_name}"

    def save_metric_choice():
        if not editor:
            return
        conn = db.connect(DB_PATH)
        chosen = st.session_state[metric_key]
        if group is not None:
            db.update_group(conn, group["id"], columns=chosen)
        else:
            db.save_settings(conn, {"default_metrics": chosen})
        conn.close()

    widget_state(metric_key, options, saved)
    left, mid, right = st.columns([5, 4, 3])
    chosen = left.multiselect(
        "Metrics",
        options,
        key=metric_key,
        format_func=lambda t: f"{arrow(t, directions)}  ({counts[t]} runs)",
        on_change=save_metric_choice,
        help="Saved per group. Set higher or lower is better in Manage.",
    )
    hp_all = data["hparams"].filter(pl.col("id").is_in(ids))
    hp_names = [
        c
        for c in hp_all.columns
        if c != "id" and hp_all[c].null_count() < hp_all.height
    ]
    varying = an.varying_columns(hp_all.select("id", *hp_names))
    widget_state(f"hp_{group_name}", hp_names, varying[:6])
    hp_cols = mid.multiselect(
        "Hyperparameters",
        hp_names,
        key=f"hp_{group_name}",
        help="Defaults to the ones that differ between the shown runs.",
    )
    props = data["properties"]
    prop_options = [PROP + p["name"] for p in props]
    widget_state(
        "info_cols",
        INFO_COLUMNS + prop_options,
        ["status", "steps", "compute", "start", "tags", *prop_options],
    )
    shown_cols = right.multiselect(
        "Show",
        INFO_COLUMNS + prop_options,
        key="info_cols",
        format_func=column_label,
        help="Extra columns, including your custom properties (Manage → "
        "Properties).",
    )
    info_cols = [c for c in shown_cols if not c.startswith(PROP)]
    left, right = st.columns([4, 1])
    expr = left.text_input(
        "Filter",
        key="f_expr",
        placeholder="optimizer.lr < 1e-3 and val/acc > 0.9 and model ~ resnet",
        help="Terms joined by `and`. Operators: == != < <= > >= and ~ "
        "(regex). Keys are hyperparameters, metric tags, or run, status, "
        "steps.",
    )
    mode = right.segmented_control(
        "Value",
        ["Best", "Last"],
        default="Best",
        required=True,
        key="value_mode",
    )

    values = an.metric_table(metrics, ids, options, directions, mode.lower())
    if expr.strip():
        base = visible.select(
            "id",
            pl.col("name").alias("run"),
            "status",
            pl.col("max_step").alias("steps"),
        )
        extra_hp = [c for c in hp_names if c not in base.columns]
        base = base.join(hp_all.select("id", *extra_hp), on="id", how="left")
        extra_metrics = [c for c in options if c not in base.columns]
        base = base.join(
            values.select("id", *extra_metrics), on="id", how="left"
        )
        for prop in props:
            cell = [
                by_id[i]["properties"].get(prop["name"]) for i in base["id"]
            ]
            dtype = pl.Float64 if prop["kind"] == "number" else pl.String
            # "prop:name" always works; plain "name" unless it's taken.
            keys = [PROP + prop["name"]]
            if prop["name"] not in base.columns:
                keys.append(prop["name"])
            base = base.with_columns(
                pl.Series(key, cell, dtype) for key in keys
            )
        try:
            mask = an.apply_filter(base, an.parse_filter(expr))
            visible = visible.filter(
                pl.col("id").is_in(base.filter(mask)["id"].to_list())
            )
        except ValueError as exc:
            st.error(str(exc))

    summary(visible)
    visible = visible.sort("start_time", descending=True, nulls_last=True)
    ids = visible["id"].to_list()
    order = pl.DataFrame({"id": ids}, schema={"id": pl.Int64})
    info = {
        "status": visible["status"],
        "steps": visible["max_step"],
        "compute": visible["compute_time"],
        "wall span": visible["wall_span"],
        "start": pl.Series(
            to_local(visible["start_time"], tz), dtype=pl.Datetime
        ),
        "last event": pl.Series(
            to_local(visible["end_time"], tz), dtype=pl.Datetime
        ),
        "size": visible["total_bytes"],
        "tags": visible["tags"],
        "groups": visible["groups"],
        "notes": visible["notes"].str.slice(0, 80),
    }
    columns = {
        "★": ["★" if s else "" for s in visible["starred"]],
        "run": visible["name"],
    }
    columns.update({c: info[c] for c in info_cols})
    # Custom properties sit between the info columns and the metrics.
    prop_cols = property_columns(shown_cols, props, columns)
    rows_here = [by_id[i] for i in ids]
    for label, prop in prop_cols.items():
        cells = [property_cell(prop, [r]) for r in rows_here]
        dtype = pl.Float64 if prop["kind"] == "number" else pl.List(pl.String)
        columns[label] = pl.Series(label, cells, dtype)
    tag_of = {}
    ordered_values = order.join(
        values, on="id", how="left", maintain_order="left"
    )
    for tag in chosen:
        tag_of[arrow(tag, directions)] = tag
        columns[arrow(tag, directions)] = ordered_values[tag]
    ordered_hp = order.join(hp_all, on="id", how="left", maintain_order="left")
    for col in hp_cols:
        if col not in columns:
            columns[col] = [format_hparam(v) for v in ordered_hp[col]]
    table = pl.DataFrame(columns)

    formats = {c: (lambda v: an.format_value(v) or "–") for c in tag_of}
    for label, prop in prop_cols.items():
        if prop["kind"] == "number":
            formats[label] = an.format_value
    for col in ("compute", "wall span"):
        if col in table.columns:
            formats[col] = an.format_duration
    if "size" in table.columns:
        formats["size"] = an.format_bytes
    highlight = {
        label: best_rows(
            table[label].to_list(), an.direction_of(tag, directions)
        )
        for label, tag in tag_of.items()
    }
    all_lists = {
        "tags": data["runs"]["tags"].to_list(),
        "groups": data["runs"]["groups"].to_list(),
    }
    event = show_table(
        styled(table, highlight, formats),
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="exp_table",
        height=table_height(table.height),
        column_config={
            "★": st.column_config.TextColumn("★", width=30),
            "run": st.column_config.TextColumn("run", width="medium"),
            "steps": st.column_config.NumberColumn(
                "steps", format="localized"
            ),
            "start": st.column_config.DatetimeColumn(
                "start", format="YYYY-MM-DD HH:mm"
            ),
            "last event": st.column_config.DatetimeColumn(
                "last event", format="YYYY-MM-DD HH:mm"
            ),
            **info_column_config(all_lists),
            **property_config(prop_cols),
        },
    )
    selected = [ids[r] for r in event.selection.rows if r < len(ids)]
    st.session_state.selected = selected

    note = " (value at the step it was logged, not smoothed)"
    st.caption(
        f"{table.height} runs · best value per metric highlighted"
        + (note if mode == "Best" else "")
    )
    st.caption(short_path(ROOT), unsafe_allow_html=True)
    # The toolbar: what's selected, then actions on it, then the download.
    bar = toolbar()
    if selected:
        bar.badge(f"{len(selected)} selected", color="blue")
        bar.markdown(":gray[|]")
        selection_actions(
            bar,
            selected,
            by_id,
            data["groups"],
            group,
            editor,
            data["properties"],
            settings,
        )
    else:
        bar.caption(
            "Select rows to plot, compare, group, tag, star or archive them."
        )
    bar.markdown(":gray[|]")
    list_cols = [c for c in ("tags", "groups") if c in table.columns]
    bar.download_button(
        "Export CSV",
        data=lambda: table.with_columns(
            pl.col(c).cast(pl.List(pl.String)).list.join(", ")
            for c in list_cols
        ).write_csv(),
        file_name="experiments.csv",
        mime="text/csv",
        icon=":material/download:",
        type="tertiary",
    )
    tensorboard_panel()


def selection_actions(
    bar,
    selected: list[int],
    by_id: dict,
    groups,
    group,
    editor: bool,
    properties: list[dict],
    settings: dict,
) -> None:
    """Draw the buttons that act on the runs ticked in Experiments.

    Args:
        bar: The toolbar to draw into.
        selected: Ticked run ids.
        by_id: Every run as a dict, by id.
        groups: All groups.
        group: The group chosen in the sidebar, or None.
        editor: Whether the visitor can edit (grouping, tagging, stars).
        properties: All custom property definitions.
        settings: App settings (for launching TensorBoard).
    """
    bar.button(
        "Plot curves",
        icon=":material/show_chart:",
        type="tertiary",
        on_click=go_to,
        args=("Curves",),
        kwargs={"curve_runs": selected[:8]},
    )
    if len(selected) > 1:
        bar.button(
            "Compare",
            icon=":material/leaderboard:",
            type="tertiary",
            on_click=go_to,
            args=("Compare",),
            kwargs={"compare_runs": selected, "cmp_source": "Selected runs"},
        )
    if len(selected) == 1:
        bar.button(
            "Details",
            icon=":material/description:",
            type="tertiary",
            on_click=go_to,
            args=("Run details",),
            kwargs={"run_id": selected[0]},
        )
    tensorboard_button(bar, [by_id[i] for i in selected], settings, "exp")
    if not editor:
        return
    runs = [by_id[i] for i in selected]
    conn = db.connect(DB_PATH)
    with bar.popover("Group", icon=":material/folder:", type="tertiary"):
        choice = st.selectbox(
            "Existing group",
            ["—", *(g["name"] for g in groups)],
            key="bulk_group",
        )
        new_name = st.text_input("…or new group", key="bulk_new_group")
        if st.button("Add to group", type="primary"):
            name = new_name.strip() or (choice if choice != "—" else "")
            if name:
                gid = db.create_group(conn, name)
                db.add_to_group(conn, gid, selected)
                st.rerun()
        if group is not None and st.button(f"Remove from {group['name']}"):
            db.remove_from_group(conn, group["id"], selected)
            st.rerun()
    with bar.popover("Tag", icon=":material/sell:", type="tertiary"):
        new_tags = st.text_input("Add tags (comma separated)", key="bulk_tags")
        if st.button("Add tags", type="primary"):
            db.add_tags(conn, selected, new_tags.split(","))
            st.rerun()
        present = sorted({t for r in runs for t in r["tags"]})
        if present:
            drop = st.multiselect("Remove tags", present, key="bulk_untag")
            if drop and st.button("Remove"):
                db.remove_tags(conn, selected, drop)
                st.rerun()
    if properties:
        with bar.popover("Property", icon=":material/label:", type="tertiary"):
            by_name = {p["name"]: p for p in properties}
            name = st.selectbox("Property", list(by_name), key="bulk_prop")
            prop = by_name[name]
            value = property_input(
                prop, None, key=f"bulk_prop_value_{name}", label="Value"
            )
            left, right = st.columns(2)
            if left.button("Set", type="primary", key="bulk_prop_set"):
                try:
                    db.set_property(conn, selected, name, value)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if right.button("Clear", key="bulk_prop_clear"):
                db.set_property(conn, selected, name, None)
                st.rerun()
    all_starred = all(r["starred"] for r in runs)
    if bar.button(
        "Unstar" if all_starred else "Star",
        icon=":material/star:",
        type="tertiary",
    ):
        db.set_run_flag(conn, selected, "starred", 0 if all_starred else 1)
        st.rerun()
    all_archived = all(r["archived"] for r in runs)
    if bar.button(
        "Unarchive" if all_archived else "Archive",
        icon=":material/archive:",
        type="tertiary",
    ):
        db.set_run_flag(conn, selected, "archived", 0 if all_archived else 1)
        st.rerun()
    conn.close()


def view_compare(data: dict, visible: pl.DataFrame, editor: bool) -> None:
    """Ablation view: one selection metric, other metrics at its best step.

    Args:
        data: Output of :func:`load_data`, plus ``by_id``.
        visible: Runs that pass the sidebar filters.
        editor: Whether the visitor can edit.
    """
    metrics, groups = data["metrics"], data["groups"]
    directions, settings = data["directions"], data["settings"]
    by_id = data["by_id"]
    cards = st.container()  # Filled with the summary once it is known.
    names = [g["name"] for g in groups]
    # "Selected runs" exists once Compare was pressed in Experiments.
    chosen_ids = [
        i for i in st.session_state.get("compare_runs", []) if i in by_id
    ]
    source_options = (["Selected runs"] if chosen_ids else []) + [
        *names,
        "Runs shown in Experiments",
    ]
    sidebar_group = st.session_state.filters["group"]
    default_source = (
        sidebar_group
        if sidebar_group in names
        else "Runs shown in Experiments"
    )
    widget_state("cmp_source", source_options, default_source)
    left, right = st.columns([2, 3])
    source = left.selectbox(
        "Compare",
        source_options,
        key="cmp_source",
        format_func=lambda o: (
            f"Selected runs ({len(chosen_ids)})" if o == "Selected runs" else o
        ),
        help="A group, the runs you ticked in Experiments (Compare "
        "button), or the runs the Experiments filters show.",
    )
    prop_options = [PROP + p["name"] for p in data["properties"]]
    show_options = ["tags", "notes", "status", "groups", "steps"]
    show_options += prop_options
    widget_state("cmp_show", show_options, ["tags", "notes", *prop_options])
    shown_cols = right.multiselect(
        "Show",
        show_options,
        key="cmp_show",
        format_func=column_label,
        help=(
            "Extra columns, to check that you're comparing the runs you meant "
            "to."
        ),
    )
    show = [c for c in shown_cols if not c.startswith(PROP)]
    group = next((g for g in groups if g["name"] == source), None)
    if source == "Selected runs":
        pool = [by_id[i] for i in chosen_ids]
    elif group is not None:
        pool = [by_id[i] for i in group["members"] if i in by_id]
        if not st.session_state.filters["archived"]:
            pool = [r for r in pool if not r["archived"]]
        if group["description"]:
            st.caption(group["description"])
    else:
        pool = [by_id[i] for i in visible["id"]]
    if not pool:
        st.info(
            "This group has no runs yet. Select runs in Experiments and use "
            "Group → Add to group."
        )
        return
    pool_ids = [r["id"] for r in pool]

    options = list(tag_counts(metrics, pool_ids))
    if not options:
        st.info("These runs have no scalar metrics.")
        return
    saved_sel = group["selection_metric"] if group else None
    sel_default = (
        saved_sel if saved_sel in options else default_metrics(options, 1)[0]
    )
    c1, c2, c3 = st.columns([2, 3, 2])
    widget_state(f"sel_{source}", options, sel_default)
    sel = c1.selectbox(
        "Selection metric",
        options,
        key=f"sel_{source}",
        format_func=lambda t: arrow(t, directions),
        help="The checkpoint (step) is chosen where this metric is best. "
        "Other metrics are reported at that step, which avoids "
        "cherry-picking.",
    )
    if editor and group is not None and sel != saved_sel:
        conn = db.connect(DB_PATH)
        db.update_group(conn, group["id"], selection_metric=sel)
        conn.close()
    report_default = [
        t for t in (group["columns"] if group else []) if t in options
    ]
    report_default = [t for t in report_default if t != sel] or [
        t for t in default_metrics(options, 5) if t != sel
    ][:3]
    report_options = [t for t in options if t != sel]
    widget_state(f"report_{source}", report_options, report_default)
    report = c2.multiselect(
        "Metrics at that step",
        report_options,
        key=f"report_{source}",
        format_func=lambda t: arrow(t, directions),
    )
    aggregate = c3.toggle("Aggregate seeds", key=f"agg_{source}")
    key_mode, seed_regex = "name", settings["seed_regex"]
    if aggregate:
        a, b = st.columns([2, 3])
        key_mode = a.radio(
            "Seeds share a config when…",
            ["name", "hparams"],
            format_func={
                "name": "names match without the seed",
                "hparams": "hyperparameters match (except seed)",
            }.get,
            horizontal=True,
        )
        if key_mode == "name":
            seed_regex = b.text_input(
                "Seed pattern (regex)", value=settings["seed_regex"]
            )

    direction = an.direction_of(sel, directions)
    sel_rows = {
        m["run_id"]: m
        for m in metrics.filter(
            (pl.col("tag") == sel) & pl.col("run_id").is_in(pool_ids)
        ).iter_rows(named=True)
    }
    conn = db.connect(DB_PATH)
    rows = []
    for run in pool:
        m = sel_rows.get(run["id"])
        if m is None:
            continue
        step = an.best_step(m, direction)
        row = {
            "id": run["id"],
            "run": run["name"],
            "best step": step,
            sel: m["min"] if direction == "min" else m["max"],
        }
        for tag in report:
            series = db.load_series(conn, run["id"], tag)
            row[tag], _ = an.value_at_step(series, step or 0)
        row["config"] = (
            an.seed_key(run["name"], seed_regex)
            if key_mode == "name"
            else an.hparam_key(run["hparams"])
        )
        rows.append(row)
    conn.close()
    if not rows:
        st.info(f"None of these runs logged {sel}.")
        return
    value_cols = [sel, *report]
    df = pl.DataFrame(rows).with_columns(
        pl.col(c).cast(pl.Float64).fill_nan(None) for c in value_cols
    )
    descending = direction == "max"
    all_lists = {
        "tags": data["runs"]["tags"].to_list(),
        "groups": data["runs"]["groups"].to_list(),
    }
    labels = {c: arrow(c, directions) for c in value_cols}

    if aggregate:
        stats = (
            df.group_by("config", maintain_order=True)
            .agg(
                pl.len().alias("n"),
                pl.col("id"),
                *[pl.col(c).mean().alias(f"{c}__mean") for c in value_cols],
                *[pl.col(c).std().alias(f"{c}__std") for c in value_cols],
            )
            .sort(f"{sel}__mean", descending=descending, nulls_last=True)
        )
        members = [[by_id[i] for i in ids] for ids in stats["id"]]
        columns = {
            "config": stats["config"],
            "n": stats["n"],
            "runs": [sorted(m["name"] for m in ms) for ms in members],
        }
        columns.update(
            {col: [info_value(col, ms) for ms in members] for col in show}
        )
        prop_cols = property_columns(
            shown_cols, data["properties"], {*columns, *labels.values()}
        )
        add_property_columns(columns, prop_cols, members)
        highlight = {}
        for c in value_cols:
            means = stats[f"{c}__mean"].to_list()
            stds = stats[f"{c}__std"].fill_null(0.0).to_list()
            columns[labels[c]] = [
                f"{an.format_value(m)} ± {s:.2g}" if m is not None else ""
                for m, s in zip(means, stds)
            ]
            highlight[labels[c]] = best_rows(
                means, an.direction_of(c, directions)
            )
        shown = pl.DataFrame(columns)
        chart = {
            "label": stats["config"].to_list(),
            "value": stats[f"{sel}__mean"].to_list(),
            "err": stats[f"{sel}__std"].fill_null(0.0).to_list(),
        }
        show_table(
            styled(shown, highlight, number_formats(prop_cols)),
            hide_index=True,
            height=table_height(shown.height),
            column_config={
                **info_column_config(all_lists),
                **property_config(prop_cols),
            },
        )
        st.caption(
            f"Mean ± std over seeds of {sel} at its best step, and the "
            "other metrics at that same step."
        )
        export = shown
        best_ids = stats["id"][0].to_list() if stats.height else []
        n_configs = stats.height
    else:
        df = df.sort(sel, descending=descending, nulls_last=True)
        runs_here = [by_id[i] for i in df["id"]]
        columns = {"run": df["run"]}
        columns.update(
            {col: [info_value(col, [r]) for r in runs_here] for col in show}
        )
        prop_cols = property_columns(
            shown_cols,
            data["properties"],
            {*columns, "best step", *labels.values()},
        )
        add_property_columns(columns, prop_cols, [[r] for r in runs_here])
        columns["best step"] = df["best step"]
        columns.update({labels[c]: df[c] for c in value_cols})
        shown = pl.DataFrame(columns)
        highlight = {
            labels[c]: best_rows(
                df[c].to_list(), an.direction_of(c, directions)
            )
            for c in value_cols
        }
        formats = {labels[c]: an.format_value for c in value_cols}
        formats.update(number_formats(prop_cols))
        show_table(
            styled(shown, highlight, formats),
            hide_index=True,
            height=table_height(shown.height),
            column_config={
                **info_column_config(all_lists),
                **property_config(prop_cols),
            },
        )
        st.caption(
            "Other metrics use the logged step closest to the best step "
            f"of {sel}."
        )
        export = shown.with_columns(
            pl.col(labels[c]).map_elements(
                an.format_value, return_dtype=pl.String
            )
            for c in value_cols
        )
        chart = {
            "label": df["run"].to_list(),
            "value": df[sel].to_list(),
            "err": [0.0] * df.height,
        }
        best_ids = df["id"][:1].to_list()
        n_configs = df.height

    best_label = chart["label"][0] if chart["label"] else ""
    best_value = chart["value"][0] if chart["value"] else None
    c1, c2, c3, c4 = cards.columns(4)
    c1.metric("Runs compared", df.height, border=True)
    c2.metric("Configs", n_configs, border=True)
    c3.metric(
        f"Best {sel}",
        an.format_value(best_value) or "–",
        help=f"{'Mean over seeds, ' if aggregate else ''}at the best step.",
        border=True,
    )
    c4.metric("Best", best_label, help=best_label, border=True)

    # A dot plot, not bars: metric values rarely start at zero, and a
    # truncated bar axis would exaggerate small differences.
    fig = go.Figure(
        go.Scatter(
            x=chart["value"],
            y=chart["label"],
            mode="markers",
            marker={"color": palette()[0], "size": 11},
            error_x={
                "type": "data",
                "array": chart["err"],
                "visible": aggregate,
                "thickness": 2,
                "width": 6,
            },
            hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(180, 32 * len(chart["label"]) + 80),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title={
            "text": f"{arrow(sel, directions)} at best step",
            "font": {"size": 14},
        },
        yaxis={"autorange": "reversed", "title": None},
        xaxis={"title": None},
    )
    st.plotly_chart(fig, key="compare_chart")

    if not aggregate:
        compare_hparams(data["hparams"], df, best_ids)
    parameter_explorer(data, df, [sel, *report], directions)

    # Lists (tags, groups, runs) become comma-separated text in exports.
    export = export.with_columns(
        pl.col(name).cast(pl.List(pl.String)).list.join(", ")
        for name, dtype in export.schema.items()
        if isinstance(dtype, pl.List)
    )
    bold = {
        (row, column) for column, rows in highlight.items() for row in rows
    }
    latex = an.to_latex(export, bold)
    bar = toolbar()
    tensorboard_button(
        bar, [by_id[i] for i in df["id"]], data["settings"], "cmp"
    )
    bar.markdown(":gray[|]")
    bar.markdown(":gray[Export]")
    for label, content, ext in [
        ("CSV", export.write_csv(), "csv"),
        ("Markdown", an.to_markdown(export), "md"),
        ("LaTeX", latex, "tex"),
    ]:
        bar.download_button(
            label,
            content,
            file_name=f"{source}.{ext}",
            icon=":material/download:",
            type="tertiary",
            key=f"export_{ext}",
        )
    with bar.popover("LaTeX source", icon=":material/code:", type="tertiary"):
        st.code(latex, language="latex")
    tensorboard_panel()


def compare_hparams(
    hparams: pl.DataFrame, df: pl.DataFrame, best_ids: list[int]
) -> None:
    """Show the hyperparameters that differ between the compared runs.

    Args:
        hparams: Wide hyperparameter table of all runs.
        df: The compared runs (``id`` and ``run``), best first.
        best_ids: The best run (cells that differ from it are marked).
    """
    order = df.select("id", "run")
    hp = order.join(hparams, on="id", how="left", maintain_order="left")
    hp = hp.select(
        "id",
        "run",
        *[c for c in hp.columns[2:] if hp[c].null_count() < hp.height],
    )
    varying = an.varying_columns(hp, exclude=("id", "run"))
    if not varying or not best_ids:
        return
    st.markdown("**What differs between these runs**")
    diff = hp.select(
        "run",
        *[
            pl.col(c).map_elements(format_hparam, return_dtype=pl.String)
            for c in varying
        ],
    )
    best = diff.row(0, named=True)
    highlight = {
        c: {i for i, v in enumerate(diff[c]) if v != best[c]} for c in varying
    }
    show_table(
        styled(diff, highlight, css=cell_css("differ")), hide_index=True
    )
    st.caption(f"Highlighted: differs from the best run ({best['run']}).")


def parameter_explorer(
    data: dict, df: pl.DataFrame, targets: list[str], directions: dict
) -> None:
    """Show which settings lead to the best value of a target metric.

    Like TensorBoard's HParams dashboard: a parallel coordinates plot with
    one axis per chosen hyperparameter or property and the target last,
    one line per run colored by the target, and below it the target
    against each setting.

    Args:
        data: Output of :func:`load_data`, plus ``by_id``.
        df: The compared runs: ``id``, ``run`` and one column per metric
            (the selection metric at its best step, the others at that
            step).
        targets: Metrics that can be the target, the selection metric
            first.
        directions: User-set metric directions.
    """
    by_id = data["by_id"]
    hp = df.select("id").join(
        data["hparams"], on="id", how="left", maintain_order="left"
    )
    hp_cols = [
        c for c in hp.columns if c != "id" and hp[c].null_count() < hp.height
    ]
    prop_values = {
        p["name"]: [by_id[i]["properties"].get(p["name"]) for i in df["id"]]
        for p in data["properties"]
    }
    prop_values = {
        name: values
        for name, values in prop_values.items()
        if any(v is not None for v in values)
    }
    options = [*hp_cols, *(PROP + name for name in prop_values)]
    if not options:
        return
    varying = an.varying_columns(hp.select(hp_cols), exclude=())
    default = varying[:5] + [
        PROP + name
        for name, values in prop_values.items()
        if len({str(v) for v in values}) > 1
    ]

    st.markdown("#### Which settings lead to the best result")
    left, right = st.columns([1, 3])
    widget_state("px_target", targets, targets[0])
    target = left.selectbox(
        "Target metric",
        targets,
        key="px_target",
        format_func=lambda t: arrow(t, directions),
        help="The selection metric at its best step, or another metric "
        "at that step.",
    )
    widget_state("px_dims", options, default or options[:4])
    dims = right.multiselect(
        "Settings",
        options,
        key="px_dims",
        format_func=show_label,
        help="Hyperparameters and properties to plot, in this order.",
    )
    st.caption(
        "One line per run, from its settings to the target metric. Drag "
        "along an axis to keep only the runs in that range; drag an axis "
        "name to reorder."
    )
    if not dims:
        st.info("Choose at least one setting.")
        return

    keep = [i for i, v in enumerate(df[target].to_list()) if v is not None]
    if len(keep) < 2:
        st.info(f"Fewer than two runs logged {target}.")
        return
    names = [df["run"][i] for i in keep]
    scores = [df[target][i] for i in keep]
    axes = {}
    for dim in dims:
        if dim.startswith(PROP):
            values = prop_values[dim.removeprefix(PROP)]
        else:
            values = hp[dim].to_list()
        axes[dim] = an.explorer_axis([values[i] for i in keep])

    lower = an.direction_of(target, directions) == "min"
    # The best runs get the strongest color, whichever way the metric
    # points: darkest on a light background, palest on a dark one. The
    # end of the ramp that would fade into the background is skipped.
    ramp = SEQUENTIAL[:4][::-1] if is_dark() else SEQUENTIAL[2:]
    scale = [[i / (len(ramp) - 1), color] for i, color in enumerate(ramp)]
    best = (min if lower else max)(range(len(scores)), key=scores.__getitem__)
    straight_lines(axes, scores, arrow(target, directions), scale, lower)
    scatter_grid(names, scores, axes, target, directions, scale, lower, best)


def straight_lines(
    axes: dict, scores: list, target: str, scale: list, lower: bool
) -> None:
    """Draw the explorer's parallel coordinates plot, which can be filtered.

    Args:
        axes: :func:`analysis.explorer_axis` of each setting, by name.
        scores: The target metric of each run.
        target: Label of the target axis.
        scale: Colorscale, from worst to best.
        lower: Whether lower target values are better.
    """
    no_shadow = {"shadow": "none"}
    dimensions = []
    for dim, axis in axes.items():
        spec = {
            "label": show_label(dim).replace(" · property", ""),
            "values": axis["positions"],
        }
        if axis["tickvals"]:
            spec["tickvals"] = axis["tickvals"]
            spec["ticktext"] = axis["ticktext"]
        if axis["kind"] == "category":
            spec["range"] = [-0.3, len(axis["tickvals"]) - 0.7]
        dimensions.append(spec)
    dimensions.append({"label": target, "values": scores})
    fig = go.Figure(
        go.Parcoords(
            dimensions=dimensions,
            line={
                "color": scores,
                "colorscale": scale,
                "reversescale": lower,
                "showscale": False,
            },
            labelfont={"size": 13, **no_shadow},
            tickfont={"size": 11, **no_shadow},
            rangefont={"size": 10, **no_shadow},
            unselected={"line": {"opacity": 0.08}},
        )
    )
    fig.update_layout(height=430, margin={"l": 70, "r": 40, "t": 60, "b": 30})
    st.plotly_chart(fig, key="px_parcoords")


def scatter_grid(
    names: list,
    scores: list,
    axes: dict,
    target: str,
    directions: dict,
    scale: list,
    lower: bool,
    best: int,
) -> None:
    """Draw the target against each setting, one small plot per setting.

    Args:
        names: Run names.
        scores: The target metric of each run.
        axes: :func:`analysis.explorer_axis` of each setting, by name.
        target: The target metric.
        directions: User-set metric directions.
        scale: Colorscale, from worst to best.
        lower: Whether lower target values are better.
        best: Index of the best run.
    """
    cols = min(3, len(axes))
    rows = -(-len(axes) // cols)
    grid = make_subplots(
        rows=rows,
        cols=cols,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.16 if rows > 1 else 0.1,
        subplot_titles=[
            show_label(d).replace(" · property", "") for d in axes
        ],
    )
    for n, axis in enumerate(axes.values()):
        row, col = n // cols + 1, n % cols + 1
        x = axis["positions"]
        if axis["kind"] == "category":
            # A small, fixed jitter so runs with the same value stay apart.
            x = [p + 0.12 * ((k % 5) - 2) / 2 for k, p in enumerate(x)]
        text = [
            f"{name}<br>{label} → {an.format_value(v)}"
            for name, label, v in zip(names, axis["labels"], scores)
        ]
        grid.add_trace(
            go.Scatter(
                x=x,
                y=scores,
                mode="markers",
                text=text,
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "size": [14 if k == best else 9 for k in range(len(x))],
                    "color": scores,
                    "colorscale": scale,
                    "reversescale": lower,
                    "line": {
                        "width": [
                            2 if k == best else 0 for k in range(len(x))
                        ],
                        "color": palette()[1],
                    },
                },
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        xaxis = {"zeroline": False}
        if axis["tickvals"]:
            xaxis.update(tickvals=axis["tickvals"], ticktext=axis["ticktext"])
        if axis["kind"] == "category":
            xaxis["range"] = [-0.6, len(axis["tickvals"]) - 0.4]
        grid.update_xaxes(row=row, col=col, **xaxis)
    grid.update_yaxes(title_text=arrow(target, directions), col=1)
    grid.update_annotations(font={"size": 13})
    grid.update_layout(
        height=260 * rows + 40,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    st.plotly_chart(grid, key="px_grid")
    st.caption(
        f"{arrow(target, directions)} against each setting. The ringed dot "
        f"is the best run, {names[best]}. Log-scale settings show powers of "
        "ten; missing values are shown as –."
    )


def open_tensorboard(runs: list[dict], settings: dict) -> None:
    """Button callback: start TensorBoard on some runs.

    Args:
        runs: The runs to show (their folders are linked, not copied).
        settings: App settings (extra TensorBoard arguments and limit).
    """
    folders = {
        (ROOT.name if r["path"] == "." else r["path"]): ROOT / r["path"]
        for r in runs
        if not r["missing"]
    }
    if not folders:
        st.session_state.tb_error = "None of these run folders exist."
        return
    try:
        inst = tbview.start(
            folders,
            host=TENSORBOARD_HOST,
            extra=shlex.split(settings["tensorboard_args"]),
            limit=int(settings["max_tensorboards"]),
        )
        st.session_state.tb_opened = inst.id
    except (RuntimeError, OSError, ValueError) as exc:
        st.session_state.tb_error = str(exc)


def tensorboard_button(
    where, runs: list[dict], settings: dict, key: str, label: str = ""
) -> None:
    """Draw the "Open in TensorBoard" button.

    Args:
        where: Container to draw into (usually a toolbar).
        runs: The runs to show.
        settings: App settings.
        key: Unique suffix for the widget key.
        label: Button text; "Open in TensorBoard" by default.
    """
    where.button(
        label or "Open in TensorBoard",
        icon=":material/monitoring:",
        type="tertiary",
        key=f"tb_open_{key}",
        on_click=open_tensorboard,
        args=(runs, settings),
        help="Starts TensorBoard on just "
        + ("this run" if len(runs) == 1 else "these runs")
        + " and gives you a link to open it in a new tab.",
    )


def tensorboard_panel() -> None:
    """List the TensorBoards started from the app, with links and Stop."""
    error = st.session_state.pop("tb_error", None)
    if error:
        st.error(error)
    opened = st.session_state.pop("tb_opened", None)
    instances = tbview.running()
    for inst in instances:
        if inst.id == opened and not inst.ready():
            with st.spinner("Starting TensorBoard…"):
                ok = tbview.wait_ready(inst)
            if not ok:
                st.error(
                    "TensorBoard didn't start. Its last output:\n\n"
                    f"```\n{inst.log_tail()}\n```"
                )
                continue
        row = st.container(
            border=True,
            horizontal=True,
            gap="small",
            vertical_alignment="center",
        )
        names = ", ".join(inst.runs)
        row.markdown(
            f":material/monitoring: **TensorBoard** · {len(inst.runs)} runs · "
            f"port {inst.port} · started "
            f"{an.format_ago(inst.started, time.time())}",
            help=names,
        )
        row.link_button(
            "Open in new tab",
            inst.url,
            icon=":material/open_in_new:",
            type="primary" if inst.id == opened else "secondary",
        )
        row.button(
            "Stop",
            icon=":material/stop_circle:",
            type="tertiary",
            key=f"tb_stop_{inst.id}",
            on_click=tbview.stop,
            args=(inst.id,),
        )
    if instances:
        port = instances[-1].port
        st.caption(
            f"TensorBoard listens on {TENSORBOARD_HOST} of the machine that "
            "runs tensorboard-book, and has no password. If you reach this "
            "app through an SSH tunnel, forward that port too, e.g. "
            f"`ssh -L {port}:localhost:{port} server`. Stopped automatically "
            "when the app exits."
        )


def view_curves(data: dict, candidates: pl.DataFrame) -> None:
    """Overlay the training curves of up to eight runs.

    Args:
        data: Output of :func:`load_data`, plus ``by_id``.
        candidates: Runs to choose from (every run that isn't
            archived; the sidebar filters don't apply here).
    """
    metrics, directions, by_id = (
        data["metrics"],
        data["directions"],
        data["by_id"],
    )
    tz = viewer_tz()
    newest = candidates.sort("start_time", descending=True, nulls_last=True)
    pool = newest["id"].to_list()
    pool += [
        i
        for i in st.session_state.get("curve_runs", [])
        if i in by_id and i not in pool
    ]
    if not pool:
        st.info("No runs to plot.")
        return
    widget_state("curve_runs", pool, pool[:4])
    chosen_runs = st.multiselect(
        "Runs",
        pool,
        key="curve_runs",
        max_selections=8,
        format_func=lambda i: by_id[i]["name"],
    )
    options = list(tag_counts(metrics, chosen_runs or pool))
    widget_state("curve_metrics", options, default_metrics(options, 4))
    tags = st.multiselect(
        "Metrics",
        options,
        key="curve_metrics",
        format_func=lambda t: arrow(t, directions),
    )
    c1, c2, c3 = st.columns([3, 3, 1])
    smoothing = c1.slider("Smoothing", 0.0, 0.99, 0.6, 0.01)
    x_mode = c2.segmented_control(
        "X axis",
        ["Step", "Hours since start", "Wall time"],
        default="Step",
        required=True,
    )
    log_y = c3.toggle("Log y")
    if not chosen_runs or not tags:
        st.info("Pick at least one run and one metric.")
        return

    colors = run_colors(chosen_runs)
    conn = db.connect(DB_PATH)
    columns = st.columns(2)
    for k, tag in enumerate(tags):
        fig = go.Figure()
        for rid in chosen_runs:
            series = db.load_series(conn, rid, tag)
            if series.is_empty():
                continue
            run = by_id[rid]
            name = run["name"]
            steps = series["step"].to_numpy()
            values = series["value"].to_numpy()
            if x_mode == "Step":
                x = steps
            elif x_mode == "Hours since start":
                start = run["start_time"] or 0.0
                x = (series["wall_time"].to_numpy() - start) / 3600
            else:
                x = to_local(series["wall_time"], tz)
            if smoothing > 0:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=values,
                        mode="lines",
                        line={"color": colors[rid], "width": 1},
                        opacity=0.25,
                        legendgroup=name,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=an.ema_smooth(values, smoothing),
                    mode="lines" if len(values) > 2 else "lines+markers",
                    name=name,
                    legendgroup=name,
                    line={"color": colors[rid], "width": 2},
                    customdata=np.stack([steps, values], axis=-1),
                    hovertemplate=f"{name}<br>step %{{customdata[0]:,.0f}}"
                    "<br>value %{customdata[1]:.4g}<extra></extra>",
                )
            )
        fig.update_layout(
            title={"text": arrow(tag, directions), "font": {"size": 14}},
            height=340,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            hovermode="x unified" if x_mode != "Wall time" else "closest",
            legend={"orientation": "h", "y": -0.2},
            xaxis_title=None if x_mode == "Wall time" else x_mode.lower(),
        )
        if log_y:
            fig.update_yaxes(type="log")
        with columns[k % 2]:
            st.plotly_chart(fig, key=f"curve_{tag}")
    conn.close()


def view_timeline(data: dict, visible: pl.DataFrame, now: float) -> None:
    """Gantt timeline of when runs ran, plus a compute calendar.

    Args:
        data: Output of :func:`load_data`, plus ``by_id``.
        visible: Runs that pass the sidebar filters.
        now: Current time.
    """
    by_id = data["by_id"]
    tz = viewer_tz()
    cards = st.container()  # Filled with the summary at the end.
    c1, c2 = st.columns([3, 2])
    span = c1.segmented_control(
        "Range",
        ["7 days", "30 days", "90 days", "1 year", "All"],
        default="30 days",
        required=True,
        key="tl_range",
    )
    color_by = c2.segmented_control(
        "Color by",
        ["Group", "Status", "Tag"],
        default="Group",
        required=True,
        key="tl_color",
    )
    days = {"7 days": 7, "30 days": 30, "90 days": 90, "1 year": 365}.get(span)
    start_limit = now - days * 86400 if days else -np.inf

    group_order = [
        g["name"] for g in sorted(data["groups"], key=lambda g: g["id"])
    ]
    tag_order = sorted(
        {t for tags in data["runs"]["tags"].to_list() for t in tags}
    )
    rows = []
    for rid in visible["id"]:
        run = by_id[rid]
        if color_by == "Group":
            ranked = sorted(run["groups"], key=group_order.index)
            label = ranked[0] if ranked else "Ungrouped"
        elif color_by == "Tag":
            label = run["tags"][0] if run["tags"] else "Untagged"
        else:
            label = run["status"]
        # One continuous bar from the first to the last event of the run.
        start, end = run["start_time"], run["end_time"]
        if start is None or end < start_limit:
            continue
        shown_start = max(start, start_limit)
        rows.append(
            {
                "run_id": rid,
                "run": run["name"],
                "start": shown_start,
                "end": max(end, shown_start + 60),
                "label": label,
                "span": an.format_duration(run["wall_span"]),
                "compute": an.format_duration(run["compute_time"]),
                "status": run["status"],
            }
        )
    if not rows:
        st.info("No run activity in this range.")
        return
    df = pl.DataFrame(rows).with_columns(
        pl.Series("Start", to_local([r["start"] for r in rows], tz)),
        pl.Series("End", to_local([r["end"] for r in rows], tz)),
    )
    order = df.sort("start")["run"].unique(maintain_order=True).to_list()
    labels = sorted(df["label"].unique().to_list())
    if color_by == "Status":
        cmap = {lab: STATUS_COLOR.get(lab, OTHER_COLOR) for lab in labels}
    else:
        stable = group_order if color_by == "Group" else tag_order
        cmap = label_colors(labels, stable)
        for lab in ("Ungrouped", "Untagged"):
            if lab in cmap:
                cmap[lab] = OTHER_COLOR

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="run",
        color="label",
        color_discrete_map=cmap,
        category_orders={"run": order},
        custom_data=["run_id", "span", "compute", "status"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{base|%b %d %H:%M} → "
        "%{x|%b %d %H:%M}<br>span %{customdata[1]} · compute "
        "%{customdata[2]} · %{customdata[3]}<extra></extra>",
        marker={"cornerradius": 4},
    )
    fig.add_vline(
        x=to_local([now], tz)[0],
        line_width=1.5,
        line_dash="dot",
        line_color=STATUS_COLOR["Missing"],
    )
    fig.update_layout(
        height=max(260, 26 * len(order) + 120),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis={"title": None},
        xaxis={"title": None},
        legend={
            "title": None,
            "orientation": "h",
            "y": 1.02,
            "yanchor": "bottom",
        },
        bargap=0.3,
    )
    event = st.plotly_chart(
        fig, key="timeline", on_select="rerun", selection_mode="points"
    )
    points = event.selection.get("points", []) if event else []
    if points:
        rid = int(points[0]["customdata"][0])
        st.button(
            f"Open {by_id[rid]['name']}",
            icon=":material/open_in_new:",
            on_click=go_to,
            args=("Run details",),
            kwargs={"run_id": rid},
        )
    else:
        st.caption(
            "Each bar runs from a run's first to its last logged event. "
            "Click a bar to open that run. The dotted line is now."
        )

    conc = an.concurrency([[r["start"], r["end"]] for r in rows])
    left, right = st.columns([3, 2])
    fig2 = go.Figure(
        go.Scatter(
            x=to_local(conc["time"], tz),
            y=conc["running"].to_list(),
            mode="lines",
            line={"shape": "hv", "width": 2, "color": palette()[0]},
            fill="tozeroy",
            hovertemplate="%{x|%b %d %H:%M}<br>%{y} running<extra></extra>",
        )
    )
    fig2.update_layout(
        title={"text": "Runs at once", "font": {"size": 14}},
        height=240,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis={"rangemode": "tozero", "dtick": 1},
    )
    left.plotly_chart(fig2, key="concurrency")

    # Compute hours use the activity segments, so pauses don't count.
    segments = [
        [max(seg_start, start_limit), seg_end]
        for r in rows
        for seg_start, seg_end in by_id[r["run_id"]]["segments"]
        if seg_end >= start_limit
    ]
    hours = an.daily_hours(segments, tz)
    right.plotly_chart(compute_calendar(hours, now, days, tz), key="calendar")
    total = sum(hours.values())
    busiest = max(hours, key=hours.get) if hours else None
    c1, c2, c3, c4 = cards.columns(4)
    c1.metric("Runs in range", len(rows), border=True)
    c2.metric(
        "Compute",
        f"{total:.1f} h",
        help="Pauses are not counted.",
        border=True,
    )
    c3.metric(
        "Busiest day",
        f"{busiest:%a %b %d}" if busiest else "–",
        help=f"{hours[busiest]:.1f} compute hours" if busiest else None,
        border=True,
    )
    c4.metric(
        "Running now", sum(r["status"] == "Active" for r in rows), border=True
    )
    st.caption(
        "Times are shown in your time zone and come from the clock of the "
        "machine that ran each job."
    )


def compute_calendar(hours: dict, now: float, days: int | None, tz):
    """Draw compute hours per day as a week-by-weekday heatmap.

    Args:
        hours: Mapping of date to compute hours.
        now: Current time.
        days: Days shown (None for everything with data).
        tz: Viewer time zone.

    Returns:
        A plotly Figure.
    """
    today = datetime.fromtimestamp(now, tz).date()
    if days:
        first = today - timedelta(days=days - 1)
    else:
        first = min(hours) if hours else today
    first = first - timedelta(days=first.weekday())
    dates = [
        first + timedelta(days=i) for i in range((today - first).days + 1)
    ]
    weeks = sorted({d - timedelta(days=d.weekday()) for d in dates})
    z = np.full((7, len(weeks)), np.nan)
    text = np.full((7, len(weeks)), "", dtype=object)
    for d in dates:
        col = weeks.index(d - timedelta(days=d.weekday()))
        z[d.weekday(), col] = hours.get(d, 0.0)
        text[d.weekday(), col] = (
            f"{d:%a %b %d}: {hours.get(d, 0.0):.1f} compute h"
        )
    dark = is_dark()
    ramp = SEQUENTIAL[::-1] if dark else SEQUENTIAL
    empty = "#2C2A26" if dark else "#ECE9E1"
    scale = [[0.0, empty], [1e-6, ramp[0]]]
    scale += [
        [i / (len(ramp) - 1), color] for i, color in enumerate(ramp) if i
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[w.strftime("%b %d") for w in weeks],
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=scale,
            zmin=0,
            xgap=2,
            ygap=2,
            text=text,
            hovertemplate="%{text}<extra></extra>",
            colorbar={"title": "h", "thickness": 10},
        )
    )
    fig.update_layout(
        title={"text": "Compute hours per day", "font": {"size": 14}},
        height=240,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis={"autorange": "reversed"},
    )
    return fig


def view_run_details(
    data: dict, candidates: pl.DataFrame, editor: bool, now: float
) -> None:
    """Everything about one run: timing, annotations, metrics and files.

    Args:
        data: Output of :func:`load_data`, plus ``by_id``.
        candidates: Runs to choose from (every run that isn't
            archived; the sidebar filters don't apply here).
        editor: Whether the visitor can edit.
        now: Current time.
    """
    metrics, groups = data["metrics"], data["groups"]
    directions, settings, by_id = (
        data["directions"],
        data["settings"],
        data["by_id"],
    )
    tz = viewer_tz()
    newest = candidates.sort("start_time", descending=True, nulls_last=True)
    options = newest["id"].to_list()
    current = st.session_state.get("run_id")
    if current in by_id and current not in options:
        options.insert(0, current)
    if not options:
        st.info("No runs match the filters.")
        return
    widget_state("run_id", options, options[0])
    rid = st.selectbox(
        "Run", options, key="run_id", format_func=lambda i: by_id[i]["name"]
    )
    run = by_id[rid]
    run_dir = ROOT / run["path"]
    all_tags = data["runs"]["tags"].to_list()

    # Status, flags, groups, properties and tags as one row of soft pills,
    # spaced like modelboard's, instead of badges packed edge to edge.
    status_color = {"Active": "green", "Stopped": "gray", "Missing": "red"}
    items = [(html.escape(run["status"]), status_color[run["status"]])]
    if run["starred"]:
        items.append(("★ starred", "orange"))
    if run["archived"]:
        items.append(("archived", "gray"))
    items += [(html.escape(g), "blue") for g in run["groups"]]
    if run["has_nonfinite"]:
        items.append(("NaN/inf logged", "red"))
    for name, value in sorted(run["properties"].items()):
        shown = an.format_value(value) if isinstance(value, float) else value
        items.append(
            (
                f'<span class="tb-muted">{html.escape(name)}</span> '
                f"{html.escape(str(shown))}",
                "violet",
            )
        )
    tag_colors = name_colors(all_tags)
    items += [(html.escape(t), tag_colors.get(t, "gray")) for t in run["tags"]]
    pills = "".join(
        f'<span class="tb-pill {color}">{text}</span>' for text, color in items
    )
    # Badges on the left, the edit button on the right, the note below:
    # everything about the run's annotations in two lines.
    row = st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    )
    row.markdown(
        f'<span class="tb-pills tb-run-pills">{pills}</span>',
        unsafe_allow_html=True,
    )
    actions = row.container(
        horizontal=True,
        horizontal_alignment="right",
        vertical_alignment="center",
        gap="small",
        width="content",
    )
    tensorboard_button(actions, [run], settings, "run", "TensorBoard")
    if editor:
        with actions.popover(
            "Edit tags, notes and properties", icon=":material/edit:"
        ):
            annotation_form(run, groups, all_tags, data["properties"])
    if run["notes"]:
        note = run["notes"].strip().replace("\n", " ")
        st.markdown(f":material/sticky_note_2: {note}")
    elif editor:
        st.caption("No notes yet. Use Edit to add tags, notes and properties.")
    tensorboard_panel()
    if run["parse_error"]:
        st.warning(
            f"Some event data could not be read:\n\n{run['parse_error']}"
        )

    c = st.columns(6)
    c[0].metric(
        "Compute time", an.format_duration(run["compute_time"]), border=True
    )
    c[1].metric(
        "Wall span",
        an.format_duration(run["wall_span"]),
        help=f"{run['n_segments']} activity segment(s)",
        border=True,
    )
    steps_per_s = (
        (run["max_step"] - run["min_step"]) / run["compute_time"]
        if run["compute_time"] and run["max_step"] is not None
        else None
    )
    c[2].metric(
        "Throughput",
        f"{steps_per_s:.2f} it/s" if steps_per_s else "–",
        border=True,
    )
    last_step = run["max_step"]
    c[3].metric(
        "Last step",
        f"{last_step:,}" if last_step is not None else "–",
        border=True,
    )
    c[4].metric("Last event", an.format_ago(run["end_time"], now), border=True)
    c[5].metric("Disk", an.format_bytes(run["total_bytes"]), border=True)
    started, ended = to_local([run["start_time"], run["end_time"]], tz)
    st.caption(
        f"`{run_dir}` · started {started:%Y-%m-%d %H:%M} · last event "
        f"{ended:%Y-%m-%d %H:%M}"
    )

    tabs = st.tabs(
        ["Artifacts", "Metrics", "Hyperparameters", "Text", "Timing"]
    )
    with tabs[0]:
        artifacts_tab(run, run_dir, by_id, settings, tz)
    with tabs[1]:
        metrics_tab(metrics.filter(pl.col("run_id") == rid), run, directions)
        st.code(f'tensorboard --logdir "{run_dir}"', language="bash")
    with tabs[2]:
        hp = run["hparams"]
        if hp:
            show_table(
                pl.DataFrame(
                    {
                        "key": list(hp),
                        "value": [format_hparam(v) for v in hp.values()],
                    }
                ),
                hide_index=True,
            )
            st.caption("Sources: " + ", ".join(run["hparam_sources"]))
        else:
            st.info(
                "No hyperparameters found in the events or in config files "
                "(config.yaml, hparams.yaml, args.json, …)."
            )
    with tabs[3]:
        if run["texts"]:
            for tag, text in run["texts"].items():
                st.markdown(f"**{tag}**")
                with st.container(border=True):
                    st.markdown(text)
        else:
            st.info("No text summaries.")
    with tabs[4]:
        segments = run["segments"]
        if segments:
            starts = [s for s, _ in segments]
            ends = [e for _, e in segments]
            pauses = [b - a for a, b in zip(ends, starts[1:])] + [None]
            show_table(
                pl.DataFrame(
                    {
                        "start": to_local(starts, tz),
                        "end": to_local(ends, tz),
                        "duration": [
                            an.format_duration(e - s) for s, e in segments
                        ],
                        "pause after": [an.format_duration(p) for p in pauses],
                    }
                ),
                hide_index=True,
            )
        st.caption(
            f"A silence longer than {settings['gap_minutes']} minutes "
            "starts a new segment. Compute time is the sum of the segments; "
            "wall span includes the pauses."
        )


def annotation_form(
    run: dict, groups: list[dict], all_tags, properties: list[dict]
) -> None:
    """Form to edit a run's tags, groups, flags, notes and properties.

    Args:
        run: The run, as a dict.
        groups: All groups.
        all_tags: The tags of every run (a list of lists).
        properties: All custom property definitions.
    """
    rid = run["id"]
    names = {g["id"]: g["name"] for g in groups}
    with st.form(f"annotate_{rid}", border=False):
        tag_options = sorted({t for tags in all_tags for t in tags})
        tags = st.multiselect("Tags", tag_options, default=run["tags"])
        new_tags = st.text_input("New tags (comma separated)")
        group_ids = st.multiselect(
            "Groups",
            list(names),
            default=[g["id"] for g in groups if rid in g["members"]],
            format_func=names.get,
        )
        new_group = st.text_input("New group")
        flags = st.container(horizontal=True, gap="medium")
        starred = flags.checkbox("Starred", value=bool(run["starred"]))
        archived = flags.checkbox(
            "Archived",
            value=bool(run["archived"]),
            help="Archived runs are hidden unless you turn on Show archived. "
            "Their files are untouched.",
        )
        notes = st.text_area("Notes", value=run["notes"], height=100)
        values = {}
        if properties:
            st.markdown("**Properties**")
            cells = st.columns(2)
            for k, prop in enumerate(properties):
                with cells[k % 2]:
                    values[prop["name"]] = property_input(
                        prop,
                        run["properties"].get(prop["name"]),
                        key=f"prop_{rid}_{prop['name']}",
                    )
        else:
            st.caption("Add custom properties in Manage → Properties.")
        if st.form_submit_button("Save", type="primary"):
            conn = db.connect(DB_PATH)
            if new_group.strip():
                group_ids.append(db.create_group(conn, new_group))
            db.set_run_tags(conn, rid, tags + new_tags.split(","))
            db.set_run_groups(conn, rid, group_ids)
            db.set_run_flag(conn, [rid], "starred", int(starred))
            db.set_run_flag(conn, [rid], "archived", int(archived))
            db.set_run_flag(conn, [rid], "notes", notes)
            try:
                for name, value in values.items():
                    db.set_property(conn, [rid], name, value)
            except ValueError as exc:
                st.error(str(exc))
                conn.close()
                return
            conn.close()
            st.rerun()


def metrics_tab(metrics: pl.DataFrame, run: dict, directions: dict) -> None:
    """Table of every scalar of one run with best, last, min and max.

    Args:
        metrics: The run's rows of the metrics table.
        run: The run, as a dict.
        directions: User-set directions.
    """
    if metrics.is_empty():
        st.info("No scalar metrics.")
        return
    rows = []
    for m in metrics.iter_rows(named=True):
        lower = an.direction_of(m["tag"], directions) == "min"
        rows.append(
            {
                "tag": arrow(m["tag"], directions),
                "best": m["min"] if lower else m["max"],
                "best step": m["step_min"] if lower else m["step_max"],
                "last": m["last"],
                "last step": m["last_step"],
                "min": m["min"],
                "max": m["max"],
                "points": m["n"],
                "NaN/inf": m["nonfinite"],
            }
        )
    table = pl.DataFrame(rows)
    formats = {c: an.format_value for c in ("best", "last", "min", "max")}
    show_table(styled(table, formats=formats), hide_index=True)
    if run["other_tags"]:
        found = ", ".join(
            f"{n} {plugin} tag(s)" for plugin, n in run["other_tags"].items()
        )
        st.caption(f"Also in the logs (open TensorBoard to view): {found}")


def artifacts_tab(
    run: dict, run_dir: Path, by_id: dict, settings: dict, tz
) -> None:
    """File browser for one run: preview, download, zip and diff.

    Args:
        run: The run, as a dict.
        run_dir: Absolute run folder.
        by_id: Every run as a dict (for diffing against another run).
        settings: App settings (zip size cap).
        tz: Viewer time zone.
    """
    if not run_dir.is_dir():
        st.warning("The run folder no longer exists.")
        return
    files = list_artifacts(run_dir)
    if files.is_empty():
        st.info("This run has only event files.")
        return
    left, right = st.columns([3, 1])
    needle = left.text_input(
        "Filter files", key="art_filter", placeholder="checkpoints/ or .png"
    )
    hide_ckpt = right.toggle("Hide checkpoints", key="art_hide_ckpt")
    if needle:
        files = files.filter(
            pl.col("path")
            .str.to_lowercase()
            .str.contains(needle.lower(), literal=True)
        )
    if hide_ckpt:
        files = files.filter(~pl.col("path").str.contains(CHECKPOINT_PATTERN))
    st.caption(
        f"{files.height} files · {an.format_bytes(files['size'].sum())} · "
        "event files hidden"
    )
    shown = files.with_columns(
        pl.col("size").map_elements(an.format_bytes, return_dtype=pl.String),
        pl.Series("modified", to_local(files["modified"], tz), pl.Datetime),
    )
    event = show_table(
        shown,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"art_table_{run['id']}",
        height=table_height(shown.height, 423),
        column_config={
            "modified": st.column_config.DatetimeColumn(
                "modified", format="YYYY-MM-DD HH:mm"
            )
        },
    )
    rows = [r for r in event.selection.rows if r < files.height]
    bar = toolbar()
    if not rows:
        bar.caption(
            "Select a file to preview and download it, or several for a zip."
        )
        return
    picked = files[rows]
    bar.badge(f"{picked.height} selected", color="blue")
    bar.markdown(":gray[|]")
    folder = Path(run["path"]).name

    if picked.height > 1:
        size = picked["size"].sum()
        if size > settings["max_zip_mb"] * 1024**2:
            bar.caption(
                f"The selection is {an.format_bytes(size)}, above the "
                f"{settings['max_zip_mb']} MB zip limit (Manage → Settings)."
            )
            return

        def build_zip() -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(
                buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1
            ) as zf:
                for rel in picked["path"]:
                    path = safe_file(run_dir, rel)
                    if path is not None:
                        zf.write(path, arcname=f"{folder}/{rel}")
            return buf.getvalue()

        bar.download_button(
            f"Download zip ({an.format_bytes(size)})",
            data=build_zip,
            file_name=f"{folder}_artifacts.zip",
            mime="application/zip",
            icon=":material/download:",
            type="tertiary",
        )
        return

    rel = picked["path"][0]
    path = safe_file(run_dir, rel)
    if path is None:
        bar.caption(
            "This file is outside the runs folder (e.g. a symlink) and "
            "won't be served."
        )
        return
    size = path.stat().st_size
    bar.download_button(
        f"Download {Path(rel).name} ({an.format_bytes(size)})",
        data=lambda: path.read_bytes(),
        file_name=Path(rel).name,
        icon=":material/download:",
        type="tertiary",
    )
    preview_file(path, rel, size, run, by_id)


def preview_file(
    path: Path, rel: str, size: int, run: dict, by_id: dict
) -> None:
    """Show a file: image, audio, video, table, or text with a diff.

    Args:
        path: The resolved file.
        rel: Its path relative to the run.
        size: File size in bytes.
        run: The run it belongs to.
        by_id: Every run as a dict (for the diff).
    """
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXT:
        st.image(str(path))
        return
    if suffix in AUDIO_EXT:
        st.audio(str(path))
        return
    if suffix in VIDEO_EXT:
        st.video(str(path))
        return
    if suffix in TABLE_EXT:
        try:
            st.dataframe(
                pl.read_csv(
                    path,
                    separator="\t" if suffix == ".tsv" else ",",
                    n_rows=2000,
                    infer_schema_length=2000,
                ),
                hide_index=True,
            )
        except (pl.exceptions.PolarsError, OSError) as exc:
            st.error(f"Can't read as a table: {exc}")
        return
    text = read_text(path)
    if text is None:
        st.info("Binary file. Download it to inspect.")
        return
    if size > MAX_PREVIEW_BYTES:
        st.caption(f"Showing the first {an.format_bytes(MAX_PREVIEW_BYTES)}.")
    st.code(text, language=CODE_LANG.get(suffix, "text"), line_numbers=True)
    others = [
        i
        for i, r in by_id.items()
        if i != run["id"]
        and not r["missing"]
        and (ROOT / r["path"] / rel).is_file()
    ]
    if not others:
        return
    other = st.selectbox(
        f"Diff {rel} with the same file in",
        [None, *others],
        format_func=lambda i: "—" if i is None else by_id[i]["name"],
        key=f"diff_{run['id']}_{rel}",
    )
    if other is None:
        return
    other_path = safe_file(ROOT / by_id[other]["path"], rel)
    other_text = read_text(other_path) if other_path else None
    if other_text is None:
        st.info("The other file is binary or unreadable.")
        return
    diff = "".join(
        difflib.unified_diff(
            other_text.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=f"{by_id[other]['name']}/{rel}",
            tofile=f"{run['name']}/{rel}",
        )
    )
    st.code(diff or "Files are identical.", language="diff")


def view_manage(data: dict, editor: bool) -> None:
    """Groups, tags, metric directions, disk usage, settings and backups.

    Args:
        data: Output of :func:`load_data`, plus ``by_id``.
        editor: Whether the visitor can edit.
    """
    sections = ["Disk usage", "Backup"]
    if editor:
        sections = [
            "Groups",
            "Tags",
            "Properties",
            "Metric directions",
            "Disk usage",
            "Settings",
            "Backup",
        ]
    section = st.segmented_control(
        "Section",
        sections,
        default=sections[0],
        required=True,
        key="manage_section",
    )
    conn = db.connect(DB_PATH)
    {
        "Groups": manage_groups,
        "Tags": manage_tags,
        "Properties": manage_properties,
        "Metric directions": manage_directions,
        "Disk usage": manage_disk,
        "Settings": manage_settings,
        "Backup": manage_backup,
    }[section](conn, data, editor)
    conn.close()


def manage_groups(conn, data: dict, editor: bool) -> None:
    """Create, rename, describe, edit members of and delete groups."""
    groups, by_id = data["groups"], data["by_id"]
    with st.form("new_group", clear_on_submit=True):
        a, b = st.columns([1, 2])
        name = a.text_input("New group name")
        description = b.text_input(
            "Description", placeholder="What this ablation tests"
        )
        if st.form_submit_button("Create group") and name.strip():
            db.create_group(conn, name, description)
            st.rerun()
    if not groups:
        st.info(
            "No groups yet. Create one here or select runs in Experiments "
            "→ Group."
        )
        return
    show_table(
        pl.DataFrame(
            {
                "group": [g["name"] for g in groups],
                "runs": [len(g["members"]) for g in groups],
                "selection metric": [
                    g["selection_metric"] or "" for g in groups
                ],
                "compute": [
                    an.format_duration(
                        sum(
                            by_id[i]["compute_time"] or 0
                            for i in g["members"]
                            if i in by_id
                        )
                    )
                    for g in groups
                ],
                "description": [g["description"] for g in groups],
            }
        ),
        hide_index=True,
    )
    by_gid = {g["id"]: g for g in groups}
    gid = st.selectbox(
        "Edit group", list(by_gid), format_func=lambda i: by_gid[i]["name"]
    )
    g = by_gid[gid]
    with st.form(f"edit_group_{gid}"):
        new_name = st.text_input("Name", value=g["name"])
        desc = st.text_input("Description", value=g["description"])
        members = st.multiselect(
            "Members",
            list(by_id),
            default=[m for m in g["members"] if m in by_id],
            format_func=lambda i: by_id[i]["name"],
        )
        if st.form_submit_button("Save group", type="primary"):
            db.update_group(
                conn,
                gid,
                name=new_name.strip() or g["name"],
                description=desc,
            )
            removed = [m for m in g["members"] if m not in members]
            db.remove_from_group(conn, gid, removed)
            db.add_to_group(conn, gid, members)
            st.rerun()
    confirm = st.checkbox(
        f"Yes, delete the group {g['name']!r} (its runs stay)"
    )
    if st.button(
        "Delete group", disabled=not confirm, icon=":material/delete:"
    ):
        db.delete_group(conn, gid)
        st.rerun()


def manage_tags(conn, data: dict, editor: bool) -> None:
    """Rename, merge and delete tags."""
    all_tags = data["runs"]["tags"].to_list()
    counts = Counter(t for tags in all_tags for t in tags).most_common()
    if not counts:
        st.info("No tags yet.")
        return
    show_table(
        pl.DataFrame(
            {
                "tag": [[t] for t, _ in counts],
                "runs": [n for _, n in counts],
            }
        ),
        hide_index=True,
        column_config={"tag": pill_column("tag", all_tags)},
    )
    a, b, c = st.columns([2, 2, 1])
    old = a.selectbox("Tag", [t for t, _ in counts])
    new = b.text_input("Rename to (an existing tag merges)")
    c.write("")
    if c.button("Rename") and new.strip():
        db.rename_tag(conn, old, new)
        st.rerun()
    confirm = st.checkbox(f"Yes, remove the tag {old!r} from all runs")
    if st.button("Delete tag", disabled=not confirm):
        db.delete_tag(conn, old)
        st.rerun()


def manage_properties(conn, data: dict, editor: bool) -> None:
    """Define custom properties; rename, describe or delete them."""
    props = data["properties"]
    with st.form("new_property", clear_on_submit=True):
        a, b, c = st.columns([2, 1, 3])
        name = a.text_input("New property", placeholder="dataset")
        kind = b.selectbox("Kind", ["category", "number"])
        description = c.text_input(
            "Description", placeholder="What it means, units"
        )
        if st.form_submit_button("Add property"):
            try:
                db.create_property(conn, name, kind, description)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.caption(
        "Properties are values you add by hand: a **number** (e.g. GPU "
        "hours, a human rating) or a **category** (e.g. dataset, the person "
        "who ran it). Set them in Run details → Edit, or for many runs at "
        "once in Experiments (select rows → Property). Show them as columns "
        "with **Show** in Experiments and Compare."
    )
    if not props:
        st.info("No properties yet.")
        return
    show_table(
        pl.DataFrame(
            {
                "property": [p["name"] for p in props],
                "kind": [p["kind"] for p in props],
                "runs": [p["runs"] for p in props],
                "values": [
                    ", ".join(
                        an.format_value(v) if isinstance(v, float) else v
                        for v in p["values"][:12]
                    )
                    + (" …" if len(p["values"]) > 12 else "")
                    for p in props
                ],
                "description": [p["description"] for p in props],
            }
        ),
        hide_index=True,
    )
    by_name = {p["name"]: p for p in props}
    name = st.selectbox("Edit property", list(by_name), key="edit_prop")
    prop = by_name[name]
    with st.form(f"edit_property_{name}"):
        a, b = st.columns([1, 2])
        new_name = a.text_input("Name", value=prop["name"])
        desc = b.text_input("Description", value=prop["description"])
        if st.form_submit_button("Save property", type="primary"):
            try:
                db.update_property(conn, name, new_name, desc)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    confirm = st.checkbox(
        f"Yes, delete {name!r} and its value on {prop['runs']} runs"
    )
    if st.button(
        "Delete property", disabled=not confirm, icon=":material/delete:"
    ):
        db.delete_property(conn, name)
        st.rerun()


def manage_directions(conn, data: dict, editor: bool) -> None:
    """Choose whether higher or lower is better for each metric."""
    directions = data["directions"]
    runs_per_tag = dict(
        data["metrics"].group_by("tag").len().sort("tag").iter_rows()
    )
    tags = list(runs_per_tag)
    word = {"max": "higher", "min": "lower"}
    # "auto" follows the guess from the name; higher/lower pin a choice.
    # Offering "auto" makes choosing "higher" on a metric guessed as higher
    # a real change, so it is saved like any other.
    table = pl.DataFrame(
        {
            "metric": tags,
            "better": [word.get(directions.get(t), "auto") for t in tags],
            "runs": [runs_per_tag[t] for t in tags],
        }
    )
    edited = st.data_editor(
        table,
        hide_index=True,
        disabled=["metric", "runs"],
        column_config=headers(
            table.columns,
            {
                "better": st.column_config.SelectboxColumn(
                    "better",
                    options=["auto", "higher", "lower"],
                    required=True,
                    help="auto uses the guess from the name; higher or lower "
                    "fixes it.",
                )
            },
        ),
        key="directions_editor",
    )
    touched = st.session_state.directions_editor.get("edited_rows", {})
    buttons = st.container(horizontal=True, gap="small")
    if buttons.button("Save directions", type="primary", disabled=not touched):
        rows = (
            edited.to_dicts()
            if isinstance(edited, pl.DataFrame)
            else edited.to_dict("records")
        )
        chosen = {r["metric"]: r["better"] for r in rows}
        db.set_directions(
            conn,
            {
                t: "min" if b == "lower" else "max"
                for t, b in chosen.items()
                if b != "auto"
            },
        )
        db.clear_directions(
            conn, [t for t, b in chosen.items() if b == "auto"]
        )
        del st.session_state.directions_editor
        st.rerun()
    if directions and buttons.button(
        "Reset all to auto",
        type="tertiary",
        icon=":material/restart_alt:",
        help="Forget every direction you set; each metric goes back to "
        "the guess from its name.",
    ):
        db.clear_directions(conn)
        del st.session_state.directions_editor
        st.rerun()
    st.caption(
        "Names containing loss, error, mse, … default to lower is better."
    )


def manage_disk(conn, data: dict, editor: bool) -> None:
    """Disk usage per run and per group."""
    runs, by_id = data["runs"], data["by_id"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", an.format_bytes(runs["total_bytes"].sum()), border=True)
    c2.metric(
        "Artifacts", an.format_bytes(runs["artifact_bytes"].sum()), border=True
    )
    archived = runs.filter(pl.col("archived") == 1)["total_bytes"].sum()
    c3.metric("Archived runs", an.format_bytes(archived), border=True)
    per_run = runs.sort("total_bytes", descending=True, nulls_last=True)
    show_table(
        per_run.select(
            pl.col("name").alias("run"),
            pl.col("total_bytes")
            .map_elements(an.format_bytes, return_dtype=pl.String)
            .alias("total"),
            pl.col("artifact_bytes")
            .map_elements(an.format_bytes, return_dtype=pl.String)
            .alias("artifacts"),
            pl.col("n_artifacts").alias("files"),
            "status",
            pl.col("archived").cast(pl.Boolean),
        ),
        hide_index=True,
    )
    if data["groups"]:
        sizes = sorted(
            (
                sum(
                    by_id[i]["total_bytes"] or 0
                    for i in g["members"]
                    if i in by_id
                ),
                g["name"],
            )
            for g in data["groups"]
        )
        fig = go.Figure(
            go.Bar(
                x=[b / 1024**2 for b, _ in sizes],
                y=[name for _, name in sizes],
                orientation="h",
                marker={"color": palette()[0], "cornerradius": 4},
                hovertemplate="%{y}: %{x:.1f} MB<extra></extra>",
            )
        )
        fig.update_layout(
            title={"text": "Disk usage per group (MB)", "font": {"size": 14}},
            height=max(160, 30 * len(sizes) + 80),
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        st.plotly_chart(fig, key="disk_groups")
    st.caption(
        "Archiving hides a run without touching its files. tensorboard-book "
        "never deletes files."
    )


def manage_settings(conn, data: dict, editor: bool) -> None:
    """Thresholds, seed pattern and limits."""
    settings = data["settings"]
    with st.form("settings"):
        a, b = st.columns(2)
        active = a.number_input(
            "Active if written to within (minutes)",
            1,
            100_000,
            int(settings["active_minutes"]),
            help="A run counts as active while its event files keep being "
            "written. Set this above your longest time between two logged "
            "values (e.g. one epoch). Default: 300 (5 h).",
        )
        gap = b.number_input(
            "Pause that splits a run into segments (minutes)",
            1,
            100_000,
            int(settings["gap_minutes"]),
            help="A silence longer than this counts as a pause (e.g. a "
            "crashed job resumed later) and is left out of compute time. "
            "Keep it above your longest time between two logged values. "
            "Default: 360 (6 h).",
        )
        seed = a.text_input(
            "Seed pattern in run names (regex)", settings["seed_regex"]
        )
        points = b.number_input(
            "Max stored points per curve (0 = all)",
            0,
            10_000_000,
            int(settings["max_points"]),
        )
        zip_mb = a.number_input(
            "Max zip download (MB)",
            1,
            1_000_000,
            int(settings["max_zip_mb"]),
        )
        max_tb = b.number_input(
            "Max TensorBoards at once",
            1,
            50,
            int(settings["max_tensorboards"]),
            help="Each one is a separate process using memory.",
        )
        tb_args = st.text_input(
            "Extra TensorBoard arguments",
            settings["tensorboard_args"],
            placeholder="--samples_per_plugin images=100 --reload_interval 30",
            help="Added to every TensorBoard started from Compare.",
        )
        if st.form_submit_button("Save settings", type="primary"):
            reparse = (gap, points) != (
                settings["gap_minutes"],
                settings["max_points"],
            )
            db.save_settings(
                conn,
                {
                    "active_minutes": active,
                    "gap_minutes": gap,
                    "seed_regex": seed,
                    "max_points": points,
                    "max_zip_mb": zip_mb,
                    "max_tensorboards": max_tb,
                    "tensorboard_args": tb_args,
                },
            )
            if reparse:
                run_scan()  # The changed settings re-parse every run.
            st.rerun()
    st.caption(
        "Changing the pause or the stored points re-parses every run, since "
        "both change what is computed from the event files."
    )
    if st.button("Re-parse every run", icon=":material/sync:"):
        counts = run_scan(force=True)
        st.toast(
            f"Re-parsed {counts['parsed']} runs"
            if counts
            else "A scan is already running."
        )
        st.rerun()


def manage_backup(conn, data: dict, editor: bool) -> None:
    """Download and import annotations; forget runs whose folder is gone."""
    st.markdown(
        "Your tags, notes, groups and settings are saved automatically "
        f"after every edit to `{ANNOTATIONS_PATH}`. Commit it to git or copy "
        "it somewhere safe. If the database is deleted, the next start "
        "restores annotations from it."
    )

    def annotations_json() -> str:
        export_conn = db.connect(DB_PATH)
        text = json.dumps(db.export_annotations(export_conn), indent=2)
        export_conn.close()
        return text

    toolbar().download_button(
        "Download annotations (JSON)",
        data=annotations_json,
        file_name="annotations.json",
        mime="application/json",
        icon=":material/download:",
        type="tertiary",
    )
    if not editor:
        return
    upload = st.file_uploader(
        "Import annotations (merges into the current ones)", type=["json"]
    )
    if upload is not None and st.button("Import"):
        try:
            result = db.import_annotations(conn, json.loads(upload.getvalue()))
            st.success(
                f"Imported: {result['matched']} runs matched, "
                f"{result['skipped']} not found."
            )
        except ValueError as exc:
            st.error(str(exc))
    missing = data["runs"]["missing"].sum()
    st.divider()
    st.markdown(
        f"**{missing} missing runs** (folder deleted or moved outside the "
        "root)."
    )
    if missing:
        confirm = st.checkbox(
            "Yes, forget them and their annotations (files are not touched)"
        )
        if st.button("Forget missing runs", disabled=not confirm):
            db.forget_missing_runs(conn)
            st.rerun()


# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------


def main() -> None:
    """Log in, sync the database with the folder, and draw the view."""
    st.set_page_config(
        page_title="tensorboard-book",
        page_icon=str(ASSETS / "logo.png"),
        layout="wide",
    )
    st.html(STYLE)
    role = auth.require_login(header=brand_header)
    editor = role == "editor"

    conn = db.connect(DB_PATH)
    if db.get_meta(conn, "annotations_path") != str(ANNOTATIONS_PATH):
        db.set_meta(conn, "annotations_path", str(ANNOTATIONS_PATH))
        conn.commit()
    if not st.session_state.get("scanned"):
        # The CLI already did the first full scan; this picks up changes.
        run_scan()
        st.session_state.scanned = True
        st.session_state.last_scan = time.time()
    data = dict(load_data(str(DB_PATH), db.data_version(conn)))
    conn.close()

    now = time.time()
    active_minutes = data["settings"]["active_minutes"]
    runs = data["runs"]
    status = [
        an.run_status(end, mtime, missing, now, active_minutes)
        for end, mtime, missing in runs.select(
            "end_time", "last_file_mtime", "missing"
        ).iter_rows()
    ]
    runs = runs.with_columns(pl.Series("status", status, pl.String))
    data["runs"] = runs
    # Every run as a dict (row plus decoded JSON fields), for lookups.
    data["by_id"] = {
        r["id"]: {
            **r,
            **data["details"][r["id"]],
            "properties": data["run_properties"].get(r["id"], {}),
        }
        for r in runs.iter_rows(named=True)
    }
    if st.session_state.get("view") not in VIEWS:
        st.session_state.view = "Experiments"
    if "filters" not in st.session_state:
        st.session_state.filters = json.loads(json.dumps(FILTER_DEFAULTS))
    sidebar(runs, data["groups"], editor, now)
    # The runs that pass the sidebar filters (shown in Experiments).
    visible = apply_filters(runs, data["by_id"], st.session_state.filters)
    # Views without filters list every run that isn't archived.
    unarchived = runs.filter(pl.col("archived") == 0)

    if runs.is_empty():
        st.info(
            f"No TensorBoard runs found in `{ROOT}` at depth {DEPTH}. A run "
            "is a folder containing `events.out.tfevents.*` files. Try "
            "`--depth 2` if your runs are nested."
        )
        return
    view = st.session_state.view
    page_header(view)
    if st.session_state.pop("help_open", False):
        help_dialog(st.session_state.get("help_topic", view))
    if view == "Experiments":
        view_experiments(data, visible, editor, now)
    elif view == "Compare":
        view_compare(data, visible, editor)
    elif view == "Curves":
        view_curves(data, unarchived)
    elif view == "Timeline":
        view_timeline(data, visible, now)
    elif view == "Run details":
        view_run_details(data, unarchived, editor, now)
    else:
        view_manage(data, editor)


main()
