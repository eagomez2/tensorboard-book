"""Streamlit app for browsing and annotating a folder of TensorBoard runs.

Start it with the `tensorboard-book` command rather than
`streamlit run`: the command checks the password setup, runs the first
(parallel) scan, and passes the configuration through environment
variables (`TBOOK_ROOT`, `TBOOK_DB`, `TBOOK_DEPTH`).

Scans started from the app always parse in this process. Worker
processes would re-import this script as their main module and start the
app again.

Tables are polars DataFrames. Streamlit can only highlight single cells
through a pandas Styler, so :func:`styled` converts a finished table right
before it is drawn; pandas is not used anywhere else.

Each view is one function below; `main` handles login, scanning, the
sidebar filters and picks the view.
"""

from __future__ import annotations

import base64
import difflib
import html
import inspect
import io
import json
import os
import re
import shlex
import threading
import time
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from tensorboard_book import __version__, auth, db, fields, scanner, tbview
from tensorboard_book import analysis as an
from tensorboard_book.fields import Field
from tensorboard_book.helptext import HELP

ROOT = Path(os.environ.get("TBOOK_ROOT", ".")).resolve()
DB_PATH = Path(os.environ.get("TBOOK_DB") or db.default_path(ROOT)).resolve()
DEPTH = int(os.environ.get("TBOOK_DEPTH", "1"))
ANNOTATIONS_PATH = db.annotations_file(DB_PATH)
ASSETS = Path(__file__).with_name("assets")
TENSORBOARD_HOST = os.environ.get("TBOOK_TENSORBOARD_HOST", tbview.LOCAL_HOST)
# "Show" options that are fields carry this prefix; filters accept it too.
PROP = "prop:"
# The fields and metric display settings of the database, refreshed on
# every run of the script. Every session reads the same database.
FIELDS_BY_NAME: dict[str, Field] = {}
METRIC_DISPLAY: dict[str, dict] = {}
HELP_SCALE = (
    "Shows numbers with a prefix: with unit FLOPs and scale G, a stored "
    "4213000000 shows as 4.21 GFLOPs. Values are stored as they are, and "
    "typing 4.2G also works, so ranking and comparisons don't change."
)

# View name -> one-color Material Symbols icon for the sidebar menu.
VIEWS = {
    "Experiments": ":material/science:",
    "Run details": ":material/description:",
    "Compare": ":material/compare_arrows:",
    "Curves": ":material/show_chart:",
    "Timeline": ":material/view_timeline:",
    "Collections": ":material/folder_special:",
    "Fields and settings": ":material/tune:",
    "Help": ":material/menu_book:",
}
# The sidebar menu in sections, with Help last.
MENU = {
    "Explore": ["Experiments", "Run details", "Compare", "Curves", "Timeline"],
    "Organize": ["Collections"],
    "Project": ["Fields and settings", "Help"],
}
# One line under each view's title saying what the view is for.
PURPOSE = {
    "Experiments": (
        "Every run that matches the sidebar filters, with its best scalars."
    ),
    "Compare": (
        "Each run is read at the step where its selection scalar is best, "
        "and every scalar is reported at that step."
    ),
    "Curves": "Training curves of up to eight runs, overlaid.",
    "Timeline": "When runs were running, and how much compute they used.",
    "Run details": (
        "Everything about one run: timing, notes, scalars and files."
    ),
    "Collections": (
        "Named sets of runs, such as a sweep or an ablation, each with the "
        "scalar they are compared on."
    ),
    "Fields and settings": (
        "The values you add to runs and how scalars are shown. Add, rename, "
        "regroup or hide fields; choose which numbers rank and in which "
        "direction."
    ),
    "Help": (
        "How tensorboard-book works, from an overview to the guide of each "
        "page. Every page also has its own guide under How this page works."
    ),
}
# Fonts ship inside the package and are served by Streamlit's static file
# server (`--server.enableStaticServing`), so the app works offline. The
# rest adds what the theme options cannot express: tighter page padding, a
# see-through header, the side menu, summary tiles and pills. Colors are
# theme-neutral (opacity and translucent fills), so they work in both the
# light and the dark theme.
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
   bottom-left corner. */
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
.st-key-nav .tb-section {{ padding-bottom: 0.45rem; opacity: 0.7; }}
.st-key-nav .tb-section-next {{ margin-top: 0.9rem; }}
.tb-kv {{
    display: flex; flex-direction: column; gap: 3px; padding: 4px 0 8px;
}}
.tb-kv > span {{ font-size: 0.78rem; opacity: 0.68; }}
.tb-kv b {{ font-weight: 500; font-size: 0.95rem; overflow-wrap: anywhere; }}
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
.tb-pill.primary {{ background: rgba(15, 140, 130, 0.22); }}
</style>
"""
# Views whose content the sidebar filters change.
FILTER_VIEWS = {"Experiments", "Timeline"}
# The Baseline option in Compare that shows no change columns.
NO_BASELINE = "None"
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
    "collections",
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
    "collections",
    "collection",
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
    "selection scalar",
    "description",
    "property",
    "type",
    "values",
    "metric",
    "scalar",
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

# Chart colors: checked for color vision
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
# Named Streamlit colors for collection pills: drawn as soft,
# translucent fills with normal text, in tables and elsewhere.
PILL_COLORS = ("blue", "green", "orange", "violet", "red", "yellow", "gray")
# Cell highlights per theme: the best value (warm amber)
# and values that differ between runs.
CELL_CSS = {
    "light": {
        "best": "background-color: #FBE7CF; color: #7A3F06; font-weight: 600",
        "differ": "background-color: #F1EEE6; font-weight: 600",
        "good": "color: #0B6B4F; font-weight: 600",
        "bad": "color: #A0400C; font-weight: 600",
    },
    "dark": {
        "best": "background-color: #4A3417; color: #F6C68A; font-weight: 600",
        "differ": "background-color: #2C2A26; font-weight: 600",
        "good": "color: #5CC9A0; font-weight: 600",
        "bad": "color: #F29A6B; font-weight: 600",
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
        version: `db.data_version`; any write invalidates the cache.

    Returns:
        Dict with `runs` and `metrics` (DataFrames), `groups` (a
        list of dicts), `details` (decoded JSON fields per run id),
        `hparams` (a wide DataFrame), `directions`, `settings`,
        `fields` (the fields you defined), `tags_field`,
        `metric_display` and `run_fields` (values by run id).
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
        "fields": db.load_fields(conn),
        "tags_field": db.tags_field(conn),
        "metric_display": db.load_metric_display(conn),
    }
    data["run_fields"] = db.load_run_fields(conn, data["fields"])
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
                    done / total,
                    text=f"Parsed {done}/{total}: {path}",
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
        kind: `"best"`, `"differ"`, or `"good"` and `"bad"` for a
            difference from the baseline that is better or worse.
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
        HTML for `st.caption(..., unsafe_allow_html=True)`.
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
        view: One of `VIEWS`.
        **state: Extra `st.session_state` values, e.g. `run_id`.
    """
    st.session_state.view = view
    for key, value in state.items():
        st.session_state[key] = value


def link_params(params: dict) -> dict[str, list[str]]:
    """Drop empty values and make every value a list, as in the URL.

    Args:
        params: Query parameters; values are strings, lists or None.

    Returns:
        The parameters to put in the URL.
    """
    out = {}
    for key, value in params.items():
        values = value if isinstance(value, list) else [value]
        values = [str(v) for v in values if v not in (None, "")]

        if values:
            out[key] = values
    return out


def update_link(params: dict) -> None:
    """Show the current view's options in the address bar.

    Args:
        params: Query parameters (see :func:`link_params`).
    """
    wanted = link_params(params)
    current = {k: st.query_params.get_all(k) for k in st.query_params}

    if current != wanted:
        st.query_params.from_dict(wanted)


def open_link(data: dict) -> None:
    """Open the comparison a shared link describes, once per session.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
    """
    if st.session_state.get("link_opened"):
        return
    st.session_state.link_opened = True
    params = st.query_params

    if params.get("view") != "compare":
        return
    st.session_state.view = "Compare"
    ids = {r["name"]: r["id"] for r in data["by_id"].values()}
    runs = [ids[n] for n in params.get_all("run") if n in ids]

    if runs:
        st.session_state.compare_runs = runs
        source = "Selected runs"
    else:
        source = params.get("group", "Runs shown in Experiments")
    st.session_state.cmp_source = source

    if "show" in params:
        st.session_state.cmp_show = params.get_all("show")

    if "metric" in params:
        st.session_state[f"sel_{source}"] = params["metric"]

    if "at" in params:
        st.session_state[f"report_{source}"] = params.get_all("at")
    st.session_state[f"agg_{source}"] = params.get("agg") == "1"

    if "seeds" in params:
        st.session_state[f"seeds_{source}"] = params["seeds"]

    if "baseline" in params:
        st.session_state[f"base_{source}"] = params["baseline"]


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
    `default=` or `index=`: Streamlit warns when a widget has both a
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


def sync_pill(pill: str, store, key: str, option: str) -> None:
    """Widget callback: copy a pill's state into an on/off value.

    Args:
        pill: Session key of the pills widget.
        store: Where the value lives: `st.session_state` or a dict in it.
        key: Key of the value in `store`.
        option: The pill's only option.
    """
    store[key] = st.session_state.get(pill) == option


def pill_switch(
    label: str,
    key: str,
    where=None,
    store=None,
    text: str | None = None,
    help: str | None = None,
    label_visibility: str = "collapsed",
) -> bool:
    """Draw an on/off switch as a single pill, clicked to turn it on or off.

    Args:
        label: Widget label, and the pill's text unless `text` is given.
        key: Key of the on/off value in `store`. The pill's own key is
            `<key>_pill`.
        where: Container to draw in. Defaults to the page.
        store: Where the value lives. Defaults to `st.session_state`.
        text: Text on the pill.
        help: Tooltip.
        label_visibility: `"collapsed"`, or `"hidden"` to line the pill
            up with labeled widgets next to it.

    Returns:
        Whether the switch is on.
    """
    where = where or st
    store = st.session_state if store is None else store
    pill = f"{key}_pill"
    st.session_state[pill] = label if store.get(key) else None
    where.pills(
        label,
        [label],
        key=pill,
        format_func=lambda _: text or label,
        label_visibility=label_visibility,
        help=help,
        on_change=sync_pill,
        args=(pill, store, key, label),
    )
    return bool(store.get(key))


def tags_field() -> Field:
    """Return the built-in tags field, whose options order the tag colors."""
    return FIELDS_BY_NAME.get(fields.TAGS) or Field(fields.TAGS, type="tags")


def pill_column(label: str, lists: list[list[str]]):
    """Column config that shows collections as pills.

    Collections all take the primary color, so they never look like a
    tag.

    Args:
        label: Column label.
        lists: The names of every run (a list of lists).

    Returns:
        A `MultiselectColumn` config.
    """
    return st.column_config.MultiselectColumn(
        label,
        options=sorted({name for names in lists for name in names}),
        color="primary",
        width="medium",
    )


def info_value(column: str, members: list[dict]):
    """Summarize an info column over one run or the runs of one config.

    Args:
        column: One of `tags`, `notes`, `status`, `collections`,
            `steps`.
        members: The runs (as dicts) behind one table row.

    Returns:
        A list for tags and collections (shown as pills), otherwise text
        or a number.
    """
    if column in ("tags", "collections"):
        key = "groups" if column == "collections" else column
        names = {name for m in members for name in m[key]}

        if column == "tags":
            order = {t: i for i, t in enumerate(tags_field().options)}
            return sorted(names, key=lambda t: (order.get(t, len(order)), t))
        return sorted(names)

    if column == "notes":
        return " · ".join(m["notes"] for m in members if m["notes"])

    if column == "status":
        return ", ".join(sorted({m["status"] for m in members}))
    steps = [m["max_step"] for m in members if m["max_step"] is not None]
    return max(steps) if steps else None


def info_column_config(lists: dict[str, list[list[str]]]) -> dict:
    """Column configs for tag, collection and run-list columns.

    Args:
        lists: `{"tags": ..., "groups": ...}` with the tags and
            collections of every run, so pill colors match the rest of the
            app. Tags take their colors from the order of their options.

    Returns:
        A `column_config` mapping.
    """
    return {
        "tags": options_column("tags", tags_field()),
        "collections": pill_column("collections", lists["groups"]),
        "runs": st.column_config.ListColumn("runs", width="medium"),
        "notes": st.column_config.TextColumn("notes", width="medium"),
    }


def field_cell(fld: Field, members: list[dict]):
    """Value of a field for one table row.

    Args:
        fld: The field.
        members: The runs behind the row (one, or all seeds of a config).

    Returns:
        For a number, the value (the mean when several runs share the row)
        or None; for choice and tags, the distinct values as a list in
        option order, so they show as pills; for yes/no, the value when
        the runs agree; otherwise the distinct values joined by " · ".
    """
    values = [
        m["fields"][fld.name] for m in members if fld.name in m["fields"]
    ]

    if fld.is_number:
        return sum(values) / len(values) if values else None

    if fld.has_options:
        items = {
            v
            for value in values
            for v in (value if fld.type == "tags" else [value])
        }
        order = {o: i for i, o in enumerate(fld.options)}
        return sorted(items, key=lambda v: (order.get(v, len(order)), v))

    if fld.type == "bool":
        return values[0] if values and len(set(values)) == 1 else None
    return " · ".join(dict.fromkeys(str(v) for v in values)) or None


def filter_value(fld: Field, value):
    """Return a field value as the Filter box compares it.

    Args:
        fld: The field.
        value: The stored value, or None.

    Returns:
        A float for numbers; text otherwise (tags joined by commas, yes/no
        as `yes` or `no`).
    """
    if value is None or fld.is_number:
        return value

    if fld.type == "tags":
        return ", ".join(value)

    if fld.type == "bool":
        return "yes" if value else "no"
    return str(value)


FIELD_DTYPES = {
    "number": pl.Float64,
    "choice": pl.List(pl.String),
    "tags": pl.List(pl.String),
    "bool": pl.Boolean,
}


def field_columns(
    options: list[str],
    defined: list[Field],
    taken,
) -> dict[str, Field]:
    """Map the chosen field options of "Show" to column labels.

    A field's column is labeled with its name, unit and direction, such
    as `params (M) ↓`, unless another column already uses it; then
    `(field)` is appended.

    Args:
        options: Chosen "Show" options (only `prop:` ones are used).
        defined: All fields.
        taken: Column labels already in the table.

    Returns:
        Mapping of column label to field, in option order.
    """
    by_name = {f.name: f for f in defined}
    columns = {}
    for option in options:
        fld = by_name.get(option.removeprefix(PROP))

        if not option.startswith(PROP) or fld is None:
            continue
        label = fld.label()

        if label in taken or label in columns:
            label = f"{label} (field)"
        columns[label] = fld
    return columns


def add_field_columns(
    columns: dict,
    field_cols: dict[str, Field],
    members: list[list[dict]],
) -> None:
    """Add field columns to a table being built, one cell per row.

    Args:
        columns: The table's columns so far (changed in place).
        field_cols: Output of :func:`field_columns`.
        members: For each row, the runs behind it.
    """
    for label, fld in field_cols.items():
        cells = [field_cell(fld, ms) for ms in members]
        dtype = FIELD_DTYPES.get(fld.type, pl.String)
        columns[label] = pl.Series(label, cells, dtype)


def field_formats(field_cols: dict[str, Field]) -> dict:
    """Display formats for number fields: their scale and decimals.

    Args:
        field_cols: Output of :func:`field_columns`.

    Returns:
        Mapping of column label to format function.
    """
    return {
        label: fld.format for label, fld in field_cols.items() if fld.is_number
    }


def field_highlights(table: pl.DataFrame, field_cols: dict) -> dict:
    """Rows holding the best value of each number field that ranks.

    Args:
        table: The finished table.
        field_cols: Output of :func:`field_columns`.

    Returns:
        Column label -> row positions, for :func:`styled`.
    """
    return {
        label: best_rows(
            table[label].to_list(),
            "min" if fld.direction == "lower" else "max",
        )
        for label, fld in field_cols.items()
        if fld.direction
    }


def options_column(label: str, fld: Field):
    """Column config that shows a choice or tags field as colored pills.

    Every option keeps the color of its position in the field's options,
    the same everywhere in the app.

    Args:
        label: Column label.
        fld: A choice or tags field.

    Returns:
        A `MultiselectColumn` config.
    """
    colors = fld.option_colors()
    return st.column_config.MultiselectColumn(
        label,
        options=list(colors),
        color=list(colors.values()) or "auto",
        help=fld.description or None,
        width="medium",
    )


def field_config(columns: dict[str, Field]) -> dict:
    """Column configs for field columns, by type.

    Args:
        columns: Output of :func:`field_columns`.

    Returns:
        A `column_config` mapping.
    """
    config = {}
    for label, fld in columns.items():
        help_text = fld.description or None

        if fld.has_options:
            config[label] = options_column(label, fld)
        elif fld.type == "link":
            config[label] = st.column_config.LinkColumn(label, help=help_text)
        elif fld.type == "bool":
            config[label] = st.column_config.CheckboxColumn(
                label,
                help=help_text,
            )
        else:
            config[label] = st.column_config.Column(label, help=help_text)
    return config


def field_input(fld: Field, current, key: str, label: str | None = None):
    """Draw the input for one field value, by type.

    Args:
        fld: The field.
        current: The current value, or None.
        key: Widget key.
        label: Widget label (defaults to the name with the unit values
            are typed in, without the scale prefix).

    Returns:
        What was entered, in a form :func:`fields.coerce` reads; None or
        "" when left empty.
    """
    if label is None:
        label = f"{fld.name} ({fld.unit})" if fld.unit else fld.name
    common = {"key": key, "help": fld.description or None}

    if fld.is_number:
        placeholder = fld.unit or "number"

        if fld.scale:
            factor = fields.SCALES[fld.scale]
            placeholder = (
                f"{placeholder}, such as {fields.input_text(4.2 * factor)} "
                f"or 4.2{fld.scale}"
            )
        return st.text_input(
            label,
            value=fields.input_text(current),
            placeholder=placeholder,
            **common,
        )

    if fld.type == "choice":
        options = list(fld.options)

        if current is not None and current not in options:
            options.append(current)
        return st.selectbox(
            label,
            options,
            index=options.index(current) if current in options else None,
            placeholder="Choose or type a new value",
            accept_new_options=True,
            **common,
        )

    if fld.type == "tags":
        current = list(current or [])
        options = [*fld.options, *(t for t in current if t not in fld.options)]
        return st.multiselect(
            label,
            options,
            default=current,
            placeholder="Choose or type new tags",
            accept_new_options=True,
            **common,
        )

    if fld.type == "bool":
        options = ["yes", "no"]
        index = None if current is None else (0 if current else 1)
        return st.selectbox(label, options, index=index, **common)

    if fld.type == "date":
        value = datetime.fromisoformat(current).date() if current else None
        return st.date_input(label, value=value, **common)
    return st.text_input(label, value=current or "", **common)


def field_html(fld: Field, value) -> str:
    """Format a field value as HTML for the Overview grid.

    Args:
        fld: The field.
        value: The stored value, or None.

    Returns:
        Pills for choice and tags, a link for links, escaped text otherwise
        (the unit is in the label); empty when there is no value.
    """
    if value is None:
        return ""

    if fld.has_options:
        colors = fld.option_colors()
        items = value if isinstance(value, list) else [value]
        pills = "".join(
            f'<span class="tb-pill {colors.get(v, "gray")}">'
            f"{html.escape(str(v))}</span>"
            for v in items
        )
        return f'<span class="tb-pills">{pills}</span>'
    text = html.escape(fld.format(value))

    if fld.type == "link" and text.startswith(("http://", "https://")):
        return f'<a href="{text}" target="_blank">{text}</a>'
    return text


def key_values(pairs: list[tuple[str, str]], columns: int = 3) -> None:
    """Draw labels with values in a compact grid.

    Args:
        pairs: `(label, html)` pairs; the value is escaped HTML. Empty
            values show a dash.
        columns: Number of grid columns.
    """
    cols = st.columns(columns)
    for i, (label, value) in enumerate(pairs):
        cols[i % columns].markdown(
            f'<div class="tb-kv"><span>{html.escape(label)}</span>'
            f"<b>{value or '–'}</b></div>",
            unsafe_allow_html=True,
        )


def group_fields(defined: list[Field]) -> dict[str, list[Field]]:
    """Group fields by their group, keeping their order.

    Args:
        defined: Fields in display order.

    Returns:
        Fields keyed by group, ungrouped ones first under `""`.
    """
    groups: dict[str, list[Field]] = {"": []}
    for fld in defined:
        groups.setdefault(fld.group, []).append(fld)

    if not groups[""]:
        del groups[""]
    return groups


def card(title: str | None = None):
    """Start a bordered section with an optional heading.

    Args:
        title: Heading shown at the top.

    Returns:
        The container, to use with `with`.
    """
    box = st.container(border=True)

    if title:
        box.markdown(f"#### {html.escape(title)}")
    return box


def show_label(option: str) -> str:
    """Display text of a "Show" option: fields show their group."""
    if option.startswith(PROP):
        fld = FIELDS_BY_NAME.get(option.removeprefix(PROP))

        if fld is not None:
            return fld.label(with_group=True, arrow=False)
        return option.removeprefix(PROP)
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
    """Format a hyperparameter for display (`2` rather than `2.0`).

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
    """Label a metric with its unit and ↑ or ↓, like a field.

    For example `val/acc ↑` or `eval/latency (ms) ↓`: higher or lower
    is better, and the unit (with its scale prefix) set in Fields and
    settings → Metrics.
    """
    down = an.direction_of(tag, directions) == "min"
    shown = metric_field(tag).shown_unit
    unit = f" ({shown})" if shown else ""
    return f"{tag}{unit} {'↓' if down else '↑'}"


def metric_field(tag: str, directions: dict | None = None) -> Field:
    """Describe a logged metric as a read-only number field.

    Args:
        tag: The scalar tag.
        directions: User-set directions, for the field's direction.

    Returns:
        A number field with the metric's group (its tag up to the last
        `/`, unless set), unit, scale and decimals.
    """
    shown = METRIC_DISPLAY.get(tag, {})
    group = shown.get("group")
    direction = None

    if directions is not None:
        down = an.direction_of(tag, directions) == "min"
        direction = "lower" if down else "higher"
    return Field(
        tag,
        group=fields.metric_group(tag) if group is None else group,
        type="number",
        unit=shown.get("unit", ""),
        scale=shown.get("scale", ""),
        decimals=shown.get("decimals"),
        direction=direction,
    )


def metric_format(tag: str):
    """Return the function that formats values of a metric for display.

    Metrics without settings keep the usual 4 significant digits.

    Args:
        tag: The scalar tag.

    Returns:
        A function from value to text.
    """
    shown = METRIC_DISPLAY.get(tag)

    if not shown or (shown["decimals"] is None and not shown["scale"]):
        return an.format_value
    fld = metric_field(tag)
    return lambda v: "" if an.is_missing(v) else fld.format(v)


def best_rows(values: list, direction: str) -> set[int]:
    """Find the rows holding the best value of a column.

    Args:
        values: Column values; missing values are ignored.
        direction: `"max"` or `"min"`.

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
    cells: dict[str, list[str]] | None = None,
):
    """Turn a finished polars table into a Styler for `st.dataframe`.

    Streamlit only supports per-cell styling through pandas' Styler, so
    this converts the table at the last moment. List columns become plain
    lists so they still render as pills.

    Args:
        frame: The table to show.
        highlight: Column name -> row positions to highlight.
        formats: Column name -> function that formats a cell for display.
        css: The CSS applied to highlighted cells; the best-value
            highlight of the active theme by default.
        cells: Column name -> CSS of each cell, applied after the
            highlight (e.g. green or orange differences on a best cell).

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
    for column, marks in (cells or {}).items():
        styler = styler.apply(lambda _, m=marks: m, subset=[column])
    for column, func in (formats or {}).items():
        styler = styler.format(func, subset=[column], na_rep="")
    return styler


def headers(columns, config: dict | None = None) -> dict:
    """Add capitalized headers for the app's own columns to a config.

    Args:
        columns: Column names of the table.
        config: The table's `column_config`, if any.

    Returns:
        A `column_config` where every column in `BUILTIN_COLUMNS` is
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


def file_stem(name: str) -> str:
    """Turn a chart or table name into a safe file name, without suffix.

    Args:
        name: Any text, such as a metric tag or a collection name.

    Returns:
        Lowercase letters, digits, dots and dashes only.
    """
    stem = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-.")
    return stem or "figure"


def report_chart(fig, key: str, name: str, where=None, **kwargs):
    """Draw a plotly chart with a download of the figure.

    The download is the figure as plotly JSON, which report editors that
    take plotly figures open as an interactive chart.

    Args:
        fig: The plotly figure.
        key: Widget key of the chart.
        name: What the chart shows; becomes the file name.
        where: Container to draw into; the page by default.
        **kwargs: Passed to `st.plotly_chart` (e.g. `on_select`).

    Returns:
        What `st.plotly_chart` returns (the selection, if any).
    """
    where = where or st
    event = where.plotly_chart(fig, key=key, **kwargs)
    where.container(
        horizontal=True,
        horizontal_alignment="right",
    ).download_button(
        "Download figure",
        data=lambda: fig.to_json(),
        file_name=f"{file_stem(name)}.json",
        mime="application/json",
        icon=":material/add_chart:",
        type="tertiary",
        key=f"report_{key}",
        help="Saves this chart as a plotly JSON file. A report that takes "
        "plotly figures shows it as an interactive chart.",
    )
    return event


def show_table(data, **kwargs):
    """Draw `st.dataframe` with capitalized headers for built-in columns.

    Args:
        data: A polars DataFrame or a pandas Styler from :func:`styled`.
        **kwargs: Passed on to `st.dataframe`.

    Returns:
        What `st.dataframe` returns (the selection, when enabled).
    """
    frame = getattr(data, "data", data)  # A Styler wraps its DataFrame.
    kwargs["column_config"] = headers(
        list(frame.columns),
        kwargs.get("column_config"),
    )
    return st.dataframe(data, **kwargs)


def table_height(rows: int, limit: int = 740) -> int:
    """Height in pixels that fits a table's rows, up to a limit.

    Args:
        rows: Number of rows.
        limit: Maximum height.

    Returns:
        Pixels for `st.dataframe(height=...)`.
    """
    return min(35 * (rows + 1) + 3, limit)


def list_artifacts(run_dir: Path) -> pl.DataFrame:
    """List the non-event files of a run.

    Args:
        run_dir: Absolute run folder.

    Returns:
        DataFrame with `path` (relative), `size` and `modified`.
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


def view_help() -> None:
    """Show the guide of every page in one place, one topic at a time."""
    topics = list(HELP)
    widget_state("help_topic", topics, topics[0])
    choice = st.pills(
        "Topic",
        topics,
        key="help_topic",
        label_visibility="collapsed",
    )
    with st.container(border=True):
        st.markdown(HELP[choice or topics[0]])


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

    if view in HELP:
        with row.popover(
            "How this page works",
            icon=":material/help:",
            type="tertiary",
        ):
            st.markdown(HELP[view])
    st.caption(PURPOSE[view])


def toolbar():
    """Return the button row under a table.

    Returns:
        A horizontal container for plain (secondary) buttons; push a
        download to the right with `bar.space("stretch")`.
    """
    return st.container(horizontal=True, vertical_alignment="center")


def section_label(text: str, first: bool = True) -> None:
    """Draw a small section heading in the sidebar (e.g. "Explore").

    Args:
        text: The heading.
        first: False for a heading below another section of the menu,
            which gets space above it.
    """
    extra = "" if first else " tb-section-next"
    st.markdown(
        f'<div class="tb-section{extra}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def remember_filter(name: str) -> None:
    """Widget callback: copy a filter widget's value into `filters`.

    Filter widgets are only drawn in the views they apply to, and
    Streamlit forgets the value of widgets that aren't drawn. Keeping the
    values in `st.session_state.filters` preserves them across views.

    Args:
        name: Filter name (the widget key is `f_<name>`).
    """
    st.session_state.filters[name] = st.session_state[f"f_{name}"]


def filter_widget(draw, name: str, label: str, *args, **kwargs):
    """Draw one filter widget, starting from its remembered value.

    Args:
        draw: The Streamlit widget function, e.g. `st.multiselect`.
        name: Filter name in `st.session_state.filters`.
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
    runs: pl.DataFrame,
    by_id: dict[int, dict],
    filters: dict,
) -> pl.DataFrame:
    """Return the runs that pass the sidebar filters.

    Args:
        runs: All runs, with a `status` column.
        by_id: Every run as a dict, by id (for the text search).
        filters: `st.session_state.filters`.

    Returns:
        The filtered runs.
    """
    visible = runs
    group = filters["group"]

    if group == "No collection":
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
            props = " ".join(
                f"{k}={filter_value(FIELDS_BY_NAME[k], v)}"
                for k, v in run["fields"].items()
                if k in FIELDS_BY_NAME
            )
            parts = [run["name"], *run["tags"], run["notes"], params, props]
            return " ".join(parts).lower()

        hits = [i for i in visible["id"] if query in haystack(by_id[i])]
        visible = visible.filter(pl.col("id").is_in(hits))
    return visible


def sidebar(
    runs: pl.DataFrame,
    groups: list[dict],
    editor: bool,
    now: float,
) -> None:
    """Draw the sidebar: the menu, filters where they apply, and status.

    The menu comes first; the scan status and controls
    and the app version sit at the bottom.

    Args:
        runs: All runs, with a `status` column.
        groups: All collections.
        editor: Whether the visitor can edit.
        now: Current time.
    """
    view = st.session_state.view
    with st.sidebar:
        with st.container(key="nav", gap=None):
            for i, (section, names) in enumerate(MENU.items()):
                section_label(section, first=i == 0)
                for name in names:
                    st.button(
                        name,
                        icon=VIEWS[name],
                        key=f"nav_{name}",
                        type="primary" if view == name else "tertiary",
                        width="stretch",
                        on_click=go_to,
                        args=(name,),
                    )

        if view in FILTER_VIEWS:
            section_label("Filters")
            filter_widget(
                st.selectbox,
                "group",
                "Collection",
                ["All runs", "No collection", *(g["name"] for g in groups)],
                format_func=lambda g: (
                    "Not in a collection" if g == "No collection" else g
                ),
            )
            tags = list(tags_field().options)
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
        auto = st.session_state.get("auto_refresh", False)
        right.button(
            "Auto",
            icon=":material/autorenew:",
            width="stretch",
            type="primary" if auto else "secondary",
            on_click=st.session_state.update,
            kwargs={"auto_refresh": not auto},
            help="Rescan every 60 s. Click again to stop.",
        )

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


# Saved views of Experiments: the session keys they set, and how each
# column is named in the column order ("i:" info columns and fields, "s:"
# scalars, "h:" hyperparameters).
VIEW_FILTERS = ("group", "tags", "status", "search", "starred", "archived")


def filter_pills(where=None) -> None:
    """Draw the Starred only and Show archived pills of the filters.

    Args:
        where: Container to draw in. Defaults to the page.
    """
    where = where or st
    filters = st.session_state.filters
    pill_switch(
        "Starred only",
        "starred",
        where=where,
        store=filters,
        text="★ Starred only",
        help="Only runs with a star.",
    )
    pill_switch(
        "Show archived",
        "archived",
        where=where,
        store=filters,
        help="Archived runs are hidden unless this is on. Their files are "
        "untouched.",
    )


def experiment_views(settings: dict) -> tuple[dict, str]:
    """Return the saved views of Experiments and the default one.

    Args:
        settings: App settings.

    Returns:
        The views by name, and the name of the default view (or "").
    """
    views = settings.get("experiment_views") or {}
    default = settings.get("default_experiment_view") or ""
    return views, default if default in views else ""


def current_view() -> dict:
    """Collect what a saved view of Experiments stores.

    Returns:
        The sidebar filters, the chosen scalars, hyperparameters and
        extra columns, the column order, the Value switch and the Filter
        box.
    """
    filters = st.session_state.filters
    group = filters["group"]
    return {
        "filters": {k: filters[k] for k in VIEW_FILTERS},
        "scalars": list(st.session_state.get(f"metrics_{group}", [])),
        "hparams": list(st.session_state.get(f"hp_{group}", [])),
        "show": list(st.session_state.get("info_cols", [])),
        "order": list(st.session_state.get("exp_order", [])),
        "value": st.session_state.get("value_mode", "Best"),
        "filter": st.session_state.get("f_expr", ""),
    }


def apply_view(view: dict) -> None:
    """Load a saved view into the widgets of Experiments and the sidebar.

    Columns that no longer exist are dropped when the widgets are drawn.

    Args:
        view: Output of :func:`current_view`.
    """
    filters = st.session_state.filters
    for key in VIEW_FILTERS:
        if key in view.get("filters", {}):
            filters[key] = view["filters"][key]
            st.session_state[f"f_{key}"] = filters[key]
    group = filters["group"]
    st.session_state[f"metrics_{group}"] = list(view.get("scalars", []))
    st.session_state[f"hp_{group}"] = list(view.get("hparams", []))
    st.session_state["info_cols"] = list(view.get("show", []))
    st.session_state["exp_order"] = list(view.get("order", []))
    st.session_state["value_mode"] = view.get("value", "Best")
    st.session_state["f_expr"] = view.get("filter", "")


def open_default_view(settings: dict) -> None:
    """Apply the default view the first time Experiments opens.

    Args:
        settings: App settings.
    """
    if st.session_state.get("views_opened"):
        return
    st.session_state.views_opened = True
    views, default = experiment_views(settings)

    if default:
        apply_view(views[default])
        st.session_state.current_view = default


def restore_experiments() -> None:
    """Show Experiments as it was when another view was opened.

    Streamlit forgets the widgets of a view that isn't drawn, so the
    table is set up again from the copy :func:`view_experiments` keeps.
    The sidebar filters are left as they are, since other views change
    them too.
    """
    last = st.session_state.get("exp_last")

    if last is None or "info_cols" in st.session_state:
        return
    group = st.session_state.filters["group"]

    if last["filters"]["group"] == group:
        st.session_state[f"metrics_{group}"] = list(last["scalars"])
        st.session_state[f"hp_{group}"] = list(last["hparams"])
    st.session_state["info_cols"] = list(last["show"])
    st.session_state["exp_order"] = list(last["order"])
    st.session_state["value_mode"] = last["value"]
    st.session_state["f_expr"] = last["filter"]


def keep_selection(ids: list[int]) -> None:
    """Select again the rows that were selected when Experiments was left.

    Args:
        ids: The runs in the table, in order.
    """
    if "exp_table" in st.session_state:
        return
    rows = [
        ids.index(i) for i in st.session_state.get("selected", []) if i in ids
    ]

    if rows:
        st.session_state.exp_table = {"selection": {"rows": rows}}


def open_view(name: str, view: dict) -> None:
    """Callback of a view button: apply the view and say so.

    Args:
        name: View name.
        view: The saved view.
    """
    apply_view(view)
    st.session_state.current_view = name
    st.toast(f"Opened view '{name}'")


def save_views(views: dict, default: str) -> None:
    """Store the saved views and the default one.

    Args:
        views: Every view, by name.
        default: Name of the default view, or "".
    """
    conn = db.connect(DB_PATH)
    db.save_settings(
        conn,
        {"experiment_views": views, "default_experiment_view": default},
    )
    conn.close()


def save_view(name: str, settings: dict) -> None:
    """Save the current table as a view, replacing one with that name.

    Args:
        name: View name.
        settings: App settings.
    """
    views, default = experiment_views(settings)
    save_views({**views, name: current_view()}, default)
    st.session_state.current_view = name
    st.toast(f"Saved view '{name}'")


def set_default_view(name: str, settings: dict) -> None:
    """Callback of a view's home button: make it the default, or not.

    Args:
        name: View name.
        settings: App settings.
    """
    views, default = experiment_views(settings)

    if default == name:
        save_views(views, "")
        st.toast("Experiments opens with the usual columns")
    else:
        save_views(views, name)
        st.toast(f"Experiments opens with the view '{name}'")


def delete_view(name: str, settings: dict) -> None:
    """Callback of a view's delete button.

    Args:
        name: View name.
        settings: App settings.
    """
    views, default = experiment_views(settings)
    views = {k: v for k, v in views.items() if k != name}
    save_views(views, "" if default == name else default)

    if st.session_state.get("current_view") == name:
        st.session_state.pop("current_view")
    st.toast(f"Deleted view '{name}'")


def render_views(settings: dict, editor: bool) -> None:
    """Draw the saved views, each as a button, and the form to save one.

    Args:
        settings: App settings.
        editor: Whether the visitor can edit.
    """
    views, default = experiment_views(settings)
    st.markdown(
        "**Saved views**",
        help="A view keeps the filters, the columns and their order, the "
        "Value switch and the Filter box. The view marked with the home "
        "button opens when the app starts.",
    )

    if not views:
        st.caption("No saved views yet.")
    current = st.session_state.get("current_view")

    if current not in views:
        current = None
    for i, (name, view) in enumerate(views.items()):
        row = st.container(horizontal=True, vertical_alignment="center")
        row.button(
            f"{name} · default" if name == default else name,
            key=f"view_open_{i}",
            icon=":material/check:" if name == current else None,
            width="stretch",
            on_click=open_view,
            args=(name, view),
            help="Show the filters and columns of this view.",
        )

        if editor:
            row.button(
                "",
                key=f"view_home_{i}",
                icon=":material/home:",
                type="primary" if name == default else "tertiary",
                on_click=set_default_view,
                args=(name, settings),
                help=(
                    "Stop opening Experiments with this view."
                    if name == default
                    else "Open Experiments with this view."
                ),
            )
            row.button(
                "",
                key=f"view_delete_{i}",
                icon=":material/delete:",
                type="tertiary",
                on_click=delete_view,
                args=(name, settings),
                help=f"Delete the view '{name}'.",
            )

    if not editor:
        return

    if current is not None:
        if views[current] != current_view():
            st.caption(f"The table no longer matches the view '{current}'.")
            st.button(
                f"Save changes to '{current}'",
                key="view_update",
                icon=":material/save:",
                type="primary",
                width="stretch",
                on_click=save_view,
                args=(current, settings),
            )
        else:
            st.caption(f"The table shows the view '{current}'.")
    # A form sends the typed name with the button, without pressing Enter.
    with st.form("new_view", clear_on_submit=True, border=False):
        name = st.text_input(
            "Save the current table as a new view",
            placeholder="View name",
            help="A view with the same name is replaced.",
        )

        if st.form_submit_button("Save view", icon=":material/bookmark_add:"):
            if name.strip():
                save_view(name.strip(), settings)
                st.rerun()
            else:
                st.error("Type a name for the view.")


def ordered_columns(keys: list[str]) -> list[str]:
    """Put the shown columns in the order chosen under Columns.

    Args:
        keys: The shown columns, in their default order.

    Returns:
        The same columns: first the ones with a chosen place, in that
        order, then new ones in their default order.
    """
    order = [k for k in st.session_state.get("exp_order", []) if k in keys]
    order += [k for k in keys if k not in order]
    st.session_state.exp_order = order
    return order


def move_column(key: str, step: int) -> None:
    """Callback of an arrow: move a column up or down in the order.

    Args:
        key: The column.
        step: `-1` to move it up, `1` to move it down.
    """
    order = list(st.session_state.get("exp_order", []))

    if key not in order:
        return
    i = order.index(key)
    j = i + step

    if 0 <= j < len(order):
        order[i], order[j] = order[j], order[i]
    st.session_state.exp_order = order


def render_column_order(order: list[str], names: dict[str, str]) -> None:
    """Draw the shown columns with arrows to change their order.

    Args:
        order: The shown columns, in order.
        names: The name of each column.
    """
    st.markdown(
        "**Column order**",
        help="Run is always first. Saved views keep this order. Dragging a "
        "column header in the table only lasts until the page reloads.",
    )

    if not order:
        st.caption("Pick scalars, hyperparameters or Show columns first.")
        return
    rows = st.container(gap=None)
    for n, key in enumerate(order):
        row = rows.container(
            horizontal=True,
            vertical_alignment="center",
            gap="small",
        )
        row.button(
            "",
            key=f"col_up_{key}",
            icon=":material/arrow_upward:",
            type="tertiary",
            disabled=n == 0,
            on_click=move_column,
            args=(key, -1),
        )
        row.button(
            "",
            key=f"col_down_{key}",
            icon=":material/arrow_downward:",
            type="tertiary",
            disabled=n == len(order) - 1,
            on_click=move_column,
            args=(key, 1),
        )
        row.markdown(names[key])
    st.button(
        "Default order",
        key="col_reset",
        type="tertiary",
        icon=":material/restart_alt:",
        on_click=st.session_state.pop,
        args=("exp_order", None),
    )


def view_experiments(
    data: dict,
    visible: pl.DataFrame,
    editor: bool,
    now: float,
) -> None:
    """Main table of runs with metric, hyperparameter and info columns.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
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
            "threshold in Fields and settings → Settings.",
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
        filter_pills(st.container(horizontal=True))
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
        "Scalars",
        options,
        key=metric_key,
        format_func=lambda t: f"{arrow(t, directions)}  ({counts[t]} runs)",
        on_change=save_metric_choice,
        help="Saved per collection. Set higher or lower is better, units "
        "and decimals in Fields and settings → Scalars.",
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
    defined = data["fields"]
    field_options = [PROP + f.name for f in defined]
    widget_state(
        "info_cols",
        INFO_COLUMNS + field_options,
        [
            "status",
            "steps",
            "compute",
            "start",
            "tags",
            *(PROP + f.name for f in defined if f.visible),
        ],
    )
    shown_cols = right.multiselect(
        "Show",
        INFO_COLUMNS + field_options,
        key="info_cols",
        format_func=column_label,
        help="Extra columns, including your fields (Fields and settings → "
        "Fields). Fields marked Visible are shown by default.",
    )
    info_cols = [c for c in shown_cols if not c.startswith(PROP)]
    # The default order: info columns, fields, scalars, hyperparameters.
    names = {f"i:{c}": column_label(c) for c in info_cols}
    names.update(
        {f"i:{c}": column_label(c) for c in shown_cols if c.startswith(PROP)}
    )
    names.update({f"s:{t}": arrow(t, directions) for t in chosen})
    names.update({f"h:{c}": c for c in hp_cols if f"s:{c}" not in names})
    col_order = ordered_columns(list(names))
    row = st.container(horizontal=True, vertical_alignment="bottom")
    expr = row.text_input(
        "Filter",
        key="f_expr",
        placeholder="optimizer.lr < 1e-3 and val/acc > 0.9 and model ~ resnet",
        help="Terms joined by `and`. Operators: == != < <= > >= and ~ "
        "(regex). Keys are hyperparameters, scalar tags, or run, status, "
        "steps.",
        width="stretch",
    )
    widget_state("value_mode", ["Best", "Last"], "Best")
    mode = row.segmented_control(
        "Value",
        ["Best", "Last"],
        required=True,
        key="value_mode",
    )
    with row.popover("Columns", icon=":material/view_column:"):
        render_column_order(col_order, names)
    with row.popover("Views", icon=":material/bookmarks:"):
        render_views(settings, editor)

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
            values.select("id", *extra_metrics),
            on="id",
            how="left",
        )
        for fld in defined:
            cell = [
                filter_value(fld, by_id[i]["fields"].get(fld.name))
                for i in base["id"]
            ]
            dtype = pl.Float64 if fld.is_number else pl.String
            # "prop:name" always works; plain "name" unless it's taken.
            keys = [PROP + fld.name]

            if fld.name not in base.columns:
                keys.append(fld.name)
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
    shown_row = st.container(horizontal=True, vertical_alignment="center")
    total = data["runs"]

    if not st.session_state.filters["archived"]:
        total = total.filter(pl.col("archived") == 0)
    shown_row.markdown(
        f"**{visible.height}** of {total.height} runs",
        width="content",
    )
    shown_row.space("stretch")
    filter_pills(shown_row)

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
            to_local(visible["start_time"], tz),
            dtype=pl.Datetime,
        ),
        "last event": pl.Series(
            to_local(visible["end_time"], tz),
            dtype=pl.Datetime,
        ),
        "size": visible["total_bytes"],
        "tags": visible["tags"],
        "collections": visible["groups"],
        "notes": visible["notes"].str.slice(0, 80),
    }
    columns = {
        "★": ["★" if s else "" for s in visible["starred"]],
        "run": visible["name"],
    }
    columns.update({c: info[c] for c in info_cols})
    # Fields sit between the info columns and the metrics.
    field_cols = field_columns(shown_cols, defined, columns)
    add_field_columns(columns, field_cols, [[by_id[i]] for i in ids])
    tag_of = {}
    ordered_values = order.join(
        values,
        on="id",
        how="left",
        maintain_order="left",
    )
    for tag in chosen:
        tag_of[arrow(tag, directions)] = tag
        columns[arrow(tag, directions)] = ordered_values[tag]
    ordered_hp = order.join(hp_all, on="id", how="left", maintain_order="left")
    for col in hp_cols:
        if col not in columns:
            columns[col] = [format_hparam(v) for v in ordered_hp[col]]
    labels = {f"i:{c}": c for c in info_cols}
    labels.update({f"i:{PROP}{f.name}": lab for lab, f in field_cols.items()})
    labels.update({f"s:{t}": arrow(t, directions) for t in chosen})
    labels.update({f"h:{c}": c for c in hp_cols})
    placed = list(
        dict.fromkeys(labels[k] for k in col_order if labels.get(k) in columns)
    )
    first = ["★", "run"]
    rest = [c for c in columns if c not in first and c not in placed]
    table = pl.DataFrame(columns).select(*first, *placed, *rest)

    def dash(fmt):
        return lambda v: fmt(v) or "–"

    formats = {
        label: dash(metric_format(tag)) for label, tag in tag_of.items()
    }
    formats.update(field_formats(field_cols))
    for col in ("compute", "wall span"):
        if col in table.columns:
            formats[col] = an.format_duration

    if "size" in table.columns:
        formats["size"] = an.format_bytes
    highlight = {
        label: best_rows(
            table[label].to_list(),
            an.direction_of(tag, directions),
        )
        for label, tag in tag_of.items()
    }
    highlight.update(field_highlights(table, field_cols))
    all_lists = {
        "tags": data["runs"]["tags"].to_list(),
        "groups": data["runs"]["groups"].to_list(),
    }
    keep_selection(ids)
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
                "steps",
                format="localized",
            ),
            "start": st.column_config.DatetimeColumn(
                "start",
                format="YYYY-MM-DD HH:mm",
            ),
            "last event": st.column_config.DatetimeColumn(
                "last event",
                format="YYYY-MM-DD HH:mm",
            ),
            **info_column_config(all_lists),
            **field_config(field_cols),
        },
    )
    selected = [ids[r] for r in event.selection.rows if r < len(ids)]
    st.session_state.selected = selected
    st.session_state.exp_last = current_view()

    # The toolbar: what's selected, the actions on it
    # (disabled until they apply), and the download on the right.
    bar = toolbar()
    bar.markdown(f"**{len(selected)} selected**", width="content")
    selection_actions(
        bar,
        selected,
        by_id,
        data["groups"],
        group,
        editor,
        defined,
        settings,
    )
    list_cols = [
        c for c, dtype in table.schema.items() if isinstance(dtype, pl.List)
    ]
    bar.space("stretch")
    bar.download_button(
        "Export CSV",
        data=lambda: table.with_columns(
            pl.col(c).cast(pl.List(pl.String)).list.join(", ")
            for c in list_cols
        ).write_csv(),
        file_name="experiments.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    note = " (value at the step it was logged, not smoothed)"
    st.caption(
        f"{table.height} runs · best value per scalar highlighted"
        + (note if mode == "Best" else "")
        + ". Click rows to select them."
    )
    st.caption(short_path(ROOT), unsafe_allow_html=True)
    tensorboard_panel()


def selection_actions(
    bar,
    selected: list[int],
    by_id: dict,
    groups,
    group,
    editor: bool,
    defined: list[Field],
    settings: dict,
) -> None:
    """Draw the buttons that act on the runs ticked in Experiments.

    Args:
        bar: The toolbar to draw into.
        selected: Ticked run ids.
        by_id: Every run as a dict, by id.
        groups: All collections.
        group: The collection chosen in the sidebar, or None.
        editor: Whether the visitor can edit (collections, tags, stars).
        defined: All fields.
        settings: App settings (for launching TensorBoard).
    """
    none = not selected
    runs = [by_id[i] for i in selected]
    # The ways to look at the runs share one menu, so the row fits on
    # one line.
    with bar.popover("Open in", icon=":material/open_in_new:", disabled=none):
        st.button(
            "Run details",
            icon=":material/description:",
            disabled=len(selected) != 1,
            on_click=go_to,
            args=("Run details",),
            kwargs={"run_id": selected[0] if selected else None},
            help="Pick one run to open its page.",
            width="stretch",
        )
        st.button(
            "Curves",
            icon=":material/show_chart:",
            on_click=go_to,
            args=("Curves",),
            kwargs={"curve_runs": selected[:8], "curve_seeded": selected},
            help="Plots the training curves of up to eight runs.",
            width="stretch",
        )
        tensorboard_button(
            st,
            runs,
            settings,
            "exp",
            "TensorBoard",
            width="stretch",
        )
    bar.button(
        "Compare",
        icon=":material/compare_arrows:",
        disabled=len(selected) < 2,
        on_click=go_to,
        args=("Compare",),
        kwargs={"compare_runs": selected, "cmp_source": "Selected runs"},
    )

    if not editor:
        return
    conn = db.connect(DB_PATH)
    with bar.popover(
        "Add to collection",
        icon=":material/playlist_add:",
        disabled=none,
    ):
        choice = st.selectbox(
            "Existing collection",
            ["—", *(g["name"] for g in groups)],
            key="bulk_group",
        )
        new_name = st.text_input("…or new collection", key="bulk_new_group")

        if st.button("Add", type="primary"):
            name = new_name.strip() or (choice if choice != "—" else "")

            if name:
                gid = db.create_group(conn, name)
                db.add_to_group(conn, gid, selected)
                st.rerun()

        if group is not None and st.button(f"Remove from {group['name']}"):
            db.remove_from_group(conn, group["id"], selected)
            st.rerun()
    with bar.popover("Field", icon=":material/label:", disabled=none):
        # Tags is a field too, with adding and removing instead of setting.
        by_name = {fields.TAGS: tags_field(), **{f.name: f for f in defined}}
        name = st.selectbox(
            "Field",
            list(by_name),
            key="bulk_prop",
            format_func=lambda n: by_name[n].label(
                with_group=True,
                arrow=False,
            ),
        )

        if name == fields.TAGS:
            bulk_tags(conn, selected, runs)
        else:
            fld = by_name[name]
            value = field_input(
                fld,
                None,
                key=f"bulk_prop_value_{name}",
                label="Value",
            )
            left, right = st.columns(2)

            if left.button("Set", type="primary", key="bulk_prop_set"):
                try:
                    db.set_field(conn, selected, name, value)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

            if right.button("Clear", key="bulk_prop_clear"):
                db.set_field(conn, selected, name, None)
                st.rerun()
    all_starred = bool(runs) and all(r["starred"] for r in runs)

    if bar.button(
        "Unstar" if all_starred else "Star",
        icon=":material/star:",
        disabled=none,
        help="Stars every selected run, or unstars them when all are starred.",
    ):
        db.set_run_flag(conn, selected, "starred", 0 if all_starred else 1)
        st.rerun()
    all_archived = bool(runs) and all(r["archived"] for r in runs)

    if bar.button(
        "Unarchive" if all_archived else "Archive",
        icon=":material/archive:",
        disabled=none,
        help="Archived runs are hidden unless Show archived is on.",
    ):
        db.set_run_flag(conn, selected, "archived", 0 if all_archived else 1)
        st.rerun()
    conn.close()


def bulk_tags(conn, selected: list[int], runs: list[dict]) -> None:
    """Draw adding and removing tags on the selected runs.

    Args:
        conn: Open connection.
        selected: Selected run ids.
        runs: The selected runs as dicts.
    """
    new_tags = st.multiselect(
        "Add tags",
        list(tags_field().options),
        key="bulk_tags",
        accept_new_options=True,
        placeholder="Choose or type new tags",
    )

    if st.button("Add tags", type="primary", disabled=not new_tags):
        db.add_tags(conn, selected, new_tags)
        st.rerun()
    present = sorted({t for r in runs for t in r["tags"]})

    if present:
        drop = st.multiselect("Remove tags", present, key="bulk_untag")

        if st.button("Remove tags", disabled=not drop):
            db.remove_tags(conn, selected, drop)
            st.rerun()


def view_compare(data: dict, visible: pl.DataFrame, editor: bool) -> None:
    """Ablation view: one selection metric, other metrics at its best step.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
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
        help="A collection, the runs you ticked in Experiments (Compare "
        "button), or the runs the Experiments filters show.",
    )
    field_options = [PROP + f.name for f in data["fields"]]
    show_options = ["tags", "notes", "status", "collections", "steps"]
    show_options += field_options
    widget_state(
        "cmp_show",
        show_options,
        [
            "tags",
            "notes",
            *(PROP + f.name for f in data["fields"] if f.visible),
        ],
    )
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
            "This collection has no runs yet. Select runs in Experiments and "
            "use Collection → Add to collection."
        )
        return
    pool_ids = [r["id"] for r in pool]

    options = list(tag_counts(metrics, pool_ids))

    if not options:
        st.info("These runs have no scalars.")
        return
    saved_sel = group["selection_metric"] if group else None
    sel_default = (
        saved_sel if saved_sel in options else default_metrics(options, 1)[0]
    )
    c1, c2, c3 = st.columns([2, 3, 2])
    widget_state(f"sel_{source}", options, sel_default)
    sel = c1.selectbox(
        "Selection scalar",
        options,
        key=f"sel_{source}",
        format_func=lambda t: arrow(t, directions),
        help="The checkpoint (step) is chosen where this scalar is best. "
        "Other scalars are reported at that step, which avoids "
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
        "Scalars at that step",
        report_options,
        key=f"report_{source}",
        format_func=lambda t: arrow(t, directions),
    )
    aggregate = pill_switch(
        "Aggregate seeds",
        f"agg_{source}",
        where=c3,
        label_visibility="hidden",
    )
    key_mode, seed_regex = "name", settings["seed_regex"]

    if aggregate:
        a, b = st.columns([2, 3])
        widget_state(f"seeds_{source}", ["name", "hparams"], "name")
        key_mode = a.radio(
            "Seeds share a config when…",
            ["name", "hparams"],
            key=f"seeds_{source}",
            format_func={
                "name": "names match without the seed",
                "hparams": "hyperparameters match (except seed)",
            }.get,
            horizontal=True,
        )

        if key_mode == "name":
            seed_regex = b.text_input(
                "Seed pattern (regex)",
                value=settings["seed_regex"],
            )

    def config_of(run: dict) -> str:
        if key_mode == "name":
            return an.seed_key(run["name"], seed_regex)
        return an.hparam_key(run["hparams"])

    # The baseline is a run, or a config when seeds are aggregated.
    base_options = [
        NO_BASELINE,
        *dict.fromkeys(config_of(r) if aggregate else r["name"] for r in pool),
    ]
    widget_state(f"base_{source}", base_options, NO_BASELINE)
    baseline = st.columns([2, 3])[0].selectbox(
        "Baseline",
        base_options,
        key=f"base_{source}",
        help="Numbers show their difference to this "
        f"{'config' if aggregate else 'run'} in brackets: green is better "
        "and orange is worse, following the scalar's direction.",
    )
    # The address bar keeps these choices, so the URL reopens them.
    link = {
        "view": "compare",
        **(
            {"group": source}
            if group is not None
            else {"run": [r["name"] for r in pool]}
        ),
        "metric": sel,
        "at": report,
        "show": shown_cols,
        "agg": "1" if aggregate else None,
        "seeds": key_mode if aggregate else None,
        "baseline": None if baseline == NO_BASELINE else baseline,
    }
    update_link(link)

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
        row["config"] = config_of(run)
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
        field_cols = field_columns(
            shown_cols,
            data["fields"],
            {*columns, *labels.values()},
        )
        add_field_columns(columns, field_cols, members)
        highlight, changes = {}, {}
        configs = stats["config"].to_list()
        base_row = configs.index(baseline) if baseline in configs else None

        if base_row is not None:
            columns["config"] = mark_baseline(configs, base_row)
        for c in value_cols:
            means = stats[f"{c}__mean"].to_list()
            stds = stats[f"{c}__std"].fill_null(0.0).to_list()
            fmt = metric_format(c)
            # The spread uses the metric's decimals when they are set.
            spread = fmt if fmt is not an.format_value else "{:.2g}".format
            columns[labels[c]] = [
                f"{fmt(m)} ± {spread(s)}" if m is not None else ""
                for m, s in zip(means, stds)
            ]

            if base_row is not None:
                columns[labels[c]], changes[labels[c]] = with_changes(
                    columns[labels[c]],
                    means,
                    base_row,
                    an.direction_of(c, directions),
                )
            highlight[labels[c]] = best_rows(
                means,
                an.direction_of(c, directions),
            )
        shown = pl.DataFrame(columns)
        chart = {
            "label": stats["config"].to_list(),
            "value": stats[f"{sel}__mean"].to_list(),
            "err": stats[f"{sel}__std"].fill_null(0.0).to_list(),
        }
        show_table(
            styled(shown, highlight, field_formats(field_cols), cells=changes),
            hide_index=True,
            height=table_height(shown.height),
            column_config={
                **info_column_config(all_lists),
                **field_config(field_cols),
            },
        )
        st.caption(
            f"Mean ± std over seeds of {sel} at its best step, and the "
            "other scalars at that same step."
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
        field_cols = field_columns(
            shown_cols,
            data["fields"],
            {*columns, "best step", *labels.values()},
        )
        add_field_columns(columns, field_cols, [[r] for r in runs_here])
        columns["best step"] = df["best step"]
        names = df["run"].to_list()
        base_row = names.index(baseline) if baseline in names else None
        changes = {}

        if base_row is not None:
            columns["run"] = mark_baseline(names, base_row)
        for c in value_cols:
            columns[labels[c]] = df[c]

            if base_row is not None:
                # The cells become text: the value, then the difference.
                values = df[c].to_list()
                columns[labels[c]], changes[labels[c]] = with_changes(
                    [metric_format(c)(v) for v in values],
                    values,
                    base_row,
                    an.direction_of(c, directions),
                )
        shown = pl.DataFrame(columns)
        highlight = {
            labels[c]: best_rows(
                df[c].to_list(),
                an.direction_of(c, directions),
            )
            for c in value_cols
        }
        formats = (
            {}
            if changes
            else {labels[c]: metric_format(c) for c in value_cols}
        )
        formats.update(field_formats(field_cols))
        highlight.update(field_highlights(shown, field_cols))
        show_table(
            styled(shown, highlight, formats, cells=changes),
            hide_index=True,
            height=table_height(shown.height),
            column_config={
                **info_column_config(all_lists),
                **field_config(field_cols),
            },
        )
        st.caption(
            "Other scalars use the logged step closest to the best step "
            f"of {sel}."
        )
        export = shown.with_columns(
            pl.col(labels[c]).map_elements(
                an.format_value,
                return_dtype=pl.String,
            )
            for c in value_cols
            if not changes
        )
        chart = {
            "label": df["run"].to_list(),
            "value": df[sel].to_list(),
            "err": [0.0] * df.height,
        }
        best_ids = df["id"][:1].to_list()
        n_configs = df.height

    if base_row is not None:
        st.caption(
            f"In brackets: the difference to the baseline, {baseline}. "
            "Green is better and orange is worse."
        )
    elif baseline != NO_BASELINE:
        st.caption(f"The baseline, {baseline}, didn't log {sel}.")
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
    report_chart(fig, "compare_chart", f"compare {source}")

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
        bar,
        [by_id[i] for i in df["id"]],
        data["settings"],
        "cmp",
    )
    with bar.popover("LaTeX source", icon=":material/code:"):
        st.code(latex, language="latex")
    bar.space("stretch")
    for label, content, ext in [
        ("Export CSV", export.write_csv(), "csv"),
        ("Markdown", an.to_markdown(export), "md"),
        ("LaTeX", latex, "tex"),
    ]:
        bar.download_button(
            label,
            content,
            file_name=f"{source}.{ext}",
            icon=":material/download:",
            key=f"export_{ext}",
        )
    # The chart as plotly JSON, next to the table exports.
    bar.download_button(
        "Chart",
        data=lambda: fig.to_json(),
        file_name=f"{file_stem(f'compare {source}')}.json",
        mime="application/json",
        icon=":material/add_chart:",
        key="export_report",
        help="The chart above as a plotly JSON file. A report that takes "
        "plotly figures shows it as an interactive chart.",
    )
    tensorboard_panel()


def mark_baseline(labels: list[str], base_row: int) -> list[str]:
    """Add "(baseline)" to the baseline's name, as a table column.

    Args:
        labels: Run or config names, in table order.
        base_row: Position of the baseline.

    Returns:
        The names, the baseline's marked.
    """
    return [
        f"{label} (baseline)" if i == base_row else label
        for i, label in enumerate(labels)
    ]


def with_changes(
    texts: list[str],
    values: list,
    base_row: int,
    direction: str,
) -> tuple[list[str], list[str]]:
    """Add each value's difference to the baseline's, in brackets.

    Args:
        texts: The cells as shown, in table order.
        values: The numbers behind them (means when seeds are aggregated).
        base_row: Position of the baseline.
        direction: The metric's direction, `"max"` or `"min"`.

    Returns:
        The new cells, and the CSS of each cell: green when the difference
        is better, orange when it is worse.
    """
    cells, css = [], []
    for i, (text, value) in enumerate(zip(texts, values)):
        diff = (
            None if i == base_row else an.change_from(value, values[base_row])
        )

        if diff is None:
            cells.append(text)
            css.append("")
            continue
        cells.append(f"{text}  ({an.format_change(diff)})")
        kind = an.change_kind(diff, direction)
        css.append(cell_css(kind) if kind else "")
    return cells, css


def compare_hparams(
    hparams: pl.DataFrame,
    df: pl.DataFrame,
    best_ids: list[int],
) -> None:
    """Show the hyperparameters that differ between the compared runs.

    Args:
        hparams: Wide hyperparameter table of all runs.
        df: The compared runs (`id` and `run`), best first.
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
        styled(diff, highlight, css=cell_css("differ")),
        hide_index=True,
    )
    st.caption(f"Highlighted: differs from the best run ({best['run']}).")


def parameter_explorer(
    data: dict,
    df: pl.DataFrame,
    targets: list[str],
    directions: dict,
) -> None:
    """Show which settings lead to the best value of a target metric.

    Like TensorBoard's HParams dashboard: a parallel coordinates plot with
    one axis per chosen hyperparameter or field and the target last,
    one line per run colored by the target, and below it the target
    against each setting.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
        df: The compared runs: `id`, `run` and one column per metric
            (the selection metric at its best step, the others at that
            step).
        targets: Metrics that can be the target, the selection metric
            first.
        directions: User-set metric directions.
    """
    by_id = data["by_id"]
    hp = df.select("id").join(
        data["hparams"],
        on="id",
        how="left",
        maintain_order="left",
    )
    hp_cols = [
        c for c in hp.columns if c != "id" and hp[c].null_count() < hp.height
    ]
    prop_values = {
        f.name: [
            filter_value(f, by_id[i]["fields"].get(f.name)) for i in df["id"]
        ]
        for f in data["fields"]
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
        "Target scalar",
        targets,
        key="px_target",
        format_func=lambda t: arrow(t, directions),
        help="The selection scalar at its best step, or another scalar "
        "at that step.",
    )
    widget_state("px_dims", options, default or options[:4])
    dims = right.multiselect(
        "Settings",
        options,
        key="px_dims",
        format_func=show_label,
        help="Hyperparameters and fields to plot, in this order.",
    )
    st.caption(
        "One line per run, from its settings to the target scalar. Drag "
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
    axes: dict,
    scores: list,
    target: str,
    scale: list,
    lower: bool,
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
            "label": show_label(dim),
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
    report_chart(fig, "px_parcoords", "parameters")


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
        subplot_titles=[show_label(d) for d in axes],
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
    report_chart(grid, "px_grid", f"{target} vs settings")
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
    where,
    runs: list[dict],
    settings: dict,
    key: str,
    label: str = "",
    disabled: bool = False,
    width: str = "content",
) -> None:
    """Draw the "Open in TensorBoard" button.

    Args:
        where: Container to draw into (usually a toolbar).
        runs: The runs to show.
        settings: App settings.
        key: Unique suffix for the widget key.
        label: Button text; "Open in TensorBoard" by default.
        disabled: Show the button greyed out (nothing to open yet).
        width: `"content"` or `"stretch"`, as for `st.button`.
    """
    where.button(
        label or "Open in TensorBoard",
        icon=":material/monitoring:",
        disabled=disabled,
        width=width,
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
        data: Output of :func:`load_data`, plus `by_id`.
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
    keep_curve_runs(pool[:4])
    pool += [
        i
        for i in st.session_state.get("curve_runs", [])
        if i in by_id and i not in pool
    ]

    if not pool:
        st.info("No runs to plot.")
        return
    widget_state("curve_runs", pool, pool[:4])
    top = st.columns([3, 2])
    chosen_runs = top[0].multiselect(
        "Runs",
        pool,
        key="curve_runs",
        max_selections=8,
        format_func=lambda i: by_id[i]["name"],
        help="Starts with the runs selected in Experiments.",
    )
    st.session_state.curve_runs_kept = list(chosen_runs)
    label_options = curve_label_options(data, chosen_runs or pool)
    remembered = st.session_state.get("curve_labels_kept", [])
    widget_state("curve_labels", label_options, remembered)
    labels = top[1].multiselect(
        "Label",
        label_options,
        key="curve_labels",
        max_selections=3,
        format_func=curve_label_name,
        placeholder="Run name only",
        help="Tags, fields or hyperparameters to show next to the run name "
        "in the legends and when hovering a line.",
    )
    st.session_state.curve_labels_kept = list(labels)
    names = {i: run_label(by_id[i], labels) for i in chosen_runs}
    options = list(tag_counts(metrics, chosen_runs or pool))
    widget_state("curve_metrics", options, default_metrics(options, 4))
    tags = st.multiselect(
        "Scalars",
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
    log_y = pill_switch(
        "Log y",
        "curves_log_y",
        where=c3,
        label_visibility="hidden",
    )

    if not chosen_runs or not tags:
        st.info("Pick at least one run and one scalar.")
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
            name = names[rid]
            x = curve_x(series, by_id[rid], x_mode, tz)
            add_curve(
                fig,
                x,
                series,
                smoothing,
                color=colors[rid],
                name=name,
                hover=name,
            )
        style_curves(fig, arrow(tag, directions), x_mode, log_y)
        with columns[k % 2]:
            report_chart(fig, f"curve_{tag}", tag)
    overlay_section(
        conn,
        data,
        chosen_runs,
        names,
        options,
        smoothing,
        x_mode,
        log_y,
    )
    conn.close()


def keep_curve_runs(default: list[int]) -> None:
    """Choose the runs Curves starts with when its picker isn't drawn yet.

    Streamlit forgets the picker when another view is shown. Curves then
    starts with the runs selected in Experiments if that selection changed
    since Curves last took it, and with the runs it showed last otherwise.

    Args:
        default: Runs to show when there is neither (the newest ones).
    """
    if "curve_runs" in st.session_state:
        return
    selected = st.session_state.get("selected", [])

    if selected and selected != st.session_state.get("curve_seeded"):
        st.session_state.curve_runs = selected[:8]
        st.session_state.curve_seeded = selected
    else:
        st.session_state.curve_runs = list(
            st.session_state.get("curve_runs_kept") or default
        )


def curve_label_options(data: dict, run_ids: list[int]) -> list[str]:
    """List what Curves can show next to a run name.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
        run_ids: The runs on the charts.

    Returns:
        `"tags"`, then `"f:<field>"` for each field and `"h:<name>"` for
        each hyperparameter of the runs, the ones that differ between
        them first.
    """
    hparams = data["hparams"].filter(pl.col("id").is_in(run_ids))
    names = [
        c
        for c in hparams.columns
        if c != "id" and hparams[c].null_count() < hparams.height
    ]
    varying = an.varying_columns(hparams.select("id", *names))
    names = varying + [c for c in names if c not in varying]
    return [
        "tags",
        *(f"f:{f.name}" for f in data["fields"]),
        *(f"h:{c}" for c in names),
    ]


def curve_label_name(option: str) -> str:
    """Display text of a Label option of Curves.

    Args:
        option: Output of :func:`curve_label_options`.
    """
    if option == "tags":
        return "Tags"

    if option.startswith("f:"):
        return show_label(PROP + option[2:])
    return option[2:]


def run_label(run: dict, labels: list[str]) -> str:
    """Build the name of a run on a chart: its name and the chosen labels.

    Args:
        run: The run as a dict.
        labels: Options from :func:`curve_label_options`.

    Returns:
        Such as `"resnet-0412 · lr=0.001 · baseline"`. Labels the run has
        no value for are left out.
    """
    parts = [run["name"]]
    for option in labels:
        if option == "tags":
            text = ", ".join(run["tags"])
        elif option.startswith("f:"):
            name = option[2:]
            fld = FIELDS_BY_NAME.get(name)
            value = run["fields"].get(name)

            if fld is None or value is None:
                continue

            if isinstance(value, list):
                text = ", ".join(str(v) for v in value)
            else:
                text = fld.format(value)
            text = f"{name}={text}" if text else ""
        else:
            text = format_hparam(run["hparams"].get(option[2:]))
            text = f"{option[2:]}={text}" if text else ""

        if text:
            parts.append(text)
    return " · ".join(parts)


# Line styles that tell overlaid curves of the same color apart.
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def overlay_section(
    conn,
    data: dict,
    run_ids: list[int],
    names: dict[int, str],
    options: list[str],
    smoothing: float,
    x_mode: str,
    log_y: bool,
) -> None:
    """Draw several metrics of the chosen runs on one shared chart.

    Args:
        conn: Open database connection.
        data: Output of :func:`load_data`, plus `by_id`.
        run_ids: Runs to plot.
        names: Name of each run on the chart, from :func:`run_label`.
        options: Metrics the runs logged.
        smoothing: EMA weight from the Smoothing slider.
        x_mode: Value of the X axis control.
        log_y: Whether the y axis is logarithmic.
    """
    directions, by_id = data["directions"], data["by_id"]
    st.markdown("#### Overlay scalars")
    st.caption(
        "Several scalars on one chart, for scalars on the same scale "
        "(train and validation loss, or accuracies on different splits)."
    )
    widget_state("overlay_metrics", options, an.overlay_default(options))
    c1, c2 = st.columns([5, 2])
    tags = c1.multiselect(
        "Scalars to overlay",
        options,
        key="overlay_metrics",
        max_selections=len(DASHES),
        format_func=lambda t: arrow(t, directions),
        placeholder="Choose two or more scalars",
    )
    color_by = c2.segmented_control(
        "Color by",
        ["Run", "Scalar"],
        default="Scalar" if len(run_ids) == 1 else "Run",
        required=True,
        key="overlay_color_by",
        help="The other one is told apart by the line style.",
    )

    if not tags:
        st.info("Choose the scalars to draw on one chart.")
        return
    run_colors_ = run_colors(run_ids)
    metric_colors = dict(zip(tags, palette()))
    one_run = len(run_ids) == 1
    tz = viewer_tz()
    fig = go.Figure()
    for i, rid in enumerate(run_ids):
        run = by_id[rid]
        for j, tag in enumerate(tags):
            series = db.load_series(conn, rid, tag)

            if series.is_empty():
                continue

            if color_by == "Run":
                color, dash = run_colors_[rid], DASHES[j]
            else:
                color, dash = metric_colors[tag], DASHES[i % len(DASHES)]
            label = tag if one_run else f"{names[rid]} · {tag}"
            add_curve(
                fig,
                curve_x(series, run, x_mode, tz),
                series,
                smoothing,
                color=color,
                name=label,
                hover=f"{names[rid]}<br>{tag}",
                dash=dash,
            )
    title = " vs ".join(arrow(t, directions) for t in tags)
    style_curves(fig, title, x_mode, log_y, height=460)
    report_chart(fig, "curve_overlay", " vs ".join(tags))


def curve_x(series: pl.DataFrame, run: dict, x_mode: str, tz):
    """Return the x values of a curve for the chosen X axis.

    Args:
        series: `step`, `value` and `wall_time` of one metric.
        run: The run the series belongs to.
        x_mode: `"Step"`, `"Hours since start"` or `"Wall time"`.
        tz: Viewer time zone, for wall time.
    """
    if x_mode == "Step":
        return series["step"].to_numpy()

    if x_mode == "Hours since start":
        start = run["start_time"] or 0.0
        return (series["wall_time"].to_numpy() - start) / 3600
    return to_local(series["wall_time"], tz)


def add_curve(
    fig: go.Figure,
    x,
    series: pl.DataFrame,
    smoothing: float,
    color: str,
    name: str,
    hover: str,
    dash: str = "solid",
) -> None:
    """Add one smoothed curve, over its faint raw data, to a figure.

    Args:
        fig: Figure to draw on.
        x: X values, from :func:`curve_x`.
        series: `step` and `value` of the metric.
        smoothing: EMA weight; 0 draws the raw data only.
        color: Line color.
        name: Legend entry.
        hover: First lines of the tooltip.
        dash: Plotly line style of the smoothed line.
    """
    steps = series["step"].to_numpy()
    values = series["value"].to_numpy()

    if smoothing > 0:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                line={"color": color, "width": 1, "dash": dash},
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
            line={"color": color, "width": 2, "dash": dash},
            customdata=np.stack([steps, values], axis=-1),
            hovertemplate=f"{hover}"
            "<br>step %{customdata[0]:,.0f}"
            "<br>value %{customdata[1]:.4g}<extra></extra>",
        )
    )


def style_curves(
    fig: go.Figure,
    title: str,
    x_mode: str,
    log_y: bool,
    height: int = 340,
) -> None:
    """Apply the shared layout of the Curves charts.

    Args:
        fig: Figure to style.
        title: Chart title.
        x_mode: Value of the X axis control.
        log_y: Whether the y axis is logarithmic.
        height: Chart height in pixels.
    """
    fig.update_layout(
        title={"text": title, "font": {"size": 14}},
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified" if x_mode != "Wall time" else "closest",
        # Anchored to the bottom of the figure, so a legend of several
        # rows (long labels) pushes the chart up instead of covering the
        # axis title.
        legend={
            "orientation": "h",
            "yref": "container",
            "y": 0,
            "yanchor": "bottom",
        },
        xaxis_title=None if x_mode == "Wall time" else x_mode.lower(),
    )

    if log_y:
        fig.update_yaxes(type="log")


def view_timeline(data: dict, visible: pl.DataFrame, now: float) -> None:
    """Gantt timeline of when runs ran, plus a compute calendar.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
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
        ["Collection", "Status", "Tag"],
        default="Collection",
        required=True,
        key="tl_color",
    )
    filter_pills(st.container(horizontal=True))
    days = {"7 days": 7, "30 days": 30, "90 days": 90, "1 year": 365}.get(span)
    start_limit = now - days * 86400 if days else -np.inf

    group_order = [
        g["name"] for g in sorted(data["groups"], key=lambda g: g["id"])
    ]
    # Tags keep the order, and so the colors, set in Fields and settings.
    # A run with several tags is drawn in the color of the first one.
    tag_order = info_value(
        "tags",
        [{"tags": t} for t in data["runs"]["tags"].to_list()],
    )
    rows = []
    for rid in visible["id"]:
        run = by_id[rid]
        run_tags = info_value("tags", [run])

        if color_by == "Collection":
            ranked = sorted(run["groups"], key=group_order.index)
            label = ranked[0] if ranked else "No collection"
        elif color_by == "Tag":
            label = run_tags[0] if run_tags else "Untagged"
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
                "tags": ", ".join(run_tags) or "no tags",
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
        stable = group_order if color_by == "Collection" else tag_order
        cmap = label_colors(labels, stable)
        for lab in ("No collection", "Untagged"):
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
        custom_data=["run_id", "span", "compute", "status", "tags"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{base|%b %d %H:%M} → "
        "%{x|%b %d %H:%M}<br>span %{customdata[1]} · compute "
        "%{customdata[2]} · %{customdata[3]}<br>tags: %{customdata[4]}"
        "<extra></extra>",
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
    event = report_chart(
        fig,
        "timeline",
        "timeline",
        on_select="rerun",
        selection_mode="points",
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
    report_chart(fig2, "concurrency", "runs at once", left)

    # Compute hours use the activity segments, so pauses don't count.
    segments = [
        [max(seg_start, start_limit), seg_end]
        for r in rows
        for seg_start, seg_end in by_id[r["run_id"]]["segments"]
        if seg_end >= start_limit
    ]
    hours = an.daily_hours(segments, tz)
    calendar = compute_calendar(hours, now, days, tz)
    report_chart(calendar, "calendar", "compute per day", right)
    total = sum(hours.values())
    busiest = max(hours, key=hours.get) if hours else None
    c1, c2, c3, c4 = cards.columns(4)
    c1.metric("Runs in range", len(rows), border=True)
    c2.metric(
        "Compute time",
        an.format_duration(total * 3600) or "0s",
        help="Pauses are not counted.",
        border=True,
    )
    c3.metric(
        "Busiest day",
        f"{busiest:%a %b %d}" if busiest else "–",
        help=f"{an.format_duration(hours[busiest] * 3600)} of compute"
        if busiest
        else None,
        border=True,
    )
    c4.metric(
        "Running now",
        sum(r["status"] == "Active" for r in rows),
        border=True,
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
            f"{d:%a %b %d}: "
            f"{an.format_duration(hours.get(d, 0.0) * 3600)} of compute"
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
    data: dict,
    candidates: pl.DataFrame,
    editor: bool,
    now: float,
) -> None:
    """Everything about one run: timing, annotations, metrics and files.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
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

    # Status, flags, collections and tags as one row of soft pills. Field
    # values are in the Overview tab.
    status_color = {"Active": "green", "Stopped": "gray", "Missing": "red"}
    items = [(html.escape(run["status"]), status_color[run["status"]])]

    if run["starred"]:
        items.append(("★ starred", "orange"))

    if run["archived"]:
        items.append(("archived", "gray"))
    # Collections take the primary color, so they never look like a tag.
    items += [(html.escape(g), "primary") for g in run["groups"]]

    if run["has_nonfinite"]:
        items.append(("NaN/inf logged", "red"))
    tag_colors = tags_field().option_colors()
    items += [
        (html.escape(t), tag_colors.get(t, "gray"))
        for t in info_value("tags", [run])
    ]
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
    editing = editor and st.session_state.get("editing_run") == rid

    if editor and not editing:
        actions.button(
            "Edit",
            icon=":material/edit:",
            type="primary",
            on_click=st.session_state.update,
            kwargs={"editing_run": rid},
        )

    if editing:
        # The editor takes the place of the page.
        annotation_form(run, groups, data["fields"])
        return

    if run["notes"]:
        note = run["notes"].strip().replace("\n", " ")
        st.markdown(f":material/sticky_note_2: {note}")
    elif editor:
        st.caption("No notes yet. Use Edit to add tags, notes and fields.")
    tensorboard_panel()

    if run["parse_error"]:
        st.warning(
            f"Some event data could not be read:\n\n{run['parse_error']}"
        )

    c = st.columns(6)
    c[0].metric(
        "Compute time",
        an.format_duration(run["compute_time"]),
        border=True,
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
        [
            "Overview",
            "Artifacts",
            "Scalars",
            "Hyperparameters",
            "Text",
            "Timing",
        ]
    )
    with tabs[0]:
        overview_tab(run, data)
    with tabs[1]:
        artifacts_tab(run, run_dir, by_id, settings, tz)
    with tabs[2]:
        metrics_tab(metrics.filter(pl.col("run_id") == rid), run, directions)
        st.code(f'tensorboard --logdir "{run_dir}"', language="bash")
    with tabs[3]:
        hparams_tab(run, by_id)
    with tabs[4]:
        if run["texts"]:
            for tag, text in run["texts"].items():
                st.markdown(f"**{tag}**")
                with st.container(border=True):
                    st.markdown(text)
        else:
            st.info("No text summaries.")
    with tabs[5]:
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


def config_paths(run: dict) -> list[str]:
    """Return the config files of a run, its main config file first.

    Args:
        run: The run, as a dict.

    Returns:
        Paths inside the run of the config files that were read.
    """
    return [s for s in run["hparam_sources"] if s != "event hparams"]


def text_diff(
    run: dict,
    rel: str,
    other: dict,
    other_rel: str,
) -> None:
    """Show the line-by-line difference between two text files.

    Args:
        run: The run of the file shown on the page.
        rel: That file's path inside its run.
        other: The run to compare with.
        other_rel: The other file's path inside its run.
    """
    path = safe_file(ROOT / run["path"], rel)
    other_path = safe_file(ROOT / other["path"], other_rel)
    text = read_text(path) if path else None
    other_text = read_text(other_path) if other_path else None

    if text is None or other_text is None:
        st.info("One of the files is binary or unreadable.")
        return
    diff = "".join(
        difflib.unified_diff(
            other_text.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=f"{other['name']}/{other_rel}",
            tofile=f"{run['name']}/{rel}",
        )
    )
    st.code(diff or "Files are identical.", language="diff")


def hparams_tab(run: dict, by_id: dict) -> None:
    """The hyperparameters of a run, and how they differ from another's.

    Args:
        run: The run, as a dict.
        by_id: Every run as a dict.
    """
    hp = run["hparams"]

    if not hp:
        st.info(
            "No hyperparameters found in the events or in config files. "
            "The names of config files are set in Fields and settings → "
            "Settings."
        )
        return
    show_table(
        pl.DataFrame(
            {
                "key": list(hp),
                "value": [format_hparam(v) for v in hp.values()],
            }
        ),
        hide_index=True,
    )
    configs = config_paths(run)
    st.caption(
        "Sources: "
        + ", ".join(run["hparam_sources"])
        + (f". Main config file: `{configs[0]}`." if configs else ".")
    )
    others = [
        i
        for i, r in sorted(by_id.items(), key=lambda item: item[1]["name"])
        if i != run["id"] and r["hparams"]
    ]

    if not others:
        return
    st.markdown("**Compare with another run**")
    other_id = st.selectbox(
        "Run to compare with",
        [None, *others],
        format_func=lambda i: "—" if i is None else by_id[i]["name"],
        key=f"hp_diff_{run['id']}",
    )

    if other_id is None:
        return
    other = by_id[other_id]
    keys = list(dict.fromkeys([*hp, *other["hparams"]]))
    changed = [
        k
        for k in keys
        if format_hparam(hp.get(k)) != format_hparam(other["hparams"].get(k))
    ]

    if changed:
        show_table(
            pl.DataFrame(
                {
                    "key": changed,
                    other["name"]: [
                        format_hparam(other["hparams"].get(k)) for k in changed
                    ],
                    run["name"]: [format_hparam(hp.get(k)) for k in changed],
                }
            ),
            hide_index=True,
        )
        st.caption(
            f"{len(changed)} of {len(keys)} hyperparameters differ. Values "
            "are compared after reading the files, so the file names, "
            "formats and the order of the keys don't matter."
        )
    else:
        st.success("Every hyperparameter is the same.")
    other_configs = config_paths(other)

    if not configs or not other_configs:
        return
    with st.expander("Line-by-line difference of the config files"):
        pick, other_pick = configs[0], other_configs[0]

        if len(configs) > 1 or len(other_configs) > 1:
            left, right = st.columns(2)
            other_pick = left.selectbox(
                f"File of {other['name']}",
                other_configs,
                key=f"hp_diff_file_{run['id']}_{other_id}",
            )
            pick = right.selectbox(
                f"File of {run['name']}",
                configs,
                key=f"hp_diff_own_{run['id']}",
            )
        text_diff(run, pick, other, other_pick)


def overview_tab(run: dict, data: dict) -> None:
    """Field values grouped in cards, one per field group.

    Args:
        run: The run, as a dict.
        data: Output of :func:`load_data`.
    """
    defined = data["fields"]

    if not defined:
        st.info(
            "No fields yet. Add fields such as the dataset or the GPU in "
            "Fields and settings → Fields, then set them here with Edit."
        )
    else:
        show_empty = pill_switch("Show empty fields", "run_show_empty")
        shown_any = False
        for group, members in group_fields(defined).items():
            pairs = [
                (
                    f.label(arrow=False),
                    field_html(f, run["fields"].get(f.name)),
                )
                for f in members
                if show_empty or f.name in run["fields"]
            ]

            if not pairs:
                continue
            shown_any = True
            with card(group or "General"):
                key_values(pairs)

        if not shown_any:
            st.caption(
                "This run has no field values yet. Use Edit to set them, or "
                "turn on Show empty fields."
            )
    collections = [g for g in data["groups"] if run["id"] in g["members"]]
    with card("Collections"):
        if collections:
            show_table(
                pl.DataFrame(
                    {
                        "Collection": [g["name"] for g in collections],
                        "Selection scalar": [
                            g["selection_metric"] or "" for g in collections
                        ],
                        "Description": [g["description"] for g in collections],
                    }
                ),
                hide_index=True,
            )
        else:
            st.caption("Not in any collection.")


def render_add_field(bar) -> None:
    """Draw a small form to add a field without leaving the editor.

    Args:
        bar: Container the popover goes in.
    """
    with bar.popover("Add a field", icon=":material/add:"):
        with st.form("ed_add_field", clear_on_submit=True, border=False):
            name = st.text_input("Name")
            group = st.selectbox(
                "Group",
                sorted({f.group for f in FIELDS_BY_NAME.values()} - {""}),
                index=None,
                accept_new_options=True,
                placeholder="None, or type a new group",
            )
            kind = st.selectbox("Type", fields.TYPES)
            direction = st.selectbox(
                "Better when",
                [None, *fields.DIRECTIONS],
                format_func=lambda d: d or "not ranked",
                help="Only for number fields.",
            )
            unit = st.text_input("Unit", placeholder="for example %, ms, M")

            if st.form_submit_button("Add field", type="primary"):
                conn = db.connect(DB_PATH)
                try:
                    db.create_field(
                        conn,
                        Field(
                            name,
                            group=group or "",
                            type=kind,
                            unit=unit if kind == "number" else "",
                            direction=direction if kind == "number" else None,
                            position=len(FIELDS_BY_NAME),
                        ),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.toast(f"Added field '{name}'")
                    st.rerun()
                finally:
                    conn.close()


def annotation_form(
    run: dict,
    groups: list[dict],
    defined: list[Field],
) -> None:
    """Edit a run's fields, tags, collections, flags and notes.

    Laid out as one card per field group with
    three inputs per row (the built-in Tags under General), then the
    collections, then the notes, and **Save changes** at the end.

    Args:
        run: The run, as a dict.
        groups: All collections.
        defined: All fields.
    """
    rid = run["id"]
    names = {g["id"]: g["name"] for g in groups}
    bar = st.container(horizontal=True)
    render_add_field(bar)
    bar.button(
        "Cancel",
        on_click=st.session_state.update,
        kwargs={"editing_run": None},
    )
    st.caption(
        "Choice and tags fields accept new values, which become options."
    )
    values = {}
    with st.form(f"annotate_{rid}", border=False):
        layout = group_fields([tags_field(), *defined])
        for group, members in layout.items():
            with card(group or "General"):
                cells = st.columns(3)
                for k, fld in enumerate(members):
                    current = (
                        run["tags"]
                        if fld.name == fields.TAGS
                        else run["fields"].get(fld.name)
                    )
                    with cells[k % 3]:
                        values[fld.name] = field_input(
                            fld,
                            current,
                            key=f"prop_{rid}_{fld.name}",
                        )
        with card("Collections"):
            cells = st.columns(3)
            group_ids = cells[0].multiselect(
                "Collections",
                list(names),
                default=[g["id"] for g in groups if rid in g["members"]],
                format_func=names.get,
                placeholder="Not in a collection",
            )
            new_group = cells[1].text_input(
                "New collection",
                placeholder="Name of a new collection",
            )
            with cells[2]:
                starred = st.checkbox("Starred", value=bool(run["starred"]))
                archived = st.checkbox(
                    "Archived",
                    value=bool(run["archived"]),
                    help="Archived runs are hidden unless you turn on Show "
                    "archived. Their files are untouched.",
                )
        notes = card("Notes").text_area(
            "Notes",
            value=run["notes"],
            height=140,
            label_visibility="collapsed",
            placeholder="What this run tried and what came out of it. "
            "Markdown is supported.",
        )
        submitted = st.form_submit_button(
            "Save changes",
            type="primary",
            icon=":material/save:",
        )

    if not submitted:
        return
    tags = values.pop(fields.TAGS)
    conn = db.connect(DB_PATH)
    try:
        for name, value in values.items():
            db.set_field(conn, [rid], name, value)
    except ValueError as exc:
        st.error(str(exc))
        conn.close()
        return

    if new_group.strip():
        group_ids.append(db.create_group(conn, new_group))
    db.set_field(conn, [rid], fields.TAGS, tags)
    db.set_run_groups(conn, rid, group_ids)
    db.set_run_flag(conn, [rid], "starred", int(starred))
    db.set_run_flag(conn, [rid], "archived", int(archived))
    db.set_run_flag(conn, [rid], "notes", notes)
    conn.close()
    st.session_state["editing_run"] = None
    st.rerun()


def metrics_tab(metrics: pl.DataFrame, run: dict, directions: dict) -> None:
    """Table of every scalar of one run with best, last, min and max.

    Args:
        metrics: The run's rows of the metrics table.
        run: The run, as a dict.
        directions: User-set directions.
    """
    if metrics.is_empty():
        st.info("No scalars logged.")
        return
    rows = []
    for m in metrics.sort("tag").iter_rows(named=True):
        lower = an.direction_of(m["tag"], directions) == "min"
        fmt = metric_format(m["tag"])
        rows.append(
            {
                "group": metric_field(m["tag"]).group,
                "tag": arrow(m["tag"], directions),
                "best": fmt(m["min"] if lower else m["max"]),
                "best step": m["step_min"] if lower else m["step_max"],
                "last": fmt(m["last"]),
                "last step": m["last_step"],
                "min": fmt(m["min"]),
                "max": fmt(m["max"]),
                "points": m["n"],
                "NaN/inf": m["nonfinite"],
            }
        )
    # Grouped as in Fields and settings → Metrics: by the group of each
    # metric, which is its tag up to the last "/" unless set there.
    table = pl.DataFrame(rows).sort("group", "tag")
    show_table(table, hide_index=True)

    if run["other_tags"]:
        found = ", ".join(
            f"{n} {plugin} tag(s)" for plugin, n in run["other_tags"].items()
        )
        st.caption(f"Also in the logs (open TensorBoard to view): {found}")


def artifacts_tab(
    run: dict,
    run_dir: Path,
    by_id: dict,
    settings: dict,
    tz,
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
        "Filter files",
        key="art_filter",
        placeholder="checkpoints/ or .png",
    )
    hide_ckpt = pill_switch(
        "Hide checkpoints",
        "art_hide_ckpt",
        where=right,
        label_visibility="hidden",
    )

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
                "modified",
                format="YYYY-MM-DD HH:mm",
            )
        },
    )
    rows = [r for r in event.selection.rows if r < files.height]
    bar = toolbar()

    if not rows:
        bar.markdown("**0 selected**", width="content")
        bar.caption(
            "Click a file to preview and download it, or several for a zip."
        )
        return
    picked = files[rows]
    bar.markdown(f"**{picked.height} selected**", width="content")
    folder = Path(run["path"]).name

    if picked.height > 1:
        size = picked["size"].sum()

        if size > settings["max_zip_mb"] * 1024**2:
            bar.caption(
                f"The selection is {an.format_bytes(size)}, above the "
                f"{settings['max_zip_mb']} MB zip limit (Fields and "
                "settings → Settings)."
            )
            return

        def build_zip() -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(
                buf,
                "w",
                zipfile.ZIP_DEFLATED,
                compresslevel=1,
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
    )
    preview_file(path, rel, size, run, by_id)


def preview_file(
    path: Path,
    rel: str,
    size: int,
    run: dict,
    by_id: dict,
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
    # A config file is compared with the main config file of the other
    # runs, whatever its name; any other file with the same file.
    is_config = rel in config_paths(run)
    targets = {}
    for i, r in sorted(by_id.items(), key=lambda item: item[1]["name"]):
        if i == run["id"] or r["missing"]:
            continue
        other_configs = config_paths(r)

        if is_config and other_configs:
            targets[i] = other_configs[0]
        elif (ROOT / r["path"] / rel).is_file():
            targets[i] = rel

    if not targets:
        return
    other = st.selectbox(
        f"Diff {rel} with the "
        + ("config file of" if is_config else "same file in"),
        [None, *targets],
        format_func=lambda i: (
            "—"
            if i is None
            else (
                by_id[i]["name"]
                if targets[i] == rel
                else f"{by_id[i]['name']} ({targets[i]})"
            )
        ),
        key=f"diff_{run['id']}_{rel}",
    )

    if other is None:
        return
    text_diff(run, rel, by_id[other], targets[other])


def view_collections(data: dict, editor: bool) -> None:
    """Create, rename, describe, fill and delete collections of runs.

    Collections were called groups before 0.8; the database and the
    annotations file still call them groups.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
        editor: Whether the visitor can edit.
    """
    conn = db.connect(DB_PATH)
    groups, by_id = data["groups"], data["by_id"]

    if editor:
        with st.form("new_group", clear_on_submit=True):
            a, b = st.columns([1, 2])
            name = a.text_input("New collection")
            description = b.text_input(
                "Description",
                placeholder="What this ablation tests",
            )

            if st.form_submit_button("Create collection") and name.strip():
                db.create_group(conn, name, description)
                conn.close()
                st.rerun()

    if not groups:
        st.info(
            "No collections yet. Create one here, or select runs in "
            "Experiments → Collection."
        )
        conn.close()
        return
    show_table(
        pl.DataFrame(
            {
                "collection": [g["name"] for g in groups],
                "runs": [len(g["members"]) for g in groups],
                "selection scalar": [
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

    if not editor:
        conn.close()
        return
    by_gid = {g["id"]: g for g in groups}
    gid = st.selectbox(
        "Edit collection",
        list(by_gid),
        format_func=lambda i: by_gid[i]["name"],
    )
    g = by_gid[gid]
    with st.form(f"edit_group_{gid}"):
        new_name = st.text_input("Name", value=g["name"])
        desc = st.text_input("Description", value=g["description"])
        members = st.multiselect(
            "Runs",
            list(by_id),
            default=[m for m in g["members"] if m in by_id],
            format_func=lambda i: by_id[i]["name"],
        )

        if st.form_submit_button("Save collection", type="primary"):
            db.update_group(
                conn,
                gid,
                name=new_name.strip() or g["name"],
                description=desc,
            )
            removed = [m for m in g["members"] if m not in members]
            db.remove_from_group(conn, gid, removed)
            db.add_to_group(conn, gid, members)
            conn.close()
            st.rerun()

    def delete() -> None:
        delete_conn = db.connect(DB_PATH)
        db.delete_group(delete_conn, gid)
        delete_conn.close()

    confirm_button(
        "Delete collection",
        f"Delete the collection **{g['name']}**? Its runs stay.",
        delete,
        key=f"delete_group_{gid}",
        icon=":material/delete:",
    )
    conn.close()


def confirm_button(label, message, on_confirm, key, icon=None) -> None:
    """Draw a button that asks for confirmation in a popover.

    The popover says what will happen, and
    **Yes, do it** does it.

    Args:
        label: Button label.
        message: What will happen (Markdown).
        on_confirm: Called when the user confirms.
        key: Unique key.
        icon: Optional Material icon.
    """
    with st.popover(label, icon=icon):
        st.markdown(message)

        if st.button("Yes, do it", key=key, type="primary"):
            on_confirm()
            st.rerun()


# "add" (new rows only, no deleting rows by hand: Delete does that) exists
# in newer Streamlit versions.
NUM_ROWS = (
    "add"
    if "'add'" in str(inspect.signature(st.data_editor).parameters["num_rows"])
    else "dynamic"
)
# Columns of the fields editor in Fields and settings.
FIELD_TABLE = {
    "Group": pl.String,
    "Name": pl.String,
    "Type": pl.String,
    "Better when": pl.String,
    "Unit": pl.String,
    "Scale": pl.String,
    "Decimals": pl.Int64,
    "Visible": pl.Boolean,
    "Options": pl.String,
    "Description": pl.String,
    "Order": pl.Int64,
    "Runs": pl.Int64,
    "Delete": pl.Boolean,
}


def fields_frame(defined: list[Field]) -> pl.DataFrame:
    """Build the editable table of fields, one row per field.

    Args:
        defined: All fields.

    Returns:
        The table, with a hidden `key` column holding each field's
        current name (empty for rows added in the editor).
    """
    return pl.DataFrame(
        {
            "key": [f.name for f in defined],
            "Group": [f.group for f in defined],
            "Name": [f.name for f in defined],
            "Type": [f.type for f in defined],
            "Better when": [f.direction for f in defined],
            "Unit": [f.unit for f in defined],
            "Scale": [f.scale or None for f in defined],
            "Decimals": [f.decimals for f in defined],
            "Visible": [f.visible for f in defined],
            "Options": [", ".join(f.options) for f in defined],
            "Description": [f.description for f in defined],
            "Order": [f.position for f in defined],
            "Runs": [f.runs for f in defined],
            "Delete": [False for _ in defined],
        },
        schema={"key": pl.String, **FIELD_TABLE},
    )


def _text(value) -> str:
    """Return a table cell as stripped text (None and NaN give "")."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _int(value) -> int | None:
    """Return a table cell as an int, or None when empty."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    return int(value)


def apply_field_changes(
    conn,
    original: pl.DataFrame,
    edited,
    by_name: dict[str, Field],
) -> list[str]:
    """Save added, changed and deleted fields from the editor.

    Args:
        conn: Open connection.
        original: The table before editing.
        edited: The table after editing (polars or pandas).
        by_name: The fields before editing, by name.

    Returns:
        One message per problem; empty when everything was saved.
    """
    rows = (
        edited.to_dicts()
        if isinstance(edited, pl.DataFrame)
        else edited.to_dict("records")
    )
    before = {r["key"]: r for r in original.to_dicts()}
    problems = []
    for row in rows:
        key = _text(row.get("key"))
        options = [o.strip() for o in _text(row["Options"]).split(",")]
        try:
            fld = Field(
                name=_text(row["Name"]),
                group=_text(row["Group"]),
                type=_text(row["Type"]) or "text",
                description=_text(row["Description"]),
                unit=_text(row["Unit"]),
                scale=_text(row["Scale"]),
                decimals=_int(row["Decimals"]),
                direction=_text(row["Better when"]) or None,
                options=tuple(dict.fromkeys(o for o in options if o)),
                visible=row["Visible"] is not False,
                position=_int(row["Order"]) or 0,
            )

            if not key:
                if fld.name:
                    db.create_field(conn, fld)
                continue

            if row["Delete"]:
                db.delete_field(conn, key)
                continue
            old = before.get(key)
            same = old is not None and all(
                _text(old[c]) == _text(row[c])
                for c in FIELD_TABLE
                if c not in ("Runs", "Delete")
            )

            if same:
                continue
            # Options that still have values are kept: they are renamed,
            # merged or deleted with Rename values, which changes the runs.
            used = db.option_counts(conn, key)
            kept = [o for o, n in used.items() if n and o not in fld.options]
            fld = replace(fld, options=(*fld.options, *kept))
            db.update_field(conn, key, fld)
        except ValueError as exc:
            problems.append(str(exc))
    return problems


def render_fields(conn, data: dict, editor: bool) -> None:
    """The Fields tab: export and import, options, and the editor.

    Args:
        conn: Open connection.
        data: Output of :func:`load_data`.
        editor: Whether the visitor can edit.
    """
    defined = data["fields"]
    by_name = {f.name: f for f in defined}
    bar = st.container(horizontal=True, gap="small")

    def fields_json() -> str:
        export_conn = db.connect(DB_PATH)
        text = json.dumps(db.export_fields(export_conn), indent=2)
        export_conn.close()
        return text

    bar.download_button(
        "Export fields",
        data=fields_json,
        file_name="tensorboard-book-fields.json",
        mime="application/json",
        icon=":material/download:",
        disabled=not defined,
        help="Saves every field with its settings as JSON, to set up "
        "another runs folder the same way. Values are not included.",
    )

    if editor:
        with bar.popover("Import fields", icon=":material/upload_file:"):
            upload = st.file_uploader(
                "Fields file",
                type=["json"],
                key="fields_upload",
                help="A file from **Export fields**, an annotations file, "
                "or a fields file in the same format from another app.",
            )
            existing = st.radio(
                "Fields that already exist",
                ["Keep as they are", "Update from the file"],
                key="fields_import_existing",
                help="Fields are matched by name. Values are never "
                "touched, but updating can change a field's type.",
            )

            if st.button(
                "Import",
                type="primary",
                disabled=upload is None,
                key="fields_import",
            ):
                try:
                    result = db.import_fields(
                        conn,
                        json.loads(upload.getvalue()),
                        update_existing=existing.startswith("Update"),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.toast(
                        f"Added {len(result['added'])}, updated "
                        f"{len(result['updated'])}, kept "
                        f"{len(result['kept'])} field(s)"
                    )
                    st.rerun()
        with bar.popover(
            "Rename values",
            icon=":material/find_replace:",
            help="Fixes a value on every run at once: rename a typo, merge "
            "two spellings into one, reorder or delete the values of a "
            "choice or tags field, Tags included.",
        ):
            options_editor(conn, data)

    original = fields_frame(defined)

    if not editor:
        show_table(original.drop("key", "Delete"), hide_index=True)
        return
    st.caption(
        "Add a row at the bottom to create a field. Choice and tags fields "
        "list their options, separated by commas; values typed in forms "
        "are added to them. To change a value on every run that has it, "
        "such as a typo, use **Rename values**."
    )
    edited = st.data_editor(
        original,
        hide_index=True,
        num_rows=NUM_ROWS,
        width="stretch",
        key=f"fields_editor_{db.data_version(conn)}",
        column_order=[c for c in FIELD_TABLE],
        height=min(38 + 35 * (len(original) + 2), 720),
        disabled=["Runs"],
        column_config={
            "Group": st.column_config.TextColumn(
                help="Keeps related fields together: in Run details, the "
                "edit form and the Show menus."
            ),
            "Name": st.column_config.TextColumn(required=True),
            "Type": st.column_config.SelectboxColumn(
                options=list(fields.TYPES),
                required=True,
                default="text",
            ),
            "Better when": st.column_config.SelectboxColumn(
                options=list(fields.DIRECTIONS),
                help="Ranks number fields: higher or lower is better, and "
                "the best value is highlighted. Empty for numbers that "
                "don't rank.",
            ),
            "Scale": st.column_config.SelectboxColumn(
                options=[k for k in fields.SCALES if k],
                help=HELP_SCALE,
            ),
            "Decimals": st.column_config.NumberColumn(
                min_value=0,
                max_value=10,
                step=1,
            ),
            "Visible": st.column_config.CheckboxColumn(
                default=True,
                help="Shown in Experiments and Compare by default.",
            ),
            "Order": st.column_config.NumberColumn(step=1),
            "Runs": st.column_config.NumberColumn(
                help="How many runs have a value."
            ),
            "Delete": st.column_config.CheckboxColumn(
                default=False,
                help="Deletes the field and all its values.",
            ),
        },
    )
    rows = (
        edited.to_dicts()
        if isinstance(edited, pl.DataFrame)
        else edited.to_dict("records")
    )
    deleting = [r["key"] for r in rows if _text(r.get("key")) and r["Delete"]]
    with_values = [
        f"{n} ({by_name[n].runs} runs)" for n in deleting if by_name[n].runs
    ]
    confirmed = True

    if with_values:
        confirmed = st.checkbox(
            f"Delete {', '.join(with_values)} with their values. This can't "
            "be undone.",
            key="fields_confirm",
        )

    if st.button(
        "Save fields",
        type="primary",
        icon=":material/save:",
        disabled=not confirmed,
    ):
        problems = apply_field_changes(conn, original, edited, by_name)

        if problems:
            st.error("\n".join(f"- {p}" for p in problems))
        else:
            st.toast("Fields saved")
            st.rerun()


def options_editor(conn, data: dict) -> None:
    """Rename, merge, reorder and delete the options of a field.

    The tags are the options of the built-in tags field. An option's
    position gives its pill color.

    Args:
        conn: Open connection.
        data: Output of :func:`load_data`.
    """
    choices = [
        data["tags_field"],
        *(f for f in data["fields"] if f.has_options),
    ]
    by_name = {f.name: f for f in choices}
    name = st.selectbox(
        "Field",
        list(by_name),
        key="options_field",
        format_func=lambda n: by_name[n].label(with_group=True, arrow=False),
    )
    counts = db.option_counts(conn, name)

    if not counts:
        st.caption("No options yet. They appear as you type values.")
        return
    colors = by_name[name].option_colors()
    table = pl.DataFrame(
        {
            "Option": list(counts),
            "Color": [[o] for o in counts],
            "Rename to": [None] * len(counts),
            "Order": list(range(1, len(counts) + 1)),
            "Runs": list(counts.values()),
            "Delete": [False] * len(counts),
        },
        schema={
            "Option": pl.String,
            "Color": pl.List(pl.String),
            "Rename to": pl.String,
            "Order": pl.Int64,
            "Runs": pl.Int64,
            "Delete": pl.Boolean,
        },
    )
    edited = st.data_editor(
        table,
        hide_index=True,
        disabled=["Option", "Color", "Runs"],
        key=f"options_{name}_{db.data_version(conn)}",
        column_config={
            "Color": st.column_config.MultiselectColumn(
                options=list(colors),
                color=list(colors.values()),
            ),
            "Rename to": st.column_config.TextColumn(
                help="Changes the value on every run. Renaming to an "
                "option that exists merges the two."
            ),
            "Order": st.column_config.NumberColumn(
                step=1,
                help="Sets the order, and so the color.",
            ),
            "Delete": st.column_config.CheckboxColumn(
                help="Removes the value from every run."
            ),
        },
    )
    rows = (
        edited.to_dicts()
        if isinstance(edited, pl.DataFrame)
        else edited.to_dict("records")
    )

    if st.button("Save values", type="primary", key="options_save"):
        rows.sort(key=lambda r: _int(r["Order"]) or 0)
        renames = {
            r["Option"]: _text(r["Rename to"])
            for r in rows
            if _text(r["Rename to"])
        }
        try:
            db.save_options(
                conn,
                name,
                [renames.get(r["Option"], r["Option"]) for r in rows],
                renames=renames,
                deletes=[r["Option"] for r in rows if r["Delete"]],
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.rerun()


def render_metrics(conn, data: dict, editor: bool) -> None:
    """The Metrics tab: direction, group, unit, scale and decimals.

    Logged metrics are shown like number fields. Their names come from
    the logs, so they can't be renamed, added or deleted here.

    Args:
        conn: Open connection.
        data: Output of :func:`load_data`.
        editor: Whether the visitor can edit.
    """
    directions, display = data["directions"], data["metric_display"]
    runs_per_tag = dict(
        data["metrics"].group_by("tag").len().sort("tag").iter_rows()
    )
    tags = list(runs_per_tag)

    if not tags:
        st.info("No scalars logged yet.")
        return
    word = {"max": "higher", "min": "lower"}
    # "auto" follows the guess from the name; higher/lower pin a choice.
    table = pl.DataFrame(
        {
            "Group": [metric_field(t).group for t in tags],
            "Scalar": tags,
            "Better when": [word.get(directions.get(t), "auto") for t in tags],
            "Unit": [display.get(t, {}).get("unit", "") for t in tags],
            "Scale": [display.get(t, {}).get("scale") or None for t in tags],
            "Decimals": [display.get(t, {}).get("decimals") for t in tags],
            "Runs": [runs_per_tag[t] for t in tags],
        },
        schema={
            "Group": pl.String,
            "Scalar": pl.String,
            "Better when": pl.String,
            "Unit": pl.String,
            "Scale": pl.String,
            "Decimals": pl.Int64,
            "Runs": pl.Int64,
        },
    )

    if not editor:
        show_table(table, hide_index=True)
        return
    st.caption(
        "The group of a scalar is its tag up to the last /, unless you "
        "change it. Groups, units and decimals are used in Experiments, "
        "Compare and Run details. Names containing loss, error, mse, … "
        "default to lower is better."
    )
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["Scalar", "Runs"],
        key=f"metrics_editor_{db.data_version(conn)}",
        height=min(38 + 35 * (len(tags) + 1), 720),
        column_config={
            "Better when": st.column_config.SelectboxColumn(
                options=["auto", "higher", "lower"],
                required=True,
                help="auto uses the guess from the name; higher or lower "
                "fixes it.",
            ),
            "Scale": st.column_config.SelectboxColumn(
                options=[k for k in fields.SCALES if k],
                help=HELP_SCALE,
            ),
            "Decimals": st.column_config.NumberColumn(
                min_value=0,
                max_value=10,
                step=1,
                help="Empty shows 4 significant digits.",
            ),
        },
    )
    rows = (
        edited.to_dicts()
        if isinstance(edited, pl.DataFrame)
        else edited.to_dict("records")
    )
    buttons = st.container(horizontal=True, gap="small")

    if buttons.button("Save scalars", type="primary", icon=":material/save:"):
        chosen = {r["Scalar"]: r["Better when"] for r in rows}
        try:
            db.set_metric_display(
                conn,
                {
                    r["Scalar"]: {
                        "group": None
                        if _text(r["Group"])
                        == fields.metric_group(r["Scalar"])
                        else _text(r["Group"]),
                        "unit": _text(r["Unit"]),
                        "scale": _text(r["Scale"]),
                        "decimals": _int(r["Decimals"]),
                    }
                    for r in rows
                },
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        db.set_directions(
            conn,
            {
                t: "min" if b == "lower" else "max"
                for t, b in chosen.items()
                if b != "auto"
            },
        )
        db.clear_directions(
            conn,
            [t for t, b in chosen.items() if b == "auto"],
        )
        st.toast("Scalars saved")
        st.rerun()

    if directions and buttons.button(
        "Reset directions to auto",
        type="tertiary",
        icon=":material/restart_alt:",
        help="Forget every direction you set; each scalar goes back to "
        "the guess from its name.",
    ):
        db.clear_directions(conn)
        st.rerun()


def view_fields_settings(data: dict, editor: bool) -> None:
    """Fields, metrics, settings and the project, in tabs.

    Args:
        data: Output of :func:`load_data`, plus `by_id`.
        editor: Whether the visitor can edit.
    """
    conn = db.connect(DB_PATH)
    tabs = st.tabs(["Fields", "Scalars", "Settings", "Project"])
    with tabs[0]:
        render_fields(conn, data, editor)
    with tabs[1]:
        render_metrics(conn, data, editor)
    with tabs[2]:
        if editor:
            manage_settings(conn, data, editor)
        else:
            st.info("Only editors can change the settings.")
    with tabs[3]:
        render_project(conn, data, editor)
    conn.close()


def render_project(conn, data: dict, editor: bool) -> None:
    """The Project tab: where things are, disk usage and backups.

    Args:
        conn: Open connection.
        data: Output of :func:`load_data`, plus `by_id`.
        editor: Whether the visitor can edit.
    """
    runs = data["runs"]
    tiles = st.columns(4)
    tiles[0].metric("Runs", runs.height, border=True)
    tiles[1].metric("Fields", len(data["fields"]), border=True)
    tiles[2].metric("Collections", len(data["groups"]), border=True)
    tiles[3].metric(
        "Disk usage",
        an.format_bytes(runs["total_bytes"].sum()),
        border=True,
    )
    st.markdown(f"Runs folder: `{ROOT}`")
    st.caption(
        f"Database: `{DB_PATH}`. Annotations are also saved to "
        f"`{ANNOTATIONS_PATH}` after every edit."
    )
    with card("Disk usage"):
        manage_disk(conn, data, editor)
    with card("Backup"):
        manage_backup(conn, data, editor)


def manage_disk(conn, data: dict, editor: bool) -> None:
    """Disk usage per run and per collection."""
    runs, by_id = data["runs"], data["by_id"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", an.format_bytes(runs["total_bytes"].sum()), border=True)
    c2.metric(
        "Artifacts",
        an.format_bytes(runs["artifact_bytes"].sum()),
        border=True,
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
            title={
                "text": "Disk usage per collection (MB)",
                "font": {"size": 14},
            },
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
    settings, defaults = data["settings"], db.DEFAULT_SETTINGS
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
            "Seed pattern in run names (regex)",
            settings["seed_regex"],
            help="Removed from run names to find the runs that differ only "
            f"by their seed. Default: `{defaults['seed_regex']}`.",
        )
        points = b.number_input(
            "Max stored points per curve (0 = all)",
            0,
            10_000_000,
            int(settings["max_points"]),
            help="Longer curves are thinned for plotting, keeping the "
            "first, last, lowest and highest points. Best, last, min and "
            "max always use every point. Default: "
            f"{defaults['max_points']}.",
        )
        zip_mb = a.number_input(
            "Max zip download (MB)",
            1,
            1_000_000,
            int(settings["max_zip_mb"]),
            help="The largest selection of files that can be downloaded "
            f"as one zip. Default: {defaults['max_zip_mb']} MB.",
        )
        max_tb = b.number_input(
            "Max TensorBoards at once",
            1,
            50,
            int(settings["max_tensorboards"]),
            help="Each one is a separate process using memory. Default: "
            f"{defaults['max_tensorboards']}.",
        )
        patterns = st.text_input(
            "Config file names",
            ", ".join(settings["config_patterns"]),
            help="Comma-separated patterns of the .json, .yaml, .yml or "
            ".toml files whose values become hyperparameters, e.g. "
            "`config.*` or `conf/*.toml`. A pattern without / matches "
            "names in the top folder of a run, one folder down or in "
            ".hydra/; one with / matches the path inside the run. The "
            "first pattern a run matches gives its main config file, "
            "which is compared between runs. Default: "
            + ", ".join(db.DEFAULT_SETTINGS["config_patterns"])
            + ".",
        )
        tb_args = st.text_input(
            "Extra TensorBoard arguments",
            settings["tensorboard_args"],
            placeholder="--samples_per_plugin images=100 --reload_interval 30",
            help="Added to every TensorBoard started from the app. Default: "
            "none.",
        )

        if st.form_submit_button("Save settings", type="primary"):
            config_patterns = [
                p.strip() for p in patterns.split(",") if p.strip()
            ]
            reparse = (gap, points, config_patterns) != (
                settings["gap_minutes"],
                settings["max_points"],
                list(settings["config_patterns"]),
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
                    "config_patterns": config_patterns,
                },
            )

            if reparse:
                # The pause and the stored points re-parse every run; the
                # config file names re-parse the runs whose files changed.
                run_scan()
            st.rerun()
    st.caption(
        "Changing the pause or the stored points re-parses every run, since "
        "both change what is computed from the event files. Changing the "
        "config file names re-reads the runs whose config files change."
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
        "Your tags, notes, fields, collections and settings are saved "
        "automatically "
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
    )

    if not editor:
        return
    upload = st.file_uploader(
        "Import annotations (merges into the current ones)",
        type=["json"],
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

        def forget() -> None:
            forget_conn = db.connect(DB_PATH)
            db.forget_missing_runs(forget_conn)
            forget_conn.close()

        confirm_button(
            "Forget missing runs",
            f"Forget the {missing} missing runs and their annotations? "
            "Their files, if any, are not touched.",
            forget,
            key="forget_missing",
            icon=":material/delete_sweep:",
        )


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
            "end_time",
            "last_file_mtime",
            "missing",
        ).iter_rows()
    ]
    runs = runs.with_columns(pl.Series("status", status, pl.String))
    data["runs"] = runs
    # Every run as a dict (row plus decoded JSON fields), for lookups.
    FIELDS_BY_NAME.clear()
    FIELDS_BY_NAME.update({f.name: f for f in data["fields"]})
    FIELDS_BY_NAME[fields.TAGS] = data["tags_field"]
    METRIC_DISPLAY.clear()
    METRIC_DISPLAY.update(data["metric_display"])
    data["by_id"] = {
        r["id"]: {
            **r,
            **data["details"][r["id"]],
            "fields": data["run_fields"].get(r["id"], {}),
        }
        for r in runs.iter_rows(named=True)
    }
    open_link(data)

    if st.session_state.get("view") not in VIEWS:
        st.session_state.view = "Experiments"

    if "filters" not in st.session_state:
        st.session_state.filters = json.loads(json.dumps(FILTER_DEFAULTS))

    if st.session_state.view == "Experiments":
        open_default_view(data["settings"])
        restore_experiments()
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

    if view != "Compare" and len(st.query_params):
        st.query_params.clear()
    page_header(view)

    if view == "Help":
        view_help()
    elif view == "Experiments":
        view_experiments(data, visible, editor, now)
    elif view == "Compare":
        view_compare(data, visible, editor)
    elif view == "Curves":
        view_curves(data, unarchived)
    elif view == "Timeline":
        view_timeline(data, visible, now)
    elif view == "Run details":
        view_run_details(data, unarchived, editor, now)
    elif view == "Collections":
        view_collections(data, editor)
    else:
        view_fields_settings(data, editor)


main()
