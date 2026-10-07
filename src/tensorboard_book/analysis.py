"""Pure helpers for tables, filters, aggregation, timing and export.

Nothing here touches Streamlit or the database, which keeps it easy to
test. Tables are polars DataFrames; runs are identified by an ``id``
column rather than an index.
"""

from __future__ import annotations

import difflib
import math
import re
from datetime import datetime, timedelta

import numpy as np
import polars as pl

LOWER_IS_BETTER = (
    "loss",
    "error",
    "err",
    "mse",
    "mae",
    "rmse",
    "nll",
    "wer",
    "cer",
    "fid",
    "cost",
    "regret",
    "distance",
)
FILTER_TERM = re.compile(
    r"^\s*`?(?P<key>[^`=!<>~]+?)`?\s*"
    r"(?P<op>==|!=|>=|<=|=|>|<|~)\s*(?P<val>.+?)\s*$"
)


def is_missing(value) -> bool:
    """Tell whether a value is missing (None or NaN).

    Args:
        value: Anything.

    Returns:
        True for None and float NaN.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def guess_direction(tag: str) -> str:
    """Guess whether higher or lower values of a metric are better.

    Args:
        tag: Scalar tag such as ``val/loss``.

    Returns:
        ``"min"`` if the name looks like a loss or error, else ``"max"``.
    """
    words = re.split(r"[^a-z0-9]+", tag.lower())
    return "min" if any(w in LOWER_IS_BETTER for w in words) else "max"


def direction_of(tag: str, directions: dict[str, str]) -> str:
    """Return the user-set direction of a metric, falling back to a guess.

    Args:
        tag: Scalar tag.
        directions: User-set directions.

    Returns:
        ``"max"`` or ``"min"``.
    """
    return directions.get(tag) or guess_direction(tag)


def run_status(
    end_time, file_mtime, missing, now: float, active_minutes: float
) -> str:
    """Classify a run as active, stopped or missing.

    TensorBoard logs have no "finished" marker, so the only reliable
    signal is whether the run is still being written to.

    Args:
        end_time: Wall time of the last event.
        file_mtime: Latest modification time of its event files.
        missing: Whether the folder disappeared.
        now: Current time in seconds.
        active_minutes: Silence after which a run counts as stopped.

    Returns:
        ``"Missing"``, ``"Active"`` or ``"Stopped"``.
    """
    if missing:
        return "Missing"
    times = [t for t in (end_time, file_mtime) if not is_missing(t)]
    last = max(times, default=0.0)
    return "Active" if now - last < active_minutes * 60 else "Stopped"


def metric_table(
    metrics: pl.DataFrame,
    run_ids: list[int],
    tags: list[str],
    directions: dict[str, str],
    mode: str = "best",
) -> pl.DataFrame:
    """Build a wide table with one value per run and metric.

    Args:
        metrics: Long metric summaries from ``db.load_metrics``.
        run_ids: Runs to include, in the order of the result's rows.
        tags: Metrics to include (the columns of the result).
        directions: User-set directions.
        mode: ``"best"`` for the best value given the metric's direction,
            ``"last"`` for the last logged value.

    Returns:
        DataFrame with an ``id`` column and one Float64 column per tag
        (null where a run did not log the metric).
    """
    base = pl.DataFrame({"id": run_ids}, schema={"id": pl.Int64})
    if not tags:
        return base
    lower = [t for t in tags if direction_of(t, directions) == "min"]
    value = (
        pl.col("last")
        if mode == "last"
        else pl.when(pl.col("tag").is_in(lower))
        .then(pl.col("min"))
        .otherwise(pl.col("max"))
    )
    long = metrics.filter(
        pl.col("run_id").is_in(run_ids) & pl.col("tag").is_in(tags)
    ).select(pl.col("run_id").alias("id"), "tag", value.alias("value"))
    if long.is_empty():
        wide = base
    else:
        wide = long.pivot(
            on="tag", index="id", values="value", aggregate_function="first"
        )
        wide = base.join(wide, on="id", how="left", maintain_order="left")
    missing = [
        pl.lit(None, pl.Float64).alias(t) for t in tags if t not in wide
    ]
    return wide.with_columns(missing).select(
        "id", *[pl.col(t).cast(pl.Float64) for t in tags]
    )


def best_step(metric: dict, direction: str) -> int | None:
    """Return the step at which a metric reached its best value.

    Args:
        metric: One row of ``db.load_metrics`` as a dict.
        direction: ``"max"`` or ``"min"``.

    Returns:
        The step, or None if the metric has no finite values.
    """
    step = metric["step_min"] if direction == "min" else metric["step_max"]
    return None if is_missing(step) else int(step)


def best_position(values, direction: str) -> int | None:
    """Return the position of the best value in a sequence.

    Args:
        values: Numbers; None and NaN are ignored.
        direction: ``"max"`` or ``"min"``.

    Returns:
        The index of the best value, or None if every value is missing.
    """
    pairs = [(v, i) for i, v in enumerate(values) if not is_missing(v)]
    if not pairs:
        return None
    return (min(pairs) if direction == "min" else max(pairs))[1]


def value_at_step(series: pl.DataFrame, step: int) -> tuple[float, float]:
    """Look up a series at the logged step closest to ``step``.

    Args:
        series: DataFrame with ``step`` and ``value``, sorted by step.
        step: Target step.

    Returns:
        ``(value, actual_step)``, or ``(nan, nan)`` for an empty series.
    """
    if series.is_empty():
        return math.nan, math.nan
    steps = series["step"].to_numpy()
    i = int(np.searchsorted(steps, step))
    candidates = [j for j in (i - 1, i) if 0 <= j < len(steps)]
    j = min(candidates, key=lambda k: abs(steps[k] - step))
    return float(series["value"][j]), float(steps[j])


def ema_smooth(values: np.ndarray, weight: float) -> np.ndarray:
    """Smooth a curve the way TensorBoard does (debiased EMA).

    Args:
        values: Raw values; non-finite values are passed through.
        weight: Smoothing factor in ``[0, 1)``; 0 returns the input.

    Returns:
        Smoothed values of the same length.
    """
    values = np.asarray(values, dtype=float)
    if weight <= 0 or len(values) == 0:
        return values
    out = np.empty_like(values)
    last, n = 0.0, 0
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = v
            continue
        n += 1
        last = last * weight + (1 - weight) * v
        out[i] = last / (1 - weight**n)
    return out


def _hparam_series(name: str, values: list) -> pl.Series:
    """Build one hyperparameter column with a type that fits every run.

    Numbers stay numeric (so filters compare them as numbers); a column
    that mixes numbers and text becomes text.
    """
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, bool) for v in present):
        return pl.Series(name, values, dtype=pl.Boolean)
    numeric = all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in present
    )
    if present and numeric:
        if all(isinstance(v, int) for v in present):
            return pl.Series(name, values, dtype=pl.Int64)
        return pl.Series(name, values, dtype=pl.Float64)
    return pl.Series(
        name, [None if v is None else str(v) for v in values], pl.String
    )


def hparam_frame(hparams: dict[int, dict]) -> pl.DataFrame:
    """Expand the hyperparameters of every run into columns.

    Args:
        hparams: Mapping of run id to its flat hyperparameters.

    Returns:
        DataFrame with an ``id`` column and one column per hyperparameter,
        sorted by name.
    """
    ids = list(hparams)
    keys = sorted({k for hp in hparams.values() for k in hp})
    columns = [pl.Series("id", ids, dtype=pl.Int64)]
    columns += [
        _hparam_series(k, [hparams[i].get(k) for i in ids]) for k in keys
    ]
    return pl.DataFrame(columns)


def varying_columns(
    frame: pl.DataFrame, exclude: tuple[str, ...] = ("id",)
) -> list[str]:
    """List columns whose values differ between rows.

    Args:
        frame: Any DataFrame.
        exclude: Columns to skip.

    Returns:
        Names of columns with more than one distinct value (missing counts
        as a value).
    """
    return [
        c
        for c in frame.columns
        if c not in exclude and frame[c].cast(pl.String).n_unique() > 1
    ]


def parse_filter(expr: str) -> list[tuple[str, str, str]]:
    """Parse a filter such as ``optimizer.lr < 1e-3 and model == resnet``.

    Terms are joined with ``and``. Operators: ``==`` (or ``=``), ``!=``,
    ``<``, ``<=``, ``>``, ``>=`` and ``~`` (case-insensitive regex match).
    Keys with spaces or operators can be wrapped in backticks.

    Args:
        expr: The filter text.

    Returns:
        List of ``(key, operator, value)``.

    Raises:
        ValueError: If a term cannot be parsed.
    """
    terms = []
    for part in re.split(r"\s+and\s+", expr.strip(), flags=re.IGNORECASE):
        if not part.strip():
            continue
        match = FILTER_TERM.match(part)
        if not match:
            raise ValueError(
                f"Can't read {part!r}. Use: key op value, e.g. lr < 1e-3"
            )
        op = "==" if match["op"] == "=" else match["op"]
        value = match["val"].strip().strip("'\"")
        terms.append((match["key"].strip(), op, value))
    return terms


def _compare(x, op: str, target: str) -> bool:
    if is_missing(x):
        return op == "!="
    if op == "~":
        return re.search(target, str(x), re.IGNORECASE) is not None
    try:
        t_num = float(target)
        numeric = isinstance(x, int | float) and not isinstance(x, bool)
    except ValueError:
        numeric = False
    if numeric:
        a, b = float(x), t_num
    else:
        a = str(x).lower() if isinstance(x, bool) else str(x)
        b = target.lower() if isinstance(x, bool) else target
    return {
        "==": a == b,
        "!=": a != b,
        "<": a < b,
        "<=": a <= b,
        ">": a > b,
        ">=": a >= b,
    }[op]


def apply_filter(
    frame: pl.DataFrame, terms: list[tuple[str, str, str]]
) -> pl.Series:
    """Evaluate parsed filter terms against a table.

    Args:
        frame: Table whose columns the keys refer to.
        terms: Output of :func:`parse_filter`.

    Returns:
        Boolean Series with one value per row of ``frame``.

    Raises:
        ValueError: If a key is not a column (the message suggests close
            ones) or a regex is invalid.
    """
    mask = [True] * frame.height
    for key, op, target in terms:
        if key not in frame.columns:
            close = difflib.get_close_matches(key, frame.columns, 3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(f"Unknown column {key!r}.{hint}")
        try:
            hits = [_compare(x, op, target) for x in frame[key].to_list()]
        except (TypeError, re.error) as exc:
            raise ValueError(
                f"Can't evaluate {key} {op} {target}: {exc}"
            ) from exc
        mask = [a and b for a, b in zip(mask, hits)]
    return pl.Series("mask", mask, dtype=pl.Boolean)


def seed_key(name: str, pattern: str) -> str:
    """Remove the seed part of a run name so seeds of a config share a key.

    Args:
        name: Run name, e.g. ``aug_mixup_seed2``.
        pattern: Regex matching the seed part.

    Returns:
        The name without the seed part, e.g. ``aug_mixup``.
    """
    return re.sub(pattern, "", name).strip("_-/ ") or name


def hparam_key(hparams: dict) -> str:
    """Summarize hyperparameters, ignoring seeds, as a grouping key.

    Args:
        hparams: Flat hyperparameters of a run.

    Returns:
        A stable string of the non-seed hyperparameters.
    """
    items = sorted(
        (k, v)
        for k, v in hparams.items()
        if "seed" not in k.lower().split(".")[-1]
    )
    return ", ".join(f"{k}={v}" for k, v in items) or "(no hparams)"


def concurrency(segments: list[list[float]]) -> pl.DataFrame:
    """Count how many runs were active over time.

    Args:
        segments: All ``[start, end]`` activity segments.

    Returns:
        Step-shaped DataFrame with ``time`` (seconds) and ``running``.
    """
    events = sorted(
        [(s, 1) for s, _ in segments] + [(e, -1) for _, e in segments]
    )
    times, running, count = [], [], 0
    for t, delta in events:
        count += delta
        times.append(t)
        running.append(count)
    return pl.DataFrame(
        {"time": times, "running": running},
        schema={"time": pl.Float64, "running": pl.Int64},
    )


def daily_hours(segments: list[list[float]], tz) -> dict:
    """Split activity segments into compute hours per calendar day.

    Args:
        segments: ``[start, end]`` pairs in seconds.
        tz: A ``tzinfo`` for the calendar days.

    Returns:
        Mapping of ``datetime.date`` to hours.
    """
    hours: dict = {}
    for start, end in segments:
        t = datetime.fromtimestamp(start, tz)
        stop = datetime.fromtimestamp(end, tz)
        while t < stop:
            midnight = (t + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            chunk_end = min(midnight, stop)
            chunk = (chunk_end - t).total_seconds() / 3600
            hours[t.date()] = hours.get(t.date(), 0.0) + chunk
            t = chunk_end
    return hours


def format_duration(seconds) -> str:
    """Format seconds as a short human duration like ``3h 12m``.

    Args:
        seconds: Duration in seconds (missing gives an empty string).

    Returns:
        Formatted duration.
    """
    if is_missing(seconds):
        return ""
    seconds = int(round(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_bytes(n) -> str:
    """Format a byte count like ``1.2 GB``.

    Args:
        n: Number of bytes (missing gives an empty string).

    Returns:
        Formatted size.
    """
    if is_missing(n):
        return ""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def format_ago(timestamp, now: float) -> str:
    """Format how long ago a timestamp was, like ``2d ago``.

    Args:
        timestamp: Seconds since the epoch.
        now: Current time in seconds.

    Returns:
        Relative time.
    """
    if is_missing(timestamp):
        return ""
    delta = max(0.0, now - float(timestamp))
    if delta < 60:
        return "just now"
    return format_duration(delta).split(" ")[0] + " ago"


def format_value(x) -> str:
    """Format a metric value with 4 significant digits.

    Args:
        x: Number (missing gives an empty string).

    Returns:
        Formatted value.
    """
    if is_missing(x):
        return ""
    return f"{x:.4g}"


def change_from(value, base) -> float | None:
    """Return ``value - base``, or None if either is missing.

    Args:
        value: A number or None.
        base: The baseline's number or None.

    Returns:
        The difference, or None.
    """
    if is_missing(value) or is_missing(base):
        return None
    return float(value) - float(base)


def format_change(diff) -> str:
    """Format a difference from the baseline with its sign, e.g. ``+0.012``.

    Args:
        diff: The difference (missing gives an empty string).

    Returns:
        ``+x`` or ``−x`` (with a minus sign), or ``±0``.
    """
    if is_missing(diff):
        return ""
    if math.isclose(diff, 0, abs_tol=1e-12):
        return "±0"
    return ("+" if diff > 0 else "−") + format_value(abs(diff))


def change_kind(diff, direction: str) -> str:
    """Tell whether a difference from the baseline is better or worse.

    Args:
        diff: The difference (value minus baseline), or None.
        direction: ``"max"`` or ``"min"``.

    Returns:
        ``"good"``, ``"bad"``, or ``""`` for no difference.
    """
    if is_missing(diff) or math.isclose(diff, 0, abs_tol=1e-12):
        return ""
    return "good" if (diff > 0) == (direction == "max") else "bad"


def to_markdown(df: pl.DataFrame) -> str:
    """Render a table as GitHub-flavored Markdown.

    Args:
        df: Table of already formatted values.

    Returns:
        Markdown text.
    """
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    for row in df.iter_rows():
        cells = ("" if v is None else str(v).replace("|", "\\|") for v in row)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _tex(text) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def to_latex(
    df: pl.DataFrame, bold: set[tuple[int, str]] | None = None
) -> str:
    """Render a table as a LaTeX ``tabular`` (booktabs style).

    Args:
        df: Table of already formatted values.
        bold: Optional ``(row, column)`` cells to set in bold.

    Returns:
        LaTeX source.
    """
    bold = bold or set()
    align = "l" + "r" * (len(df.columns) - 1)
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_tex(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(df.iter_rows()):
        cells = []
        for column, value in zip(df.columns, row):
            text = "" if value is None else value
            cell = _tex(text).replace("±", r"$\pm$").replace("−", "$-$")
            if (i, column) in bold:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _short(value: float) -> str:
    """Format a setting: whole numbers in full, others to 4 digits."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


def explorer_axis(values: list) -> dict:
    """Describe one setting as an axis of the parameter explorer.

    Numbers with no missing values become a numeric axis, on a log scale
    when they are all positive and span two decades or more (learning
    rates, weight decay). Anything else (text, booleans, or numbers with
    gaps) becomes a categorical axis whose positions are ``0, 1, …`` in
    sorted order, with missing values shown as ``–`` at the end.

    Args:
        values: One value per run (None or NaN for missing).

    Returns:
        Dict with ``kind`` (``"number"`` or ``"category"``), ``log``,
        ``positions`` (one float per run, as plotted), ``tickvals`` and
        ``ticktext`` (axis ticks; empty for plain numeric axes) and
        ``labels`` (each run's value as text, for hovers).
    """
    present = [v for v in values if not is_missing(v)]
    numeric = bool(present) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in present
    )
    labels = [
        "–" if is_missing(v) else (_short(v) if numeric else str(v))
        for v in values
    ]
    if numeric and len(present) == len(values):
        nums = [float(v) for v in values]
        log = min(nums) > 0 and max(nums) / min(nums) >= 100
        if not log:
            return {
                "kind": "number",
                "log": False,
                "positions": nums,
                "tickvals": [],
                "ticktext": [],
                "labels": labels,
            }
        decades = range(
            math.floor(math.log10(min(nums))),
            math.ceil(math.log10(max(nums))) + 1,
        )
        return {
            "kind": "number",
            "log": True,
            "positions": [math.log10(v) for v in nums],
            "tickvals": list(decades),
            "ticktext": [f"1e{d}" for d in decades],
            "labels": labels,
        }
    if numeric:
        order = sorted({float(v) for v in present})
        names = [_short(v) for v in order]
    else:
        names = sorted({str(v) for v in present})
    if len(present) < len(values):
        names.append("–")
    slot = {name: i for i, name in enumerate(names)}
    return {
        "kind": "category",
        "log": False,
        "positions": [float(slot[label]) for label in labels],
        "tickvals": list(range(len(names))),
        "ticktext": names,
        "labels": labels,
    }
