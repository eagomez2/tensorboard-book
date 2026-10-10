"""Fields: the values you add to runs by hand.

A field is a column you define: its name, an optional group (to keep
related fields together), a type, and for numbers a unit, a scale prefix,
decimals and whether higher or lower is better. Choice and tags fields
have an ordered list of options; an option's position gives its pill
color, so a value keeps its color for good.

The run tags are a field too, the built-in ``tags`` field, kept in their
own table so filters and the search work on them as before.

This module holds the definitions and the rules for reading, checking and
showing values. ``db`` stores them.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import re
from dataclasses import dataclass, field, replace

TYPES = ("number", "text", "choice", "tags", "bool", "date", "link")
DIRECTIONS = ("higher", "lower")
# The prefixes a number can be shown in. Values are stored as they are, so
# ranking and comparisons don't depend on the scale.
SCALES = {
    "": 1,
    "k": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
    "m": 10**-3,
    "µ": 10**-6,
    "n": 10**-9,
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
}
# Other spellings accepted when typing a value, such as 4.2u for 4.2µ.
_PREFIX_ALIASES = {"u": "µ", "μ": "µ", "K": "k"}
_NUMBER = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(Ki|Mi|Gi|Ti|[kKMGTPmµμun])?\s*$"
)
# Named Streamlit colors for option pills, in the order options get them.
PILL_COLORS = ("blue", "green", "orange", "violet", "red", "yellow", "gray")
# The built-in field that holds the run tags.
TAGS = "Tags"
# Types a field can't change to or from while it has values, because the
# stored form is different (a list).
LIST_TYPES = {"tags"}


@dataclass(frozen=True)
class Field:
    """A column of values you add to runs.

    Attributes:
        name: Name, unique within its group.
        group: Group that keeps related fields together, may be empty.
        type: One of ``TYPES``.
        description: Help text shown in forms.
        unit: Unit of a number, shown next to the name.
        scale: Prefix numbers are shown in, a key of ``SCALES``.
        decimals: Digits after the decimal point, or None for 4
            significant digits.
        direction: ``"higher"`` or ``"lower"`` for numbers that rank, else
            None.
        options: The values of a choice or tags field, in order.
        visible: Whether the field is in **Show** by default.
        position: Sort key for ordering fields.
        builtin: True for the ``tags`` field, which can't be renamed,
            retyped or deleted.
        runs: How many runs have a value.
    """

    name: str
    group: str = ""
    type: str = "text"
    description: str = ""
    unit: str = ""
    scale: str = ""
    decimals: int | None = None
    direction: str | None = None
    options: tuple[str, ...] = ()
    visible: bool = True
    position: int = 0
    builtin: bool = False
    runs: int = field(default=0, compare=False)

    @property
    def key(self) -> str:
        """The name with its group, as filters and "Show" options use it."""
        return f"{self.group}/{self.name}" if self.group else self.name

    @property
    def is_number(self) -> bool:
        """Whether the field holds numbers."""
        return self.type == "number"

    @property
    def has_options(self) -> bool:
        """Whether the field has a list of options (choice or tags)."""
        return self.type in ("choice", "tags")

    @property
    def shown_unit(self) -> str:
        """The unit with the scale prefix, such as ``"GFLOPs"``."""
        if not self.is_number or not self.scale:
            return self.unit
        return f"{self.scale}{self.unit}"

    @property
    def arrow(self) -> str:
        """↑ or ↓ for a number that ranks, else ``""``."""
        return {"higher": "↑", "lower": "↓"}.get(self.direction or "", "")

    def label(self, with_group: bool = False, arrow: bool = True) -> str:
        """Build the header of this field, such as ``"eval · WER (%) ↓"``.

        Args:
            with_group: Put the group in front of the name.
            arrow: Add the direction arrow.

        Returns:
            The header text.
        """
        text = self.name
        if with_group and self.group:
            text = f"{self.group} · {text}"
        if self.shown_unit:
            text = f"{text} ({self.shown_unit})"
        if arrow and self.arrow:
            text = f"{text} {self.arrow}"
        return text

    def option_colors(self) -> dict[str, str]:
        """Return the pill color of each option, from its position."""
        return {
            option: PILL_COLORS[i % len(PILL_COLORS)]
            for i, option in enumerate(self.options)
        }

    def format(self, value) -> str:
        """Format a stored value for display.

        Args:
            value: The value as stored, or None.

        Returns:
            The value as text, ``""`` when missing.
        """
        if value is None:
            return ""
        if self.is_number:
            return format_number(value, self.scale, self.decimals)
        if self.type == "tags":
            return ", ".join(value)
        if self.type == "bool":
            return "yes" if value else "no"
        return str(value)


def format_number(value, scale: str = "", decimals: int | None = None) -> str:
    """Format a stored number in a scale and with some decimals.

    Args:
        value: The number as stored (None and NaN give ``""``).
        scale: A key of ``SCALES``.
        decimals: Digits after the point, or None for 4 significant
            digits.

    Returns:
        The number as text.
    """
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return ""
    number /= SCALES.get(scale or "", 1)
    if decimals is not None:
        return f"{number:.{decimals}f}"
    return f"{number:.4g}"


def to_number(raw) -> float:
    """Read a number, which may end in a prefix such as ``"4.2G"``.

    Args:
        raw: A number or text.

    Returns:
        The number, with the prefix applied exactly.

    Raises:
        ValueError: If it is not a finite number.
    """
    if isinstance(raw, bool):
        raise ValueError(f"{raw!r} is not a number")
    if isinstance(raw, str):
        match = _NUMBER.match(raw)
        if match is None:
            raise ValueError(f"{raw!r} is not a number")
        digits, prefix = match.groups()
        if prefix:
            factor = SCALES[_PREFIX_ALIASES.get(prefix, prefix)]
            # Decimal keeps 4.2G at exactly 4200000000.
            return float(decimal.Decimal(digits) * decimal.Decimal(factor))
        raw = digits
    try:
        number = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{raw!r} is not a number") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{raw!r} is not a finite number")
    return number


def coerce(fld: Field, raw):
    """Convert input from a form, a table or a file into a stored value.

    Args:
        fld: The field the value is for.
        raw: The input.

    Returns:
        The value to store: a float, text, a list of text (tags), a bool,
        or an ISO date; None when the input is empty.

    Raises:
        ValueError: If the input doesn't fit the field's type. The message
            names the field.
    """
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    if isinstance(raw, str) and fld.type != "text":
        raw = raw.strip()
        if not raw:
            return None
    try:
        return _coerce(fld.type, raw)
    except ValueError as exc:
        raise ValueError(f"{fld.name}: {exc}") from None


def _coerce(kind: str, raw):
    """Convert one non-empty value for a type (see :func:`coerce`)."""
    if kind == "number":
        return to_number(raw)
    if kind in ("text", "link", "choice"):
        if isinstance(raw, list):
            raw = ", ".join(str(r) for r in raw)
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        text = str(raw).strip()
        return text or None
    if kind == "tags":
        if isinstance(raw, str):
            items = raw.split(",")
        elif isinstance(raw, (list, tuple)):
            items = raw
        else:
            items = [raw]
        items = [str(i).strip() for i in items if str(i).strip()]
        return list(dict.fromkeys(items)) or None
    if kind == "bool":
        if isinstance(raw, str):
            lowered = raw.lower()
            if lowered in ("true", "yes", "y", "1"):
                return True
            if lowered in ("false", "no", "n", "0"):
                return False
            raise ValueError(f"{raw!r} is not yes or no")
        return bool(raw)
    if kind == "date":
        if isinstance(raw, dt.datetime):
            return raw.date().isoformat()
        if isinstance(raw, dt.date):
            return raw.isoformat()
        try:
            return dt.date.fromisoformat(str(raw)[:10]).isoformat()
        except ValueError:
            raise ValueError(f"{raw!r} is not a date (YYYY-MM-DD)") from None
    raise ValueError(f"unknown field type {kind!r}")


def encode(fld: Field, value):
    """Return a stored value in the form SQLite keeps it.

    Tags are kept as JSON text and yes/no as 1 or 0; everything else as
    it is.

    Args:
        fld: The field.
        value: A value from :func:`coerce`.

    Returns:
        A value SQLite can store.
    """
    if fld.type == "tags":
        return json.dumps(value)
    if fld.type == "bool":
        return int(bool(value))
    return value


def decode(kind: str, stored):
    """Read a value back from the database (the reverse of :func:`encode`).

    Args:
        kind: The field type.
        stored: The value SQLite returned.

    Returns:
        The value as :func:`coerce` gives it.
    """
    if stored is None:
        return None
    if kind == "tags":
        try:
            value = json.loads(stored)
        except (TypeError, ValueError):
            value = [str(stored)]
        return value if isinstance(value, list) else [str(value)]
    if kind == "bool":
        return bool(stored)
    if kind == "number":
        try:
            return float(stored)
        except (TypeError, ValueError):
            return None
    return stored


def convert(old: Field, new_type: str, value):
    """Convert a stored value when a field changes type.

    Args:
        old: The field before the change.
        new_type: The type it changes to.
        value: A decoded value of ``old``.

    Returns:
        The value in the new type, or None for an empty value.

    Raises:
        ValueError: If the value doesn't convert, such as ``imagenet-1k``
            to a number.
    """
    if value is None or old.type == new_type:
        return value
    if old.type == "tags" and new_type != "tags":
        if len(value) > 1:
            raise ValueError(f"{', '.join(value)} is more than one value")
        value = value[0] if value else None
        if value is None:
            return None
    if old.type == "bool" and new_type in ("text", "choice"):
        value = "yes" if value else "no"
    if old.type == "number" and new_type in ("text", "choice", "tags"):
        value = format_number(value, "", None if value % 1 else 0)
    return _coerce(new_type, value)


def metric_group(tag: str) -> str:
    """Return the group of a logged metric: its tag up to the last ``/``.

    Args:
        tag: A scalar tag such as ``val/acc`` or ``eval/libri/wer``.

    Returns:
        ``val`` or ``eval/libri``; ``""`` for a tag without ``/``.
    """
    return tag.rpartition("/")[0]


def with_options(fld: Field, values) -> Field:
    """Return the field with new values appended to its options.

    Typing a value that isn't an option yet adds it.

    Args:
        fld: A choice or tags field.
        values: A value, a list of values, or None.

    Returns:
        The field, with any new options at the end.
    """
    if not fld.has_options or values is None:
        return fld
    items = values if isinstance(values, list) else [values]
    new = [str(v) for v in items if str(v) not in fld.options]
    if not new:
        return fld
    return replace(fld, options=(*fld.options, *dict.fromkeys(new)))


def input_text(value) -> str:
    """Write a stored number the way a form shows it for editing.

    Numbers are written as stored, without the scale, and whole numbers in
    full rather than as ``2.56e+07``.

    Args:
        value: A stored number, or None.

    Returns:
        Text such as ``"25600000"`` or ``"0.25"``; ``""`` when empty.
    """
    if value is None:
        return ""
    number = float(value)
    if number.is_integer() and abs(number) < 1e18:
        return str(int(number))
    return f"{number:.15g}"
