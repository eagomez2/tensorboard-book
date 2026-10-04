"""The tensorboard-book logo: a stack of matrices on a blue circle.

A 3 x 3 matrix with two slices behind it stands for a tensor. The logo
is the browser tab icon, the sidebar logo and the README logo. It is drawn
in the style of modelboard's: a filled circle with a white drawing, in
blue so it stands apart from modelboard's black and white.
``logo.svg`` is for light backgrounds and ``logo-dark.svg``, a lighter
blue, for dark ones. ``logo.png`` is drawn from the same shapes with
Pillow, so nothing has to be downloaded.

Run ``python -m tensorboard_book.brand`` after changing the drawing; it
also copies the SVGs to the top-level ``assets`` folder for the README.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")
README_ASSETS = Path(__file__).parents[2] / "assets"
LOGO_SVG = ASSETS / "logo.svg"
LOGO_DARK_SVG = ASSETS / "logo-dark.svg"
LOGO_PNG = ASSETS / "logo.png"
BLUE = "#2F6BD8"
DARK_BLUE = "#4C86EC"
WHITE = "#FFFFFF"

# The two slices behind the matrix: x, y, opacity of the outline. Both
# are 24 x 24 squares with rounded corners.
SLICES = ((26.0, 10.0, 0.45), (22.0, 14.0, 0.70))
SLICE_SIZE = 24
SLICE_RADIUS = 3.5
SLICE_STROKE = 2.2
# The front matrix: top-left corner, cell size, gap and corner radius.
CELLS = (14.0, 22.0, 7.0, 1.6, 1.6)


def _num(value: float) -> str:
    """Format a coordinate without needless decimals."""
    return f"{value:g}"


def cells() -> list[tuple[float, float]]:
    """Return the top-left corner of each of the nine matrix cells."""
    x0, y0, size, gap, _ = CELLS
    return [
        (x0 + i * (size + gap), y0 + j * (size + gap))
        for j in range(3)
        for i in range(3)
    ]


def logo_svg(ink: str = BLUE, paper: str = WHITE, size: int = 64) -> str:
    """Draw the logo.

    Args:
        ink: Circle color.
        paper: Color of the matrices.
        size: Width and height in pixels.

    Returns:
        A standalone SVG document.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size}" viewBox="0 0 64 64">',
        f'<circle cx="32" cy="32" r="32" fill="{ink}"/>',
    ]
    for x, y, opacity in SLICES:
        parts.append(
            f'<rect x="{_num(x)}" y="{_num(y)}" width="{SLICE_SIZE}" '
            f'height="{SLICE_SIZE}" rx="{_num(SLICE_RADIUS)}" fill="none" '
            f'stroke="{paper}" stroke-width="{_num(SLICE_STROKE)}" '
            f'stroke-opacity="{_num(opacity)}"/>'
        )
    _, _, cell, _, radius = CELLS
    for x, y in cells():
        parts.append(
            f'<rect x="{_num(x)}" y="{_num(y)}" width="{_num(cell)}" '
            f'height="{_num(cell)}" rx="{_num(radius)}" fill="{paper}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _mix(color: str, over: str, opacity: float) -> tuple[int, int, int]:
    """Blend ``color`` at ``opacity`` over ``over``, as an RGB tuple."""
    a = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(over[i : i + 2], 16) for i in (1, 3, 5)]
    r, g, bl = (round(x * opacity + y * (1 - opacity)) for x, y in zip(a, b))
    return (r, g, bl)


def write_assets() -> None:
    """Write both SVG logos and ``logo.png``, and copy the SVGs."""
    from PIL import Image, ImageDraw

    ASSETS.mkdir(exist_ok=True)
    LOGO_SVG.write_text(logo_svg() + "\n", encoding="utf-8")
    LOGO_DARK_SVG.write_text(logo_svg(ink=DARK_BLUE) + "\n", encoding="utf-8")
    if README_ASSETS.is_dir():
        for path in (LOGO_SVG, LOGO_DARK_SVG):
            shutil.copy(path, README_ASSETS / path.name)
    scale = 16
    big = 64 * scale
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, big - 1, big - 1), fill=BLUE)
    for x, y, opacity in SLICES:
        draw.rounded_rectangle(
            (
                x * scale,
                y * scale,
                (x + SLICE_SIZE) * scale,
                (y + SLICE_SIZE) * scale,
            ),
            radius=SLICE_RADIUS * scale,
            outline=_mix(WHITE, BLUE, opacity),
            width=round(SLICE_STROKE * scale),
        )
    _, _, cell, _, radius = CELLS
    for x, y in cells():
        draw.rounded_rectangle(
            (x * scale, y * scale, (x + cell) * scale, (y + cell) * scale),
            radius=radius * scale,
            fill=WHITE,
        )
    image.resize((256, 256), Image.LANCZOS).save(LOGO_PNG)


if __name__ == "__main__":
    write_assets()
