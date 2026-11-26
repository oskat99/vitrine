#!/usr/bin/env python3
"""Generate flat circular Spotify-mark icons from the user's extracted source.

The source is a glossy green disc with dark Spotify wave cutouts on a dark
background. Rebuild the mark as a clean flat circle: the circle is the chosen
foreground colour, and the three wave cutouts are the chosen background colour.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "spotify-icon-original.png"
OUT = ROOT / "icons"
COLORS = {
    "Black_White": ("#040000", "#fdffff"),
    "White_Black": ("#fdffff", "#040000"),
    "Black_Green": ("#040000", "#00dc4c"),
    "Blue_Black": ("#289dfb", "#040000"),
}

def rgb(hex_color):
    return np.array([int(hex_color[i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32)

if not SOURCE.is_file():
    raise SystemExit(f"Missing original icon: {SOURCE}")

image = Image.open(SOURCE).convert("RGB")
if image.size != (1024, 1024):
    raise SystemExit(f"Expected 1024x1024 source, got {image.size}")

pixels = np.asarray(image, dtype=np.float32)
yy, xx = np.mgrid[:1024, :1024]
radius = np.sqrt((xx - 512.0) ** 2 + (yy - 512.0) ** 2)

# Recreate the central disc as a clean circle, excluding the source's bevel,
# glow and outer shadow. A soft one-pixel edge retains smooth antialiasing.
circle_radius = 410.0
circle = np.clip(circle_radius + 0.5 - radius, 0.0, 1.0)

# The three wave cutouts are near-black. Detect them only in the central
# Spotify-mark area; this avoids mistaking the dark outer background or disc
# rim for holes. Luma interpolation preserves their antialiased edges.
luma = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
wave_area = ((radius < 315.0) &
             (yy > 285) & (yy < 750))
wave_holes = np.clip((105.0 - luma) / 65.0, 0.0, 1.0)
wave_holes *= wave_area.astype(np.float32)

# Circle is the coloured mark; the wave strokes are transparent cutouts
# showing the flat background colour through them.
coverage = circle * (1.0 - wave_holes)
coverage = coverage[..., None]

if float(coverage.max()) < 0.99 or float((wave_holes > 0.8).sum()) < 1000:
    raise SystemExit("Could not reliably identify the circular Spotify mark and wave cutouts")

OUT.mkdir(parents=True, exist_ok=True)
for name, (background_hex, mark_hex) in COLORS.items():
    bg, mark = rgb(background_hex), rgb(mark_hex)
    result = np.rint(bg[None, None, :] * (1.0 - coverage) +
                     mark[None, None, :] * coverage).clip(0, 255).astype(np.uint8)
    Image.fromarray(result, "RGB").save(OUT / f"{name}.png", optimize=True)

print("Generated flat circular Spotify icons:", ", ".join(f"{name}.png" for name in COLORS))
