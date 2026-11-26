#!/usr/bin/env python3
"""Generate clean, flat Vitrine Spotify icons from the user's extracted icon.

The extracted source is a glossy 3-D app icon, not a two-colour flat logo.
Only the dark Spotify wave mark inside the central green disc is retained.
The surrounding disc, glow, shadows, rim and outer background are discarded.
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

# The Spotify waves sit in the centre of the green disc. Restrict extraction
# to that region so the disc's dark rim, outer background and glossy effects
# can never be mistaken for parts of the mark.
yy, xx = np.ogrid[:1024, :1024]
inside_mark_area = (xx - 512) ** 2 + (yy - 512) ** 2 <= 350 ** 2

# The mark itself is near-black; the disc behind it is bright green.
# Convert darkness to coverage so the original antialiased edges survive,
# while faint green glow/shading is rejected instead of becoming a halo.
luma = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
low, high = 25.0, 95.0
coverage = np.clip((high - luma) / (high - low), 0.0, 1.0)
coverage *= inside_mark_area.astype(np.float32)
coverage = coverage[..., None]

if float(coverage.max()) < 0.9 or float((coverage > 0.5).sum()) < 1000:
    raise SystemExit("Could not reliably isolate the Spotify wave mark from the source icon")

OUT.mkdir(parents=True, exist_ok=True)
for name, (background_hex, foreground_hex) in COLORS.items():
    bg, fg = rgb(background_hex), rgb(foreground_hex)
    result = np.rint(bg[None, None, :] * (1.0 - coverage) +
                     fg[None, None, :] * coverage).clip(0, 255).astype(np.uint8)
    Image.fromarray(result, "RGB").save(OUT / f"{name}.png", optimize=True)

print("Generated clean flat icons:", ", ".join(f"{name}.png" for name in COLORS))
