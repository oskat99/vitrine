#!/usr/bin/env python3
"""Generate Vitrine alternate Spotify icons from the user's extracted original icon.

The source is kept untouched. Background and foreground colors are sampled from the
source artwork, then each pixel's coverage is reconstructed along the original
foreground/background color axis so antialiased logo edges remain smooth.
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
# Use the four corners to identify the flat background without assuming its color.
corners = np.concatenate((pixels[:32, :32].reshape(-1, 3),
                           pixels[:32, -32:].reshape(-1, 3),
                           pixels[-32:, :32].reshape(-1, 3),
                           pixels[-32:, -32:].reshape(-1, 3)))
background = np.median(corners, axis=0)

# The Spotify mark is the darkest substantial region in the extracted artwork.
distance = np.linalg.norm(pixels - background, axis=2)
candidates = pixels[distance > 35]
if candidates.size == 0:
    raise SystemExit("Could not separate the logo from the background")
luma = candidates @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
foreground = np.median(candidates[luma <= np.percentile(luma, 1)], axis=0)

axis = foreground - background
denominator = float(axis @ axis)
if denominator < 1000:
    raise SystemExit("Source logo/background colors are too similar")

# Recover logo coverage from the original two-color artwork, including antialiasing.
coverage = np.clip(((pixels - background) @ axis) / denominator, 0.0, 1.0)
coverage = coverage[..., None]

OUT.mkdir(parents=True, exist_ok=True)
for name, (background_hex, foreground_hex) in COLORS.items():
    bg, fg = rgb(background_hex), rgb(foreground_hex)
    result = np.rint(bg[None, None, :] * (1.0 - coverage) +
                     fg[None, None, :] * coverage).clip(0, 255).astype(np.uint8)
    Image.fromarray(result, "RGB").save(OUT / f"{name}.png", optimize=True)

print("Generated:", ", ".join(f"{name}.png" for name in COLORS))
