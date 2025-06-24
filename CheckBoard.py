#!/usr/bin/env python3
"""
separate_checkerboards.py

Generate individual black-and-white checkerboard patterns for each square size on A4 paper,
and directly save them as CMYK PDF files at exact net dimensions (21×29.7 cm portrait or 29.7×21 cm landscape).

Usage:
    python separate_checkerboards.py \
        --sizes-cm 1,2,3,4,5 \
        --margin-mm 10 --dpi 300 \
        [--landscape] \
        --output-dir ./output
"""
import argparse
import os
import numpy as np
from PIL import Image

# A4 size in mm (portrait)
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

def mm_to_px(value_mm: float, dpi: int) -> int:
    """Convert millimeters to pixels given DPI."""
    return int(round(dpi * value_mm / 25.4))

def generate_checkerboard(
    paper_width_mm: float,
    paper_height_mm: float,
    square_cm: float,
    margin_mm: float,
    dpi: int
) -> Image.Image:
    # Convert dimensions to pixels
    width_px = mm_to_px(paper_width_mm, dpi)
    height_px = mm_to_px(paper_height_mm, dpi)
    margin_px = mm_to_px(margin_mm, dpi)

    # Square size in pixels
    square_px = mm_to_px(square_cm * 10, dpi)

    # Compute number of full cols & rows
    cols = (width_px - 2 * margin_px) // square_px
    rows = (height_px - 2 * margin_px) // square_px
    cols = max(1, cols)
    rows = max(1, rows)

    # Build pattern
    pattern = (np.indices((rows, cols)).sum(axis=0) % 2).astype(np.uint8)
    board = np.kron(pattern, np.ones((square_px, square_px), dtype=np.uint8)) * 255
    zone_w = cols * square_px
    zone_h = rows * square_px
    board_crop = board[:zone_h, :zone_w]

    # Create grayscale canvas and paste
    img = Image.new('L', (width_px, height_px), 255)
    img.paste(Image.fromarray(board_crop), (margin_px, margin_px))
    # Convert to CMYK for PDF
    img_cmyk = img.convert('CMYK')
    return img_cmyk, rows, cols

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate separate checkerboard PDFs on A4 for given square sizes."
    )
    parser.add_argument(
        "--sizes-cm", type=str, default="1,2,3,4,5",
        help="Comma-separated list of square sizes in cm, e.g. '1,2,3,4,5'."
    )
    parser.add_argument(
        "--margin-mm", type=float, default=10.0,
        help="Margin around board in mm."
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Print resolution in DPI."
    )
    parser.add_argument(
        "--landscape", action="store_true",
        help="Generate PDF in landscape orientation (29.7×21 cm)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save output PDF files."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    sizes = [float(s) for s in args.sizes_cm.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # Set orientation dimensions
    if args.landscape:
        pw, ph = A4_HEIGHT_MM, A4_WIDTH_MM
    else:
        pw, ph = A4_WIDTH_MM, A4_HEIGHT_MM

    for size in sizes:
        img_cmyk, rows, cols = generate_checkerboard(
            pw, ph, size, args.margin_mm, args.dpi
        )
        orient = 'landscape' if args.landscape else 'portrait'
        filename = (
            f"checkerboard_A4_{int(size*10)}mm_{int(args.margin_mm)}mm_"
            f"{args.dpi}dpi_{orient}.pdf"
        )
        out_path = os.path.join(args.output_dir, filename)
        img_cmyk.save(out_path, 'PDF', resolution=args.dpi)
        print(f"Saved: {out_path} ({rows}×{cols} squares of {size} cm, {orient}) as CMYK PDF")

if __name__ == "__main__":
    main()
