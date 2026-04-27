#!/usr/bin/env python3
"""
Surya Layout Detection Visualizer
==================================
Renders PDF pages with surya's detected bounding boxes overlaid.
Outputs annotated PNG images so you can see exactly what surya detects.

Usage:
    python test_surya.py <pdf_path> [--pages 1,2,3] [--dpi 150] [--output_dir ./surya_debug]

Examples:
    python test_surya.py data/input/Economics_New_Phase.pdf
    python test_surya.py data/input/Economics_New_Phase.pdf --pages 1,2
    python test_surya.py data/input/Economics_New_Phase.pdf --dpi 200 --output_dir ./debug
"""

import argparse
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

# ── Color palette for different label types ──────────────────────────────
LABEL_COLORS = {
    "Text":           (0, 180, 0, 80),        # green
    "TextBlock":      (0, 200, 100, 80),       # teal
    "Title":          (255, 50, 50, 80),       # red
    "SectionHeader":  (255, 140, 0, 80),       # orange
    "ListItem":       (100, 100, 255, 80),     # blue
    "Table":          (200, 0, 200, 80),       # purple
    "Figure":         (0, 200, 200, 80),       # cyan
    "Caption":        (180, 180, 0, 80),       # yellow
    "Footnote":       (255, 100, 100, 80),     # salmon
    "PageFooter":     (200, 100, 50, 80),      # brown
    "PageHeader":     (150, 150, 150, 80),     # gray
    "Formula":        (50, 50, 200, 80),       # dark blue
    "Picture":        (0, 150, 150, 80),       # dark cyan
}
DEFAULT_COLOR = (128, 128, 128, 80)

BORDER_COLORS = {k: (r, g, b, 255) for k, (r, g, b, _) in LABEL_COLORS.items()}
DEFAULT_BORDER = (128, 128, 128, 255)


def get_label_color(label: str):
    """Get fill and border colors for a label."""
    fill = LABEL_COLORS.get(label, DEFAULT_COLOR)
    border = BORDER_COLORS.get(label, DEFAULT_BORDER)
    return fill, border


def run_surya(images: list[Image.Image]):
    """Run surya layout detection on a list of PIL images."""
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor

    print("  Loading surya models...")
    foundation_pred = FoundationPredictor()
    layout_pred = LayoutPredictor(foundation_pred)

    print(f"  Detecting layout on {len(images)} page(s)...")
    results = layout_pred(images, top_k=20)
    return results


def annotate_page(
    page_img: Image.Image,
    layout_result,
    page_rect,
    page_idx: int,
    page_spans: list[dict],
):
    """
    Draw surya bounding boxes on the page image with labels.
    Also marks orphaned text lines (not covered by any region).
    Returns the annotated image and a text report.
    """
    img = page_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = img.size
    scale_x = page_rect.width / img_w
    scale_y = page_rect.height / img_h

    # Try to load a readable font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    report_lines = [f"\n{'='*60}", f"PAGE {page_idx + 1}", f"{'='*60}"]
    report_lines.append(f"  Page size: {page_rect.width:.0f}×{page_rect.height:.0f} pts")
    report_lines.append(f"  Image size: {img_w}×{img_h} px")
    report_lines.append(f"  Regions detected: {len(layout_result.bboxes)}")

    # ── Draw each surya region ───────────────────────────────────────
    regions_pdf = []  # for orphan detection later
    for i, box in enumerate(sorted(layout_result.bboxes, key=lambda b: b.polygon[0][1])):
        label = box.label
        xs = [p[0] for p in box.polygon]
        ys = [p[1] for p in box.polygon]

        # Image pixel coordinates
        ix0, iy0 = int(min(xs)), int(min(ys))
        ix1, iy1 = int(max(xs)), int(max(ys))

        # PDF point coordinates
        px0 = min(xs) * scale_x
        py0 = min(ys) * scale_y
        px1 = max(xs) * scale_x
        py1 = max(ys) * scale_y
        regions_pdf.append((py0 - 5, py1 + 5, px0 - 20, px1 + 20))

        fill, border = get_label_color(label)

        # Draw filled rectangle
        draw.rectangle([ix0, iy0, ix1, iy1], fill=fill, outline=border, width=2)

        # Draw label
        label_text = f"{i}: {label}"
        draw.text((ix0 + 4, iy0 + 2), label_text, fill=(0, 0, 0, 255), font=font)

        # Get text preview from spans
        matching = [s for s in page_spans
                    if py0 - 5 <= (s["y0"] + s["y1"]) / 2 <= py1 + 5
                    and px0 - 20 <= (s["x0"] + s["x1"]) / 2 <= px1 + 20]
        matching.sort(key=lambda s: (s["y0"], s["x0"]))
        text_preview = " ".join(s["text"].strip() for s in matching[:3])[:80]
        line_count = len(matching)

        report_lines.append(f"\n  Region {i}: {label}")
        report_lines.append(f"    PDF coords: ({px0:.0f},{py0:.0f}) → ({px1:.0f},{py1:.0f})")
        report_lines.append(f"    Image coords: ({ix0},{iy0}) → ({ix1},{iy1})")
        report_lines.append(f"    Lines matched: {line_count}")
        report_lines.append(f"    Text: \"{text_preview}...\"")

    # ── Detect orphaned lines ────────────────────────────────────────
    orphans = []
    for span in page_spans:
        cy = (span["y0"] + span["y1"]) / 2
        cx = (span["x0"] + span["x1"]) / 2
        covered = any(ry0 <= cy <= ry1 and rx0 <= cx <= rx1
                      for ry0, ry1, rx0, rx1 in regions_pdf)
        if not covered:
            orphans.append(span)
            # Draw orphan marker on image (red X)
            ox = int(span["x0"] / scale_x)
            oy = int(span["y0"] / scale_y)
            ox1 = int(span["x1"] / scale_x)
            oy1 = int(span["y1"] / scale_y)
            draw.rectangle([ox, oy, ox1, oy1], outline=(255, 0, 0, 200), width=2)
            draw.text((ox + 2, oy - 14), "ORPHAN", fill=(255, 0, 0, 255), font=font_sm)

    if orphans:
        report_lines.append(f"\n  ⚠️  ORPHANED LINES: {len(orphans)}")
        for o in orphans:
            report_lines.append(f"    y={o['y0']:.1f}  \"{o['text'][:70]}\"")
    else:
        report_lines.append(f"\n  ✅ No orphaned lines")

    # Composite
    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result, "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize surya layout detection on PDF pages")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma-separated page numbers (1-indexed). Default: all pages")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Render DPI (default: 150)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for annotated images (default: ./surya_debug)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("./test_surya_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open PDF
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    # Parse page selection
    if args.pages:
        page_indices = [int(p) - 1 for p in args.pages.split(",")]
    else:
        page_indices = list(range(total_pages))

    print(f"📄 {pdf_path.name}: {total_pages} pages, processing {len(page_indices)}")

    # ── Render pages ─────────────────────────────────────────────────
    images = []
    page_rects = []
    all_page_spans = []

    for pg_idx in page_indices:
        page = doc[pg_idx]
        pix = page.get_pixmap(dpi=args.dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        page_rects.append(page.rect)

        # Extract text spans with coordinates
        spans = []
        text_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                lb = line["bbox"]
                line_spans = line.get("spans", [])
                lt = "".join(s["text"] for s in line_spans)
                total_chars = sum(len(s["text"]) for s in line_spans)
                avg_size = (sum(s["size"] * len(s["text"]) for s in line_spans)
                            / max(total_chars, 1))
                if lt.strip():
                    spans.append({
                        "text": lt, "y0": lb[1], "y1": lb[3],
                        "x0": lb[0], "x1": lb[2], "font_size": avg_size,
                    })
        all_page_spans.append(spans)

    # ── Run surya ────────────────────────────────────────────────────
    layout_results = run_surya(images)

    # ── Annotate and save ────────────────────────────────────────────
    full_report = [f"Surya Layout Analysis: {pdf_path.name}"]
    full_report.append(f"DPI: {args.dpi}")

    for i, pg_idx in enumerate(page_indices):
        annotated, report = annotate_page(
            images[i], layout_results[i], page_rects[i],
            pg_idx, all_page_spans[i],
        )
        full_report.append(report)

        out_path = output_dir / f"page_{pg_idx + 1:03d}.png"
        annotated.save(str(out_path))
        print(f"  💾 Saved: {out_path}")

    # Save text report
    report_path = output_dir / "report.txt"
    report_text = "\n".join(full_report)
    report_path.write_text(report_text)
    print(f"\n  📝 Report: {report_path}")
    print(report_text)


if __name__ == "__main__":
    main()
