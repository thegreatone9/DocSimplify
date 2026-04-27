#!/usr/bin/env python3
"""
PaddleOCR PP-Structure Layout Detection Visualizer
====================================================
Renders PDF pages with PP-Structure's detected layout boxes overlaid.
Outputs annotated PNG images so you can compare with surya's results.

Usage:
    python test_paddle.py <pdf_path> [--pages 1,2,3] [--dpi 150]

Examples:
    python test_paddle.py data/input/Economics_New_Phase.pdf
    python test_paddle.py data/input/Economics_New_Phase.pdf --pages 1,2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

# ── Color palette for different label types ──────────────────────────────
LABEL_COLORS = {
    "text":       (0, 180, 0, 80),        # green
    "title":      (255, 50, 50, 80),       # red
    "figure":     (0, 200, 200, 80),       # cyan
    "figure_caption": (180, 180, 0, 80),   # yellow
    "table":      (200, 0, 200, 80),       # purple
    "table_caption": (150, 100, 200, 80),  # lavender
    "header":     (150, 150, 150, 80),     # gray
    "footer":     (200, 100, 50, 80),      # brown
    "reference":  (100, 100, 255, 80),     # blue
    "equation":   (50, 50, 200, 80),       # dark blue
    "list":       (0, 200, 100, 80),       # teal
}
DEFAULT_COLOR = (128, 128, 128, 80)


def get_label_color(label: str):
    """Get fill and border colors for a label."""
    label_lower = label.lower()
    fill = LABEL_COLORS.get(label_lower, DEFAULT_COLOR)
    border = tuple(list(fill[:3]) + [255])
    return fill, border


def run_paddle_layout(img_np):
    """Run PaddleOCR PP-Structure layout analysis on a numpy image."""
    from paddleocr import PPStructure

    # layout=True, table=False, ocr=False → layout detection only (fast)
    engine = PPStructure(layout=True, table=False, ocr=False, show_log=False)
    result = engine(img_np)
    return result


def annotate_page(
    page_img: Image.Image,
    paddle_result: list,
    page_rect,
    page_idx: int,
    page_spans: list[dict],
):
    """Draw paddle bounding boxes on the page image with labels."""
    img = page_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = img.size
    scale_x = page_rect.width / img_w
    scale_y = page_rect.height / img_h

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    report_lines = [f"\n{'='*60}", f"PAGE {page_idx + 1}", f"{'='*60}"]
    report_lines.append(f"  Page size: {page_rect.width:.0f}×{page_rect.height:.0f} pts")
    report_lines.append(f"  Image size: {img_w}×{img_h} px")
    report_lines.append(f"  Regions detected: {len(paddle_result)}")

    # ── Draw each paddle region ──────────────────────────────────────
    regions_pdf = []
    for i, region in enumerate(sorted(paddle_result, key=lambda r: r['bbox'][1])):
        label = region.get('type', 'unknown')
        bbox = region['bbox']  # [x0, y0, x1, y1] in image pixels
        ix0, iy0, ix1, iy1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # PDF point coordinates
        px0 = ix0 * scale_x
        py0 = iy0 * scale_y
        px1 = ix1 * scale_x
        py1 = iy1 * scale_y
        regions_pdf.append((py0 - 5, py1 + 5, px0 - 20, px1 + 20))

        fill, border = get_label_color(label)
        draw.rectangle([ix0, iy0, ix1, iy1], fill=fill, outline=border, width=2)

        label_text = f"{i}: {label}"
        draw.text((ix0 + 4, iy0 + 2), label_text, fill=(0, 0, 0, 255), font=font)

        # Get text preview from pymupdf spans
        matching = [s for s in page_spans
                    if py0 - 5 <= (s["y0"] + s["y1"]) / 2 <= py1 + 5
                    and px0 - 20 <= (s["x0"] + s["x1"]) / 2 <= px1 + 20]
        matching.sort(key=lambda s: (s["y0"], s["x0"]))
        text_preview = " ".join(s["text"].strip() for s in matching[:3])[:80]

        report_lines.append(f"\n  Region {i}: {label} (score={region.get('score', 0):.3f})")
        report_lines.append(f"    Image coords: ({ix0},{iy0}) → ({ix1},{iy1})")
        report_lines.append(f"    PDF coords: ({px0:.0f},{py0:.0f}) → ({px1:.0f},{py1:.0f})")
        report_lines.append(f"    Lines matched: {len(matching)}")
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

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result, "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize PaddleOCR PP-Structure layout on PDF pages")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma-separated page numbers (1-indexed). Default: all")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    output_dir = Path("./test_paddle_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    if args.pages:
        page_indices = [int(p) - 1 for p in args.pages.split(",")]
    else:
        page_indices = list(range(total_pages))

    print(f"📄 {pdf_path.name}: {total_pages} pages, processing {len(page_indices)}")

    # ── Render pages and extract spans ───────────────────────────────
    images = []
    page_rects = []
    all_page_spans = []

    for pg_idx in page_indices:
        page = doc[pg_idx]
        pix = page.get_pixmap(dpi=args.dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        page_rects.append(page.rect)

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

    doc.close()

    # ── Run PaddleOCR LayoutDetection on each page ─────────────────
    from paddleocr import LayoutDetection
    print("  Loading LayoutDetection model...")
    engine = LayoutDetection()

    full_report = [f"PaddleOCR LayoutDetection Analysis: {pdf_path.name}"]
    full_report.append(f"DPI: {args.dpi}")

    for i, pg_idx in enumerate(page_indices):
        print(f"  Analyzing page {pg_idx + 1}...")
        img_np = np.array(images[i])
        raw_result = engine.predict(img_np)

        # PaddleOCR v3.5 returns [DetResult], where DetResult has .boxes
        # Each box is a dict: {cls_id, label, score, coordinate: [x0,y0,x1,y1]}
        det = raw_result[0]
        paddle_result = []
        for box in det['boxes']:
            coord = box['coordinate']
            paddle_result.append({
                'bbox': [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])],
                'type': box['label'],
                'score': box['score'],
            })

        annotated, report = annotate_page(
            images[i], paddle_result, page_rects[i],
            pg_idx, all_page_spans[i],
        )
        full_report.append(report)

        out_path = output_dir / f"page_{pg_idx + 1:03d}.png"
        annotated.save(str(out_path))
        print(f"  💾 Saved: {out_path}")

    report_path = output_dir / "report.txt"
    report_text = "\n".join(full_report)
    report_path.write_text(report_text)
    print(f"\n  📝 Report: {report_path}")
    print(report_text)


if __name__ == "__main__":
    main()
