#!/usr/bin/env python3
"""
Split a PDF of item cards into one PDF per card by automatically
detecting card rectangles on each page. Output PDFs are vector-preserving
clips of the original PDF (no raster downsampling) using PyMuPDF's
show_pdf_page with a clip rect.

Detection is done via OpenCV on a rasterized preview of each page, but
the final export uses vector clipping from the source PDF.

Requirements:
  - PyMuPDF (fitz)
  - opencv-python (cv2)
  - numpy

Example:
  python tools/split_cards.py \
    --input "files/items/Gloomhaven Items.pdf" \
    --output-dir out/items \
    --dpi 200 --min-area 30000 --debug

You can feed multiple PDFs or use a glob pattern with repeated --input.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

import fitz  # PyMuPDF
import numpy as np
import cv2


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    def to_rect(self) -> fitz.Rect:
        return fitz.Rect(self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 0.0


def parse_pages_arg(pages: Optional[str], page_count: int) -> List[int]:
    if not pages:
        return list(range(page_count))
    result: List[int] = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            start = int(a) - 1
            end = int(b) - 1
            for i in range(start, end + 1):
                if 0 <= i < page_count:
                    result.append(i)
        else:
            i = int(part) - 1
            if 0 <= i < page_count:
                result.append(i)
    # Remove duplicates while preserving order
    seen = set()
    ordered = []
    for i in result:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def render_page_preview(page: fitz.Page, dpi: int) -> Tuple[np.ndarray, float]:
    """Render a page to a BGR image for detection; return image and zoom.

    zoom = dpi/72 since PDF user space is 72 dpi.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img, zoom


def detect_card_bboxes(
    img_bgr: np.ndarray,
    min_area: int = 20000,
    max_area: Optional[int] = None,
    aspect_tolerance: float = 0.25,
    morph_kernel: int = 5,
    canny1: int = 50,
    canny2: int = 150,
    debug: bool = False,
) -> List[BBox]:
    """Detect card bounding boxes in pixels on the rendered image.

    Heuristic approach:
      - Convert to grayscale and blur.
      - Canny edges + close gaps with morphology.
      - Find contours and filter by area and approximate rectangularity.
      - Cluster by similar aspect to stabilize against spurious boxes.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny1, canny2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        # Reject too thin shapes
        asp = (w / h) if h else 0
        if asp < 0.5 or asp > 2.2:
            # Items are roughly portrait; adjust if landscape needed.
            pass  # keep loose; aspect clustering will prune later

        # Check rectangularity by contour area vs bbox area
        rect_area = area
        cnt_area = cv2.contourArea(cnt)
        if rect_area > 0 and (cnt_area / rect_area) < 0.6:
            # Too concave/noisy for a card candidate
            continue

        candidates.append(BBox(x, y, w, h))

    if not candidates:
        return []

    # Cluster by aspect ratio to keep the dominant card shape
    aspects = np.array([c.aspect for c in candidates])
    # Bin aspects and pick the bin with most members
    bins = np.linspace(max(0.2, aspects.min()), aspects.max() + 1e-6, num=12)
    hist, edges_bins = np.histogram(aspects, bins=bins)
    best_bin_idx = int(np.argmax(hist))
    lo = edges_bins[best_bin_idx]
    hi = edges_bins[best_bin_idx + 1]

    filtered = [c for c in candidates if lo - 1e-6 <= c.aspect <= hi + 1e-6]

    # Optionally, refine by size clustering (cards same size)
    areas = np.array([c.area for c in filtered])
    if len(areas) >= 3:
        med = np.median(areas)
        keep = []
        for c in filtered:
            if abs(c.area - med) / med <= 0.35:  # within 35% of median
                keep.append(c)
        filtered = keep if keep else filtered

    # Merge near-duplicates (overlapping boxes)
    filtered = suppress_overlaps(filtered, iou_threshold=0.5)

    if debug:
        debug_img = img_bgr.copy()
        for b in filtered:
            cv2.rectangle(debug_img, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (0, 255, 0), 3)
        cv2.imshow('detected', debug_img)
        cv2.waitKey(1)

    return filtered


def suppress_overlaps(boxes: List[BBox], iou_threshold: float = 0.5) -> List[BBox]:
    if not boxes:
        return []
    # Non-maximum suppression by area (keep larger boxes)
    boxes_sorted = sorted(boxes, key=lambda b: b.area, reverse=True)
    kept: List[BBox] = []
    for b in boxes_sorted:
        if all(iou(b, k) < iou_threshold for k in kept):
            kept.append(b)
    return kept


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def sort_boxes_reading_order(boxes: List[BBox], y_tol: float = 20.0) -> List[BBox]:
    # Group by rows (similar y), then sort each row by x
    rows: List[List[BBox]] = []
    for b in sorted(boxes, key=lambda b: (b.y, b.x)):
        placed = False
        for row in rows:
            if abs(row[0].y - b.y) <= y_tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    # Sort boxes in each row by x
    for row in rows:
        row.sort(key=lambda b: b.x)
    # Sort rows by y
    rows.sort(key=lambda row: row[0].y)
    return [b for row in rows for b in row]


def export_card_pdf(
    src_doc: fitz.Document,
    pno: int,
    clip_rect_pdf: fitz.Rect,
    out_path: str,
):
    # Create a new one-page PDF with the same size as the clip rect
    out_doc = fitz.open()
    page = out_doc.new_page(width=clip_rect_pdf.width, height=clip_rect_pdf.height)
    # Place the source page into the new page, clipped to the rect
    page.show_pdf_page(
        fitz.Rect(0, 0, clip_rect_pdf.width, clip_rect_pdf.height),
        src_doc,
        pno,
        clip=clip_rect_pdf,
    )
    out_doc.save(out_path)
    out_doc.close()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_pdf(
    input_path: str,
    output_dir: str,
    dpi: int,
    min_area: int,
    max_area: Optional[int],
    margin: int,
    pages: Optional[str],
    debug: bool,
    enforce_uniform_size: bool = False,
    size_tol: float = 0.2,
    aspect_tol: float = 0.2,
    max_per_page: Optional[int] = None,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    doc = fitz.open(input_path)
    page_indices = parse_pages_arg(pages, len(doc))

    ensure_dir(output_dir)

    # If enforcing uniform size, learn reference size from the first page with detections
    ref_area: Optional[float] = None
    ref_aspect: Optional[float] = None

    for pno in page_indices:
        page = doc[pno]
        img_bgr, zoom = render_page_preview(page, dpi=dpi)

        boxes_px = detect_card_bboxes(
            img_bgr,
            min_area=min_area,
            max_area=max_area,
            debug=debug,
        )

        if not boxes_px:
            print(f"[warn] No cards detected on page {pno+1} of {input_path}")
            continue

        # Optionally enforce uniform size across pages
        if enforce_uniform_size:
            areas = np.array([b.area for b in boxes_px], dtype=float)
            aspects = np.array([b.aspect for b in boxes_px], dtype=float)
            if ref_area is None or ref_aspect is None:
                # Learn from this page: use medians as robust references
                ref_area = float(np.median(areas))
                ref_aspect = float(np.median(aspects))
            # Filter to those within tolerance of learned reference
            kept: List[BBox] = []
            for b in boxes_px:
                if ref_area and abs(b.area - ref_area) / ref_area <= size_tol:
                    if ref_aspect and abs(b.aspect - ref_aspect) <= aspect_tol:
                        kept.append(b)
            if kept:
                boxes_px = kept
            # If nothing within tolerance, fall back to original detections for this page

        boxes_px = sort_boxes_reading_order(boxes_px)

        if max_per_page is not None and len(boxes_px) > max_per_page:
            boxes_px = boxes_px[:max_per_page]

        # Convert pixel rectangles to PDF user-space rects (divide by zoom)
        for idx, b in enumerate(boxes_px, start=1):
            # Apply margin in pixels, then convert
            x = max(0, b.x - margin)
            y = max(0, b.y - margin)
            w = min(img_bgr.shape[1] - x, b.w + 2 * margin)
            h = min(img_bgr.shape[0] - y, b.h + 2 * margin)
            clip_pdf = fitz.Rect(x / zoom, y / zoom, (x + w) / zoom, (y + h) / zoom)

            out_path = os.path.join(
                output_dir, f"{base_name}_p{pno+1:03d}_i{idx:02d}.pdf"
            )
            export_card_pdf(doc, pno, clip_pdf, out_path)
            print(f"[ok] Wrote {out_path}")

    doc.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Split item-card PDFs into individual card PDFs")
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input PDF path. Repeat for multiple PDFs.",
    )
    ap.add_argument(
        "--output-dir",
        default="out/cards",
        help="Directory to write card PDFs",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Preview DPI for detection (not export)")
    ap.add_argument("--min-area", type=int, default=20000, help="Min contour bbox area in pixels")
    ap.add_argument("--max-area", type=int, default=None, help="Max contour bbox area in pixels")
    ap.add_argument("--margin", type=int, default=8, help="Margin (pixels at preview DPI) around detected box")
    ap.add_argument("--pages", default=None, help="Pages to process, e.g. '1-3,5,8'")
    ap.add_argument("--debug", action="store_true", help="Show debug windows with detections")
    ap.add_argument("--enforce-uniform-size", action="store_true", help="Keep only boxes matching the card size learned from the first detected page")
    ap.add_argument("--size-tol", type=float, default=0.2, help="Relative area tolerance when enforcing uniform size (e.g., 0.2 = ±20%)")
    ap.add_argument("--aspect-tol", type=float, default=0.2, help="Absolute aspect ratio tolerance when enforcing uniform size")
    ap.add_argument("--max-per-page", type=int, default=None, help="Limit the maximum number of cards exported per page")
    args = ap.parse_args(argv)

    ensure_dir(args.output_dir)

    for ip in args.input:
        process_pdf(
            input_path=ip,
            output_dir=args.output_dir,
            dpi=args.dpi,
            min_area=args.min_area,
            max_area=args.max_area,
            margin=args.margin,
            pages=args.pages,
            debug=args.debug,
            enforce_uniform_size=args.enforce_uniform_size,
            size_tol=args.size_tol,
            aspect_tol=args.aspect_tol,
            max_per_page=args.max_per_page,
        )

    if args.debug:
        print("Press any key in the debug window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
