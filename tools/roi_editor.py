#!/usr/bin/env python3
"""
Interactive ROI editor for item cards.

Usage:
  python tools/roi_editor.py \
    --input "out/items/Gloomhaven Items_p001_i01.pdf" \
    --output "configs/items_template.json" \
    --dpi 400

Controls:
  - Click & drag: draw/set rectangle for the active field
  - n / p: next / previous field
  - r: clear current field rectangle
  - s: save template JSON (normalized coordinates 0..1)
  - q or ESC: quit

Fields are predefined but you can override via --fields.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import cv2


DEFAULT_FIELDS = [
    "title",
    "cost",
    "usage",
    "slot",
    "number",
    "source",
    "count",
]


def render_pdf_page(path: str, dpi: int, page_index: int = 0) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    doc = fitz.open(path)
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    page_size = (page.rect.width, page.rect.height)
    doc.close()
    return img, zoom, page_size


def normalize_rect(rect_xywh: Tuple[int, int, int, int], img_w: int, img_h: int) -> List[float]:
    x, y, w, h = rect_xywh
    return [x / img_w, y / img_h, w / img_w, h / img_h]


def denormalize_rect(norm: List[float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = norm
    return int(x * img_w), int(y * img_h), int(w * img_w), int(h * img_h)


def overlay_text(img: np.ndarray, lines: List[str], x: int = 10, y: int = 20):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="Edit ROIs for card fields")
    ap.add_argument("--input", required=True, help="Sample card PDF path")
    ap.add_argument("--output", required=True, help="Output template JSON path")
    ap.add_argument("--dpi", type=int, default=400, help="Render DPI for editing/preview")
    ap.add_argument("--fields", nargs="*", default=DEFAULT_FIELDS, help="Field names in order")
    args = ap.parse_args()

    img, zoom, page_size = render_pdf_page(args.input, dpi=args.dpi)
    ih, iw = img.shape[:2]

    # Store rectangles as normalized coords per field
    rects: Dict[str, Optional[Tuple[int, int, int, int]]] = {f: None for f in args.fields}
    current_idx = 0
    drawing = False
    start_pt: Optional[Tuple[int, int]] = None

    win = "ROI Editor"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal drawing, start_pt, rects
        field = args.fields[current_idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing and start_pt is not None:
            sx, sy = start_pt
            rects[field] = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
        elif event == cv2.EVENT_LBUTTONUP and start_pt is not None:
            drawing = False
            sx, sy = start_pt
            rects[field] = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
            start_pt = None

    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = img.copy()
        # Draw existing rects
        for i, f in enumerate(args.fields):
            r = rects[f]
            color = (0, 200, 0) if i == current_idx else (255, 0, 0)
            if r is not None:
                x, y, w, h = r
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis, f, (x, max(10, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        # Overlay help text
        overlay_text(
            vis,
            [
                f"Field [{current_idx+1}/{len(args.fields)}]: {args.fields[current_idx]}",
                "Mouse: draw rect | n/p: next/prev | r: reset",
                "s: save | q/ESC: quit",
            ],
        )
        cv2.imshow(win, vis)
        key = cv2.waitKey(16) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(args.fields)
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(args.fields)
        elif key == ord('r'):
            rects[args.fields[current_idx]] = None
        elif key == ord('s'):
            # Build template JSON
            fields_out = {}
            for f, r in rects.items():
                if r is None:
                    continue
                fields_out[f] = normalize_rect(r, iw, ih)
            tpl = {
                "dpi": args.dpi,
                "page_width": page_size[0],
                "page_height": page_size[1],
                "fields": fields_out,
            }
            with open(args.output, 'w') as f:
                json.dump(tpl, f, indent=2)
            print(f"[ok] Saved template to {args.output}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

