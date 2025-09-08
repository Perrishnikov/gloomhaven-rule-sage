#!/usr/bin/env python3
"""
Extract text-only rules from vector PDFs, skipping images, and output JSONL.

For each PDF page:
- Reads layout via PyMuPDF (import name: pymupdf) if installed.
- Skips image blocks and text near images.
- Drops likely headers/footers (top/bottom margin heuristic).
- Merges lines to paragraphs; fixes common hyphenation.
- Infers headings by relative font size and chunks text under headings.

Output: JSON Lines to stdout or a file with fields:
  {
    "source": "web/files/p3.pdf",
    "page": 3,
    "heading": "Combat",
    "anchor": "combat",
    "text": "...paragraph text..."
  }

Usage:
  # Default I/O (no args):
  python tools/extract_rules/extract_rules.py
  
  # Custom paths:
  python tools/extract_rules/extract_rules.py --input web/files --output tools/extract_rules/out/rules_index.jsonl

Notes:
- Requires PyMuPDF (pip install pymupdf).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import re


def eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def normalize_text(s: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Span:
    text: str
    size: float


@dataclass
class Line:
    bbox: Tuple[float, float, float, float]
    spans: List[Span]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans).strip()

    @property
    def font_size(self) -> float:
        if not self.spans:
            return 0.0
        # Use max size in line as proxy for visual prominence
        return max(s.size for s in self.spans)


@dataclass
class Paragraph:
    lines: List[Line]

    @property
    def text(self) -> str:
        # Merge lines with space/newline as appropriate
        parts: List[str] = []
        prev: Optional[str] = None
        for ln in self.lines:
            t = ln.text
            if not t:
                continue
            if prev and prev.endswith("-") and t and t[:1].islower():
                # de-hyphenate
                parts[-1] = prev[:-1] + t
            else:
                if parts:
                    parts.append(" \n" + t if t[:1].isupper() and not prev.endswith(".") else " " + t)
                else:
                    parts.append(t)
            prev = parts[-1]
        return "".join(parts).strip()


def overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def expand(b: Tuple[float, float, float, float], pad: float) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = b
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def collect_pdfs(root: str) -> List[str]:
    pdfs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip any folder named 'ignored'
        if os.path.basename(dirpath).lower() == "ignored":
            continue
        for f in filenames:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, f))
    pdfs.sort()
    return pdfs


def infer_heading_threshold(lines: List[Line]) -> float:
    sizes = [ln.font_size for ln in lines if ln.text]
    if not sizes:
        return 0.0
    sizes.sort()
    # Median and high percentile to separate headings
    mid = sizes[len(sizes)//2]
    p80 = sizes[int(len(sizes) * 0.8)]
    # Headings typically clearly larger than body; use a blend
    return max(mid * 1.25, p80)


def is_header_footer(line_bbox: Tuple[float, float, float, float], page_height: float, margin_ratio: float = 0.08) -> bool:
    _, y0, _, y1 = line_bbox
    top_cut = page_height * margin_ratio
    bot_cut = page_height * (1.0 - margin_ratio)
    # line top or bottom within margin band
    return y0 < top_cut or y1 > bot_cut


def extract_pdf_pymupdf(path: str) -> Iterable[dict]:
    # Prefer modern import name `pymupdf`, fall back to legacy `fitz` for compatibility
    try:
        import pymupdf as pdf
    except ImportError:
        try:
            import fitz as pdf  # type: ignore
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")

    doc = pdf.open(path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        pwidth, pheight = page.rect.width, page.rect.height
        d = page.get_text("dict")
        blocks = d.get("blocks", [])
        image_boxes: List[Tuple[float, float, float, float]] = [tuple(b["bbox"]) for b in blocks if b.get("type") == 1]
        image_boxes_expanded = [expand(b, pad=10.0) for b in image_boxes]

        text_lines: List[Line] = []
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                bbox = tuple(line.get("bbox"))  # type: ignore
                spans: List[Span] = []
                for sp in line.get("spans", []):
                    stext = sp.get("text", "")
                    if not stext.strip():
                        continue
                    size = float(sp.get("size", 0.0))
                    spans.append(Span(text=stext, size=size))
                if not spans:
                    continue
                # Skip headers/footers by position
                if is_header_footer(bbox, pheight):
                    continue
                # Drop lines that overlap image regions (with padding)
                if any(overlap(bbox, ib) for ib in image_boxes_expanded):
                    continue
                text_lines.append(Line(bbox=bbox, spans=spans))

        # Sort lines by y, then x
        text_lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))

        # Paragraph grouping: join lines that are vertically close and similarly indented
        paragraphs: List[Paragraph] = []
        cur: List[Line] = []
        prev_bottom: Optional[float] = None
        prev_x0: Optional[float] = None
        for ln in text_lines:
            if not cur:
                cur.append(ln)
                prev_bottom = ln.bbox[3]
                prev_x0 = ln.bbox[0]
                continue
            # heuristics
            v_gap = ln.bbox[1] - (prev_bottom or ln.bbox[1])
            same_indent = abs((prev_x0 or ln.bbox[0]) - ln.bbox[0]) <= 8.0
            if v_gap <= 6.0 and same_indent:
                cur.append(ln)
            else:
                paragraphs.append(Paragraph(lines=cur))
                cur = [ln]
            prev_bottom = ln.bbox[3]
            prev_x0 = ln.bbox[0]
        if cur:
            paragraphs.append(Paragraph(lines=cur))

        # Heading detection
        heading_threshold = infer_heading_threshold(text_lines)

        current_heading: Optional[str] = None
        for para in paragraphs:
            if not para.lines:
                continue
            # Treat single-line paragraphs with large font as headings
            if len(para.lines) <= 2 and max(ln.font_size for ln in para.lines) >= heading_threshold:
                # Set as current heading
                ht = para.text.strip().strip(":")
                if ht:
                    current_heading = ht
                continue

            txt = normalize_text(para.text.strip())
            if not txt:
                continue
            yield {
                "source": path,
                "page": page_index + 1,
                "heading": current_heading,
                "anchor": slugify(current_heading) if current_heading else None,
                "text": txt,
            }


def extract_pdf(path: str) -> Iterable[dict]:
    """Extract using PyMuPDF only (assumes valid vector PDFs)."""
    yield from extract_pdf_pymupdf(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract rules text from PDFs into JSONL")
    ap.add_argument(
        "--input",
        default="web/files",
        help="Input file or directory of PDFs (default: web/files)",
    )
    ap.add_argument(
        "--output",
        default="tools/extract_rules/out/rules_index.jsonl",
        help="Output JSONL file (default: tools/extract_rules/out/rules_index.jsonl)",
    )
    args = ap.parse_args()

    in_path = args.input
    if os.path.isdir(in_path):
        pdfs = collect_pdfs(in_path)
    else:
        pdfs = [in_path]

    # Prepare output stream
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_stream = open(args.output, "w", encoding="utf-8")
    wrote = 0
    try:
        for pdf in pdfs:
            for row in extract_pdf(pdf):
                out_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()

    eprint(f"Wrote {wrote} JSONL rows from {len(pdfs)} PDF(s)")


if __name__ == "__main__":
    main()
