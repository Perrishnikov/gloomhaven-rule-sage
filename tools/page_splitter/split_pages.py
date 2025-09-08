#!/usr/bin/env python3
"""
Lossless page splitter for PDFs.

Given an input file or directory, splits PDF(s) into single-page PDFs using
pikepdf without re-rendering. Outputs to tools/page_splitter/out by default,
under a subfolder named after the source file stem (to avoid collisions).

Usage:
  # Single file → out/<stem>/p{n}.pdf
  python tools/page_splitter/split_pages.py --input web/files/ignored/RuleBook.pdf

  # Directory of PDFs → each to its own subfolder in out/
  python tools/page_splitter/split_pages.py --input web/files/ignored

  # Custom output directory
  python tools/page_splitter/split_pages.py --input web/files --out tools/page_splitter/out

Notes:
- Install dependencies:
    pip install pikepdf
- Optional verification that each output opens cleanly:
    pip install pymupdf
    add --verify
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sanitize_stem(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    # Simple, file-system safe slug
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '-' for ch in base)
    while '--' in safe:
        safe = safe.replace('--', '-')
    return safe.strip('-') or 'pdf'


def verify_open(path: str) -> bool:
    try:
        import pymupdf as pdf  # type: ignore
    except Exception:
        try:
            import fitz as pdf  # type: ignore
        except Exception:
            pdf = None  # type: ignore
    if pdf is None:
        return True
    try:
        doc = pdf.open(path)
        _ = len(doc)
        doc.close()
        return True
    except Exception:
        return False


def split_pdf(src_path: str, out_root: str, flat: bool = False, overwrite: bool = False, verify: bool = False) -> tuple[int, int]:
    try:
        import pikepdf  # type: ignore
    except ImportError:
        print("pikepdf not installed. Install with: pip install pikepdf", file=sys.stderr)
        sys.exit(2)

    stem = sanitize_stem(src_path)
    out_dir = out_root if flat else os.path.join(out_root, stem)
    ensure_dir(out_dir)

    written = 0
    failed = 0
    try:
        with pikepdf.open(src_path) as src:
            total = len(src.pages)
            for i in range(total):
                page_no = i + 1
                out_path = os.path.join(out_dir, f"p{page_no}.pdf")
                if os.path.exists(out_path) and not overwrite:
                    # Skip existing
                    continue
                dst = pikepdf.Pdf.new()
                dst.pages.append(src.pages[i])
                # Linearize for better random access in viewers
                dst.save(out_path, linearize=True)
                ok = verify_open(out_path) if verify else True
                if ok:
                    written += 1
                else:
                    failed += 1
                    # Remove bad output to avoid confusion
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
        return written, failed
    except Exception as ex:
        print(f"Failed to split {src_path}: {ex}", file=sys.stderr)
        return written, failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Lossless PDF page splitter")
    ap.add_argument("--input", required=True, help="Input PDF file or directory containing PDFs to split")
    ap.add_argument("--out", default="tools/page_splitter/out", help="Output directory (default: tools/page_splitter/out)")
    ap.add_argument("--flat", action="store_true", help="Do not create per-file subfolders; write p{n}.pdf in out")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present")
    ap.add_argument("--verify", action="store_true", help="Verify each output opens with PyMuPDF")
    args = ap.parse_args()

    in_path = args.input
    out_dir = args.out
    ensure_dir(out_dir)

    # Discover PDFs
    pdfs = []
    if os.path.isdir(in_path):
        # Non-recursive: split all PDFs in the directory
        pdfs = [os.path.join(in_path, f) for f in os.listdir(in_path) if f.lower().endswith('.pdf')]
        if not pdfs:
            print(f"No PDFs found in directory: {in_path}", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.path.exists(in_path):
            print(f"Input not found: {in_path}", file=sys.stderr)
            sys.exit(1)
        if not in_path.lower().endswith('.pdf'):
            print(f"Input is not a PDF: {in_path}", file=sys.stderr)
            sys.exit(1)
        pdfs = [in_path]

    total_written = 0
    total_failed = 0
    for pdf_path in sorted(pdfs):
        w, f = split_pdf(pdf_path, out_dir, flat=args.flat, overwrite=args.overwrite, verify=args.verify)
        rel = os.path.basename(pdf_path)
        print(f"Split {rel}: wrote {w} page(s){' with '+str(f)+' failures' if f else ''}")
        total_written += w
        total_failed += f

    print(f"Done. Total pages written: {total_written}; verification failures: {total_failed}")


if __name__ == "__main__":
    main()
