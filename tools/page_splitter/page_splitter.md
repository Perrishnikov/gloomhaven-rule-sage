Page Splitter (Lossless)

What it does
- Splits each PDF in an input directory into single-page PDFs using pikepdf (qpdf backend).
- Writes outputs to `tools/page_splitter/out` by default.
- Keeps integrity: copies page objects/fonts/images without re-rendering.

Install
- `pip install pikepdf`
- Optional verification: `pip install pymupdf`

Usage
- Single file → out/<stem>/p{n}.pdf:
  - `python tools/page_splitter/split_pages.py --input web/files/ignored/RuleBook.pdf`
- Directory → each PDF to its own subfolder in out/:
  - `python tools/page_splitter/split_pages.py --input web/files/ignored`
- Custom output directory:
  - `python tools/page_splitter/split_pages.py --input web/files --out tools/page_splitter/out`
- Flat output (no per-file subfolders):
  - `python tools/page_splitter/split_pages.py --input web/files/ignored --flat`
- Overwrite existing outputs:
  - `python tools/page_splitter/split_pages.py --input web/files/ignored --overwrite`
- Verify each output opens with PyMuPDF:
  - `python tools/page_splitter/split_pages.py --input web/files/ignored --verify`

Output structure
- By default: `tools/page_splitter/out/<source-stem>/p{n}.pdf`
  - Example: `Rule-Book_2P-V9/p1.pdf`, `Rule-Book_2P-V9/p2.pdf`, ...
- With `--flat`: `tools/page_splitter/out/p{n}.pdf` (not recommended if multiple sources are present)

Notes
- The tool is non-recursive; point `--input` at the folder containing the consolidated PDFs you want to split.
- Page size differences are preserved; this does not affect correctness.
