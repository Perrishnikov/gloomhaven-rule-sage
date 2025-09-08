PDF Rules Extraction

Overview
- Extracts text-only content from vector PDFs (no OCR needed), skipping images and likely captions.
- Outputs JSONL rows for RAG indexing with fields: `source`, `page`, `heading`, `anchor`, `text`.

Requirements
- Python 3.9+
- PyMuPDF: `pip install pymupdf` (import as `pymupdf`; legacy alias `fitz` also works)

Run
- Default (no args):
  - `python tools/extract_rules/extract_rules.py`
  - Writes to `tools/extract_rules/out/rules_index.jsonl`
- Custom paths:
  - `python tools/extract_rules/extract_rules.py --input web/files --output tools/extract_rules/out/rules_index.jsonl`

Robustness
- Assumes valid, vector PDFs with a text layer. If you encounter parse errors, repair the source PDFs (e.g., with `pikepdf` or `qpdf`) before re-running.

Output normalization
- The extractor collapses all internal line breaks and repeated whitespace to single spaces in the `text` field for more consistent downstream handling.
- The script recursively reads PDFs under the input path, ignoring any `ignored/` subfolders.
- It will create the `data/` folder if it doesn’t exist.

Heuristics
- Skips image blocks and nearby text (10px padding) to avoid captions.
- Drops headers/footers using a simple top/bottom margin (8% of page height).
- Merges lines into paragraphs and fixes common hyphenation.
- Infers headings by relative font size and assigns paragraphs to the most recent heading.

Output Schema (JSONL)
- `source` (str): PDF path relative to repo (e.g., `web/files/p3.pdf`).
- `page` (int): 1-based page number.
- `heading` (str|null): Inferred section heading, if any.
- `anchor` (str|null): Slugified `heading` (e.g., `area-late-join`), if any.
- `text` (str): Paragraph text.

Notes
- If headings aren’t detected well (unusual fonts), adjust the threshold logic in `tools/extract_rules/extract_rules.py` (`infer_heading_threshold`).
- For PDFs that do contain scanned pages, run OCR first (e.g., `ocrmypdf`) to add a text layer, then run this extractor unchanged.
