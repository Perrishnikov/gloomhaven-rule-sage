#!/usr/bin/env python3
"""
OCR item cards using a ROI template.

Template JSON (example):
{
  "dpi": 400,
  "fields": {
    "title": [x, y, w, h],
    "cost": [x, y, w, h]
  }
}

Coordinates are normalized (0..1) relative to the full card page size.

Usage:
  python tools/ocr_cards.py \
    --input-dir out/items \
    --template configs/items_template.json \
    --output out/items.json \
    --key-field number \
    --dpi 400
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
import re
import unicodedata

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import pytesseract
import cv2


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_inputs(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if ext == ".pdf" or ext in IMAGE_EXTS:
            files.append(p)
    return files


def load_template(path: str) -> Dict:
    with open(path, 'r') as f:
        tpl = json.load(f)
    if 'fields' not in tpl:
        raise ValueError('Template missing fields')
    return tpl


def norm_to_pdf_rect(page: fitz.Page, norm_xywh: List[float]) -> fitz.Rect:
    x, y, w, h = norm_xywh
    W, H = page.rect.width, page.rect.height
    return fitz.Rect(x * W, y * H, (x + w) * W, (y + h) * H)


def get_text_from_pdf(page: fitz.Page, rect: fitz.Rect) -> str:
    text = page.get_textbox(rect) or ""
    return text.strip()


def _illumination_correct(gray: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Remove slow gradients / parchment texture by dividing by a blurred bg."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def _unsharp_mask(gray: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
    if radius <= 0 or amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=radius, sigmaY=radius)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _preprocess_image(
    img: Image.Image,
    *,
    prefer_bright_text: bool = True,
    mode: str = "simple",
    block_size: int = 35,
    c_const: int = 10,
    thicken: bool = False,
) -> Image.Image:
    """Preprocess ROI for OCR.

    modes:
      - simple: Gaussian blur + Otsu + light morphology (robust baseline)
      - bright: assumes bright text; tophat + adaptive threshold
      - strong: illumination correction + CLAHE + bilateral + adaptive
    """
    if img.mode != 'L':
        img = img.convert('L')
    gray = np.array(img)

    if mode == "simple":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Ensure black text on white
        if prefer_bright_text or np.mean(th) > 127:
            th = 255 - th
        k = np.ones((2, 2), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
        if thicken:
            th = cv2.dilate(th, k, iterations=1)
        return Image.fromarray(th)

    # Non-simple pipelines: remove gradients first
    gray = _illumination_correct(gray, sigma=15.0)

    if mode == "bright":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        src = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        src = cv2.medianBlur(src, 3)
        if block_size % 2 == 0:
            block_size += 1
        th = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c_const)
        if prefer_bright_text or np.mean(th) > 127:
            th = 255 - th
        k = np.ones((2, 2), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
        if thicken:
            th = cv2.dilate(th, k, iterations=1)
        return Image.fromarray(th)

    # strong
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    src = clahe.apply(gray)
    src = cv2.bilateralFilter(src, d=5, sigmaColor=40, sigmaSpace=40)
    if block_size % 2 == 0:
        block_size += 1
    th = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block_size, c_const)
    if prefer_bright_text or np.mean(th) > 127:
        th = 255 - th
    k = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    if thicken:
        th = cv2.dilate(th, k, iterations=1)
    return Image.fromarray(th)


def _render_roi(page: fitz.Page, rect: fitz.Rect, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, clip=rect)
    mode = "RGB" if pix.n == 3 else "L"
    return Image.frombytes(mode, (pix.w, pix.h), pix.samples)


def _autocrop_binary(img: Image.Image, *, pad_frac: float = 0.06, min_keep_frac: float = 0.12) -> Image.Image:
    """Crop to content on a binarized image (black text on white).

    Finds the bounding box of dark pixels and expands by pad_frac. If the
    box would be too small (less than min_keep_frac of area), returns the
    original image to avoid over-cropping.
    """
    arr = np.array(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = (arr < 200).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return img
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = arr.shape[:2]
    pad = int(round(min(w, h) * pad_frac))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    box_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    if box_area < min_keep_frac * (w * h):
        return img
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def ocr_rect(page: fitz.Page, rect: fitz.Rect, dpi: int, *, field: str = "", preproc_args: Optional[dict] = None, autocrop: bool = True, pad_frac: float = 0.06) -> str:
    img = _render_roi(page, rect, dpi)
    # Most fields are white text; Number is black text on white
    prefer_bright = (field.lower() not in ("slot", "usage", "number"))
    preproc_args = preproc_args or {}
    proc = _preprocess_image(img, prefer_bright_text=prefer_bright, **preproc_args)
    if autocrop:
        proc = _autocrop_binary(proc, pad_frac=pad_frac)
    # Per-field configs
    field = (field or "").lower()
    psm = {
        'title': 7,
        'number': 7,
        'cost': 7,
        'count': 7,
        'source': 6,
        'usage': 7,
    }.get(field, 6)
    whitelist = None
    if field in ("number", "cost", "count"):
        whitelist = "0123456789"
    # Build tesseract config
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    text = pytesseract.image_to_string(proc, config=cfg)
    return text.strip()


def _norm_basic(value: str) -> str:
    v = unicodedata.normalize('NFKC', value)
    # Common OCR artifacts and punctuation normalization
    v = v.replace('–', '-').replace('—', '-').replace('−', '-')
    v = v.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    v = v.replace('•', '-').replace('·', '-')
    v = v.replace('¥', '+')  # OCR often maps '+' as Yen
    v = v.replace('™', '').replace('©', '')
    # Collapse whitespace
    v = re.sub(r"\s+", " ", v)
    return v.strip()


def normalize_slot(value: str) -> str:
    s = value.strip().lower()
    mapping = {
        'head': 'Head',
        'body': 'Body',
        'legs': 'Legs',
        'one hand': 'One Hand',
        'one-hand': 'One Hand',
        'two hands': 'Two Hands',
        'two-hands': 'Two Hands',
        'small item': 'Small Item',
        'small-item': 'Small Item',
    }
    # exact match first
    if s in mapping:
        return mapping[s]
    # fuzzy contains
    for k, v in mapping.items():
        if k in s:
            return v
    return value.strip()


def postprocess_field(name: str, value: str) -> str:
    v = _norm_basic(value)
    if name.lower() == 'slot':
        return normalize_slot(v)
    if name.lower() in ('number', 'cost'):
        # Prefer digits; if none, map common OCR confusions and retry
        m = re.search(r"\d{1,4}", v)
        if not m:
            subst = (
                v.replace('O', '0').replace('o', '0')
                 .replace('S', '5').replace('s', '5')
                 .replace('I', '1').replace('l', '1')
                 .replace('B', '8').replace('Z', '2')
                 .replace('g', '9')
            )
            m = re.search(r"\d{1,4}", subst)
        return m.group(0) if m else ''
    if name.lower() == 'count':
        # Normalize common OCR confusions and enforce x/y pattern
        s = (
            v.replace('O', '0').replace('o', '0')
             .replace('S', '5').replace('s', '5')
             .replace('I', '1').replace('l', '1')
             .replace('B', '8').replace('Z', '2')
             .replace('g', '9')
        )
        # Unify separators likely misread as '/'
        s = s.replace('\\', '/').replace('|', '/').replace('‖', '/').replace('÷', '/')
        s = re.sub(r"\s*/\s*", "/", s)
        m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", s)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        nums = re.findall(r"\d{1,2}", s)
        if len(nums) >= 2:
            return f"{nums[0]}/{nums[1]}"
        # Heuristic: a single two-digit token like '12' may mean '1/2'
        if len(nums) == 1 and len(nums[0]) == 2:
            return f"{nums[0][0]}/{nums[0][1]}"
        return ''
    if name.lower() == 'title':
        # Strip decorative leading/trailing punctuation and underscores
        v = re.sub(r"^[\-_.:;|\s]+", "", v)
        v = re.sub(r"[|:;.,\-\s]+$", "", v)
        # Remove leading/trailing quotes
        v = v.strip('"\'')
        # Insert space before capital following lowercase (e.g., WingedShoes -> Winged Shoes)
        v = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v
    return v


class IconClassifier:
    """Icon classifier for Slot/Usage.

    - Loads templates from `icons_dir/slot/*.png` and `icons_dir/usage/*.png`.
    - Matches with two methods:
        1) Hu moments distance on binarized silhouettes (scale/rotation robust)
        2) Template matching (TM_CCOEFF_NORMED) after resizing ROI to template
    - Method is configurable; default tries both and picks the best.
    """

    def __init__(self, icons_dir: Optional[str], method: str = "both"):
        self.icons_dir = icons_dir
        self.method = method  # 'hu' | 'tm' | 'both'
        # Store both masks and Hu vectors
        self.tpl_masks: Dict[str, Dict[str, np.ndarray]] = {"slot": {}, "usage": {}}
        self.tpl_hu: Dict[str, Dict[str, np.ndarray]] = {"slot": {}, "usage": {}}
        if icons_dir and os.path.isdir(icons_dir):
            self._load_group("slot")
            self._load_group("usage")

    def _load_group(self, group: str):
        folder = os.path.join(self.icons_dir, group)
        if not os.path.isdir(folder):
            return
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            label = os.path.splitext(fn)[0]
            path = os.path.join(folder, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            mask = self._binarize(img)
            self.tpl_masks[group][label] = mask
            self.tpl_hu[group][label] = self._hu(mask)

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        gray = cv2.medianBlur(gray, 3)
        # Normalize lighting, then Otsu
        bg = cv2.GaussianBlur(gray, (0, 0), 3)
        norm = cv2.divide(gray, bg, scale=255)
        _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if mostly white background
        if np.mean(bw) > 127:
            bw = 255 - bw
        kernel = np.ones((3, 3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        return bw

    @staticmethod
    def _hu(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((7,), dtype=np.float32)
        cnt = max(contours, key=cv2.contourArea)
        m = cv2.moments(cnt)
        hu = cv2.HuMoments(m).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        return hu.astype(np.float32)

    @staticmethod
    def _tm_score(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
        # Resize ROI mask to template size and compute normalized correlation
        h, w = b_mask.shape[:2]
        a_res = cv2.resize(a_mask, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(a_res, b_mask, cv2.TM_CCOEFF_NORMED)
        return float(res.max())

    def classify(self, group: str, roi_img: Image.Image) -> Optional[str]:
        if group not in ("slot", "usage"):
            return None
        if not self.tpl_masks[group]:
            return None
        gray = np.array(roi_img.convert('L'))
        mask = self._binarize(gray)
        best_label = None
        best_score = -1.0
        if self.method in ("hu", "both"):
            roi_hu = self._hu(mask)
            for label, tpl_hu in self.tpl_hu[group].items():
                d = float(np.linalg.norm(roi_hu - tpl_hu))
                score = -d  # lower distance is better
                if score > best_score:
                    best_score, best_label = score, label
        if self.method in ("tm", "both"):
            for label, tpl_mask in self.tpl_masks[group].items():
                s = self._tm_score(mask, tpl_mask)
                # scale TM scores (0..1) above typical HU negative scores
                score = s * 10.0
                if score > best_score:
                    best_score, best_label = score, label
        return best_label


def process_card_pdf(pdf_path: str, fields_norm: Dict[str, List[float]], dpi: int, prefer_pdf_text: bool = True, icon_cls: Optional[IconClassifier] = None, debug_dir: Optional[str] = None, preproc_args: Optional[dict] = None, autocrop: bool = True, pad_frac: float = 0.06, icons_only: bool = False) -> Dict[str, str]:
    doc = fitz.open(pdf_path)
    page = doc[0]
    out: Dict[str, str] = {}
    for fname, nrect in fields_norm.items():
        rect = norm_to_pdf_rect(page, nrect)
        # Icon classification for slot/usage if templates provided
        fkey = (fname or "").lower()
        if icon_cls is not None and fkey in ("slot", "usage"):
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False, clip=rect)
            mode = "RGB" if pix.n == 3 else "L"
            roi_img = Image.frombytes(mode, (pix.w, pix.h), pix.samples)
            label = icon_cls.classify(fkey, roi_img)
            if label or icons_only:
                out[fname] = label or ""
                continue
        text = ''
        if prefer_pdf_text:
            text = get_text_from_pdf(page, rect)
        if not text:
            text = ocr_rect(page, rect, dpi, field=fname, preproc_args=preproc_args, autocrop=autocrop, pad_frac=pad_frac)
            # Optional debug crops
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                raw = _render_roi(page, rect, dpi)
                prefer_bright = (fname.lower() not in ("slot", "usage"))
                proc = _preprocess_image(raw, prefer_bright_text=prefer_bright, **(preproc_args or {}))
                if autocrop:
                    proc = _autocrop_binary(proc, pad_frac=pad_frac)
                base = os.path.splitext(os.path.basename(pdf_path))[0]
                raw.save(os.path.join(debug_dir, f"{base}_{fname}_raw.png"))
                proc.save(os.path.join(debug_dir, f"{base}_{fname}_proc.png"))
        out[fname] = postprocess_field(fname, text)
    doc.close()
    return out


def process_card_image(img_path: str, fields_norm: Dict[str, List[float]], dpi: int, icon_cls: Optional[IconClassifier] = None, debug_dir: Optional[str] = None, preproc_args: Optional[dict] = None, autocrop: bool = True, pad_frac: float = 0.06, icons_only: bool = False) -> Dict[str, str]:
    img_full = Image.open(img_path)
    W, H = img_full.size
    out: Dict[str, str] = {}
    for fname, nrect in fields_norm.items():
        x, y, w, h = nrect
        ix0 = int(round(x * W))
        iy0 = int(round(y * H))
        ix1 = int(round((x + w) * W))
        iy1 = int(round((y + h) * H))
        fkey = (fname or "").lower()
        # Optional horizontal padding for number field to avoid truncation
        if fkey == 'number':
            try:
                pad_frac_num = float(os.environ.get('OCR_NUMBER_PAD_FRAC', '0.0'))
            except Exception:
                pad_frac_num = 0.0
            pad_px = int(round((ix1 - ix0) * pad_frac_num))
            ix0 = max(0, ix0 - pad_px)
            ix1 = min(W, ix1 + pad_px)
        # Optional 2D padding for count field to ensure slash and both digits are captured
        if fkey == 'count':
            try:
                pad_x = float(os.environ.get('OCR_COUNT_PADX_FRAC', '0.05'))
                pad_y = float(os.environ.get('OCR_COUNT_PADY_FRAC', '0.05'))
            except Exception:
                pad_x, pad_y = 0.05, 0.05
            dx = int(round((ix1 - ix0) * pad_x))
            dy = int(round((iy1 - iy0) * pad_y))
            ix0 = max(0, ix0 - dx)
            ix1 = min(W, ix1 + dx)
            iy0 = max(0, iy0 - dy)
            iy1 = min(H, iy1 + dy)
        roi = img_full.crop((ix0, iy0, ix1, iy1))

        # Icon classification for slot/usage if templates provided
        if icon_cls is not None and fkey in ("slot", "usage"):
            label = icon_cls.classify(fkey, roi)
            if label or icons_only:
                out[fname] = label or ""
                continue

        # OCR
        field = fkey
        if os.environ.get('OCR_NUMBER_BRIGHT', '0') == '1':
            prefer_bright = (field not in ("slot", "usage"))
        else:
            prefer_bright = (field not in ("slot", "usage", "number"))
        # Optional upscaling for number field (helps thin digits on images)
        upscale = 1.0
        env_up = os.environ.get('OCR_NUMBER_UPSCALE')
        if field == 'number' and env_up:
            try:
                upscale = max(1.0, float(env_up))
            except Exception:
                upscale = 1.0
        if upscale > 1.0:
            nw = max(1, int(round(roi.width * upscale)))
            nh = max(1, int(round(roi.height * upscale)))
            roi = roi.resize((nw, nh), Image.LANCZOS)
        # Optional sharpening for number field
        if field == 'number' and os.environ.get('OCR_NUMBER_SHARPEN', '0') == '1':
            try:
                rad = float(os.environ.get('OCR_NUMBER_SHARPEN_RADIUS', '1.0'))
                amt = float(os.environ.get('OCR_NUMBER_SHARPEN_AMOUNT', '1.0'))
            except Exception:
                rad, amt = 1.0, 1.0
            arr = np.array(roi.convert('L'))
            arr = _unsharp_mask(arr, radius=rad, amount=amt)
            roi = Image.fromarray(arr)
        # Per-field thicken for title if requested via env
        pre_args = dict(preproc_args or {})
        if field == 'title' and os.environ.get('OCR_TITLE_THICKEN', '0') == '1':
            pre_args['thicken'] = True
        proc = _preprocess_image(roi, prefer_bright_text=prefer_bright, **pre_args)
        if autocrop:
            proc = _autocrop_binary(proc, pad_frac=pad_frac)
        
        psm = {
            'title': 7,
            'number': 7,
            'cost': 7,
            'count': 7,
            'source': 6,
            'usage': 7,
        }.get(field, 6)
        whitelist = None
        if field in ("number", "cost", "count"):
            whitelist = "0123456789"
        cfg = f"--oem 3 --psm {psm}"
        if whitelist:
            cfg += f" -c tessedit_char_whitelist={whitelist}"
        text = pytesseract.image_to_string(proc, config=cfg).strip()

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(img_path))[0]
            roi.save(os.path.join(debug_dir, f"{base}_{fname}_raw.png"))
            proc.save(os.path.join(debug_dir, f"{base}_{fname}_proc.png"))

        out[fname] = postprocess_field(fname, text)
    return out


def write_output(path: str, rows: List[Dict[str, str]], fmt: str = 'json'):
    fmt = fmt.lower()
    if fmt == 'jsonl':
        with open(path, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif fmt == 'json':
        with open(path, 'w') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    elif fmt == 'csv':
        import csv
        # Union of keys
        keys: List[str] = sorted({k for r in rows for k in r.keys()})
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description="OCR item cards using an ROI template")
    ap.add_argument("--input-dir", required=True, help="Folder with card PDFs or images (png/jpg/tiff)")
    ap.add_argument("--template", required=True, help="ROI template JSON")
    ap.add_argument("--output", required=True, help="Output file path")
    ap.add_argument("--format", default="json", choices=["json", "jsonl", "csv"], help="Output format")
    ap.add_argument("--key-field", default="number", help="Field to treat as the key")
    ap.add_argument("--dpi", type=int, default=600, help="Render DPI for OCR fallback")
    ap.add_argument("--prefer-pdf-text", action="store_true", help="Prefer PDF text extraction before OCR")
    ap.add_argument("--icons-dir", default="configs/icons", help="Icon templates directory with subfolders: slot/, usage/")
    ap.add_argument("--icon-method", default="both", choices=["hu", "tm", "both"], help="Icon matching method")
    ap.add_argument("--debug-crops", default=None, help="Directory to save per-field raw/proc crops for troubleshooting")
    ap.add_argument("--preproc", default="simple", choices=["simple", "bright", "strong"], help="OCR preprocessing pipeline")
    ap.add_argument("--block", type=int, default=35, help="Adaptive threshold block size (odd)")
    ap.add_argument("--C", type=int, default=10, help="Adaptive threshold constant C")
    ap.add_argument("--thicken", action="store_true", help="Slightly dilate text after binarization (helps titles)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of files processed (for testing)")
    ap.add_argument("--no-autocrop", dest="autocrop", action="store_false", help="Disable content-aware cropping inside ROIs")
    ap.add_argument("--pad-frac", type=float, default=0.06, help="Padding fraction for autocrop (relative to shortest side)")
    ap.add_argument("--icons-only", action="store_true", help="For slot/usage, use icon classification only (no OCR fallback)")
    args = ap.parse_args()

    tpl = load_template(args.template)
    fields_norm = tpl.get('fields', {})
    inputs = list_inputs(args.input_dir)
    if args.limit:
        inputs = inputs[: args.limit]

    icon_cls = IconClassifier(args.icons_dir if os.path.isdir(args.icons_dir) else None, method=args.icon_method)
    preproc_args = dict(mode=args.preproc, block_size=args.block, c_const=args.C, thicken=args.thicken)
    rows: List[Dict[str, str]] = []
    for path in inputs:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            data = process_card_pdf(path, fields_norm, dpi=args.dpi, prefer_pdf_text=args.prefer_pdf_text, icon_cls=icon_cls, debug_dir=args.debug_crops, preproc_args=preproc_args, autocrop=args.autocrop, pad_frac=args.pad_frac, icons_only=args.icons_only)
        else:
            data = process_card_image(path, fields_norm, dpi=args.dpi, icon_cls=icon_cls, debug_dir=args.debug_crops, preproc_args=preproc_args, autocrop=args.autocrop, pad_frac=args.pad_frac, icons_only=args.icons_only)
        data['__file'] = os.path.basename(path)
        key = data.get(args.key_field) or os.path.splitext(os.path.basename(path))[0]
        data['__key'] = key
        rows.append(data)

    write_output(args.output, rows, fmt=args.format)
    print(f"[ok] Wrote {len(rows)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
