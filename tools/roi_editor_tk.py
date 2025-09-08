#!/usr/bin/env python3
"""
Tkinter ROI editor: click-drag to create boxes, prompt for a label,
show/manage regions in a side list, delete/update, and save as a template.

Saves compatible JSON (normalized 0..1) used by ocr_cards.py

Usage:
  python tools/roi_editor_tk.py \
    --input "out/items_png/Gloomhaven Items_p001_i01.png" \
    --output "configs/items_template.json" \
    --dpi 400 \
    [--template configs/items_template.json]

Controls:
  - Drag on image: create a rectangle; prompts for label.
  - Click an item in the list: select/highlight its rectangle.
  - Delete Selected button or Delete key: remove a region.
  - Save: writes JSON with fields[label] = [x,y,w,h] normalized.
  - Q: quit.

Notes:
  - If you prefer named fields (e.g., title, cost, ...), type that as the label.
  - If you prefer numbers, accept the default sequential number.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

# Optional drag-and-drop support via tkinterdnd2. If the native tkdnd
# library is unavailable or fails to load, we gracefully fall back to
# standard Tk without DND.
import os as _os
try:  # pragma: no cover
    if _os.environ.get("DISABLE_TKDND", ""):
        raise ImportError("tkdnd disabled by env")
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except Exception:  # pragma: no cover
    DND_FILES, TkinterDnD = None, None


def _select_base_tk():
    # Try to instantiate a TkinterDnD root to ensure tkdnd loads.
    # If it fails (common on macOS when tkdnd is missing), fall back to Tk.
    global DND_FILES
    if TkinterDnD is not None:
        try:
            tmp = TkinterDnD.Tk()  # may raise if tkdnd cannot load
            tmp.withdraw()
            tmp.destroy()
            return TkinterDnD.Tk
        except Exception:
            DND_FILES = None
            return tk.Tk
    return tk.Tk


@dataclass
class ROI:
    label: str
    x: int
    y: int
    w: int
    h: int

    def as_norm(self, iw: int, ih: int) -> List[float]:
        return [self.x / iw, self.y / ih, self.w / iw, self.h / ih]


def render_card(path: str, dpi: int, page_index: int = 0) -> Tuple[Image.Image, Tuple[int, int]]:
    """Load a card image from either a PDF (first page by default) or an image file.

    Returns a PIL image and its (width, height).
    """
    lower = path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        img = Image.open(path).convert("RGB")
        return img, img.size
    # Fallback to PDF rendering
    doc = fitz.open(path)
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.n == 3 else "L"
    img = Image.frombytes(mode, (pix.w, pix.h), pix.samples)
    size = (pix.w, pix.h)
    doc.close()
    return img, size


BaseTk = _select_base_tk()


class Editor(BaseTk):
    def __init__(self, img: Image.Image, source_path: str, out_path: str, dpi: int = 400):
        super().__init__()
        self.title("ROI Editor (Tk)")

        self.source_path = source_path
        self.out_path = out_path
        self.dpi = dpi

        self.img_orig = img
        self.iw, self.ih = img.size

        # Display scale to fit on screen
        max_w, max_h = 1200, 900
        scale_w = max_w / self.iw
        scale_h = max_h / self.ih
        self.scale = min(1.0, scale_w, scale_h)
        disp_w = int(self.iw * self.scale)
        disp_h = int(self.ih * self.scale)

        self.img_disp = img.resize((disp_w, disp_h), Image.LANCZOS) if self.scale != 1.0 else img
        self.tk_img = ImageTk.PhotoImage(self.img_disp, master=self)

        # UI layout: left canvas, right controls
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, width=disp_w, height=disp_h, bg="#222")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        side = ttk.Frame(self)
        side.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)

        ttk.Label(side, text=os.path.basename(self.source_path)).grid(row=0, column=0, sticky="w")

        self.listbox = tk.Listbox(side, height=25)
        self.listbox.grid(row=1, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        btns = ttk.Frame(side)
        btns.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for i in range(7):
            btns.columnconfigure(i, weight=1)
        ttk.Button(btns, text="Delete Selected", command=self.delete_selected).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Save", command=self.save).grid(row=0, column=1, sticky="ew")
        ttk.Button(btns, text="Open Card", command=self.open_card_dialog).grid(row=0, column=2, sticky="ew")
        ttk.Button(btns, text="Load", command=self.load_dialog).grid(row=0, column=3, sticky="ew")
        ttk.Button(btns, text="Clear All", command=self.clear_all).grid(row=0, column=4, sticky="ew")
        ttk.Button(btns, text="Quit", command=self.destroy).grid(row=0, column=5, sticky="ew")
        ttk.Button(btns, text="Capture Template", command=self.capture_template).grid(row=0, column=6, sticky="ew")

        self.status = tk.StringVar(value="Drag to create a region; release to label it.")
        ttk.Label(side, textvariable=self.status).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        # ROI state
        self.rois: List[ROI] = []
        self.rect_items: Dict[int, int] = {}  # listbox index -> canvas rect id
        self.text_items: Dict[int, int] = {}  # listbox index -> canvas text id
        self.current_rect: Optional[int] = None
        self.drag_start: Optional[Tuple[int, int]] = None

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<KeyPress-s>", lambda e: self.save())
        self.bind("<KeyPress-q>", lambda e: self.destroy())
        self.bind("<KeyPress-l>", lambda e: self.load_dialog())
        self.bind("<KeyPress-o>", lambda e: self.open_card_dialog())

        # Enable drag-and-drop on the canvas if tkinterdnd2 is available
        if DND_FILES:
            try:
                self.canvas.drop_target_register(DND_FILES)  # type: ignore
                self.canvas.dnd_bind("<<Drop>>", self.on_drop)  # type: ignore
            except Exception:
                # If binding fails, silently continue without DND
                pass

    def canvas_to_image(self, x: int, y: int) -> Tuple[int, int]:
        return int(x / self.scale), int(y / self.scale)

    def on_mouse_down(self, event):
        self.drag_start = (event.x, event.y)
        if self.current_rect is not None:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

    def on_mouse_move(self, event):
        if not self.drag_start:
            return
        sx, sy = self.drag_start
        x0, y0 = min(sx, event.x), min(sy, event.y)
        x1, y1 = max(sx, event.x), max(sy, event.y)
        if self.current_rect is None:
            self.current_rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00cc66", width=2)
        else:
            self.canvas.coords(self.current_rect, x0, y0, x1, y1)

    def on_mouse_up(self, event):
        if not self.drag_start or self.current_rect is None:
            self.drag_start = None
            return
        sx, sy = self.drag_start
        x0, y0 = min(sx, event.x), min(sy, event.y)
        x1, y1 = max(sx, event.x), max(sy, event.y)
        self.drag_start = None

        # Convert to image coordinates
        ix0, iy0 = self.canvas_to_image(x0, y0)
        ix1, iy1 = self.canvas_to_image(x1, y1)
        w, h = max(0, ix1 - ix0), max(0, iy1 - iy0)
        if w < 5 or h < 5:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            return

        default_label = str(len(self.rois) + 1)
        label = simpledialog.askstring("Region Label", "Enter label (e.g., title, cost, or a number)", initialvalue=default_label, parent=self)
        if not label:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            return

        roi = ROI(label=label.strip(), x=ix0, y=iy0, w=w, h=h)
        self.rois.append(roi)
        self.add_list_item(roi)

        # finalize rectangle (keep it) and add label text
        cx0, cy0, cx1, cy1 = self.canvas.coords(self.current_rect)
        text_id = self.canvas.create_text(cx0 + 4, max(10, cy0 - 8), text=roi.label, anchor="w", fill="#00cc66", font=("TkDefaultFont", 11, "bold"))
        idx = self.listbox.size() - 1
        self.rect_items[idx] = self.current_rect
        self.text_items[idx] = text_id
        self.current_rect = None
        self.status.set(f"Added region '{roi.label}'")

    def add_list_item(self, roi: ROI):
        self.listbox.insert(tk.END, f"{roi.label}  ({roi.x},{roi.y},{roi.w},{roi.h})")

    def on_list_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        # Highlight selected rectangle
        for i, rect_id in self.rect_items.items():
            color = "#00cc66" if i == idx else "#ff6600"
            try:
                self.canvas.itemconfig(rect_id, outline=color)
            except Exception:
                pass

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        # Remove from canvas
        if idx in self.rect_items:
            self.canvas.delete(self.rect_items[idx])
            self.canvas.delete(self.text_items.get(idx, 0))
            self.rect_items.pop(idx, None)
            self.text_items.pop(idx, None)
        # Remove from list/rois
        self.listbox.delete(idx)
        self.rois.pop(idx)
        # Rebuild canvas/indices for remaining items
        self.rebuild_canvas_items()
        self.status.set("Deleted selected region")

    def clear_all(self):
        for rect_id in list(self.rect_items.values()):
            self.canvas.delete(rect_id)
        for text_id in list(self.text_items.values()):
            self.canvas.delete(text_id)
        self.rect_items.clear()
        self.text_items.clear()
        self.listbox.delete(0, tk.END)
        self.rois.clear()
        self.status.set("Cleared all regions")

    def rebuild_canvas_items(self):
        # Clear all, redraw from self.rois
        for rect_id in list(self.rect_items.values()):
            self.canvas.delete(rect_id)
        for text_id in list(self.text_items.values()):
            self.canvas.delete(text_id)
        self.rect_items.clear()
        self.text_items.clear()
        self.listbox.delete(0, tk.END)

        for i, roi in enumerate(self.rois):
            # Add to list
            self.listbox.insert(tk.END, f"{roi.label}  ({roi.x},{roi.y},{roi.w},{roi.h})")
            # Draw scaled rectangle
            x0 = int(roi.x * self.scale)
            y0 = int(roi.y * self.scale)
            x1 = int((roi.x + roi.w) * self.scale)
            y1 = int((roi.y + roi.h) * self.scale)
            rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00cc66", width=2)
            text_id = self.canvas.create_text(x0 + 4, max(10, y0 - 8), text=roi.label, anchor="w", fill="#00cc66", font=("TkDefaultFont", 11, "bold"))
            self.rect_items[i] = rect_id
            self.text_items[i] = text_id

    def capture_template(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Capture", "Select a region to capture as a template (slot/usage)")
            return
        idx = sel[0]
        roi = self.rois[idx]
        group = simpledialog.askstring("Template Group", "Enter group: slot or usage", initialvalue=("slot" if roi.label.lower()=="slot" else "usage"), parent=self)
        if not group:
            return
        group = group.strip().lower()
        if group not in ("slot", "usage"):
            messagebox.showerror("Capture", "Group must be 'slot' or 'usage'")
            return
        name = simpledialog.askstring("Template Label", "Enter class label (e.g., Head, Consumed, Spent)", parent=self)
        if not name:
            return
        name = name.strip()

        # Crop from original img
        x0, y0, x1, y1 = roi.x, roi.y, roi.x + roi.w, roi.y + roi.h
        x0 = max(0, min(self.iw - 1, x0))
        y0 = max(0, min(self.ih - 1, y0))
        x1 = max(1, min(self.iw, x1))
        y1 = max(1, min(self.ih, y1))
        crop = self.img_orig.crop((x0, y0, x1, y1))

        out_dir = os.path.join("configs", "icons", group)
        os.makedirs(out_dir, exist_ok=True)
        # safe filename
        safe = "".join(c for c in name if c.isalnum() or c in ("_", "-", " ")).strip().replace(" ", "_")
        out_path = os.path.join(out_dir, f"{safe}.png")
        crop.save(out_path)
        self.status.set(f"Captured template: {out_path}")
        messagebox.showinfo("Captured", f"Saved template to\n{out_path}")

    # ---------- Loading / swapping the displayed card ----------
    def open_card_dialog(self):
        path = filedialog.askopenfilename(
            title="Open card",
            filetypes=[
                ("Card images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.swap_card(path)

    def on_drop(self, event):  # type: ignore[no-redef]
        """Handle file(s) dropped onto the canvas."""
        data = event.data
        if not data:
            return
        # tkdnd uses space-separated paths, braces for spaces
        paths: List[str] = []
        token = ''
        in_brace = False
        for ch in data:
            if ch == '{':
                in_brace = True
                token = ''
            elif ch == '}':
                in_brace = False
                paths.append(token)
                token = ''
            elif ch == ' ' and not in_brace:
                if token:
                    paths.append(token)
                    token = ''
            else:
                token += ch
        if token:
            paths.append(token)
        for p in paths:
            if p.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.swap_card(p)
                break

    def swap_card(self, pdf_path: str):
        try:
            new_img, _ = render_card(pdf_path, dpi=self.dpi)
        except Exception as e:
            messagebox.showerror("Open", f"Failed to load card:\n{e}")
            return
        # Convert current ROIs to normalized based on old size
        prev_w, prev_h = self.iw, self.ih
        rois_norm = [r.as_norm(prev_w, prev_h) for r in self.rois]

        # Swap image and recompute scaling
        self.source_path = pdf_path
        self.img_orig = new_img
        self.iw, self.ih = new_img.size

        max_w, max_h = 1200, 900
        scale_w = max_w / self.iw
        scale_h = max_h / self.ih
        self.scale = min(1.0, scale_w, scale_h)
        disp_w = int(self.iw * self.scale)
        disp_h = int(self.ih * self.scale)
        self.img_disp = new_img.resize((disp_w, disp_h), Image.LANCZOS) if self.scale != 1.0 else new_img
        self.tk_img = ImageTk.PhotoImage(self.img_disp, master=self)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.itemconfigure(self.canvas_img, image=self.tk_img)

        # Rebuild ROIs from normalized coords to new pixel size
        self.rois = [ROI(label=self.rois[i].label,
                         x=int(round(nx * self.iw)),
                         y=int(round(ny * self.ih)),
                         w=int(round(nw * self.iw)),
                         h=int(round(nh * self.ih)))
                     for i, (nx, ny, nw, nh) in enumerate(rois_norm)]
        self.rebuild_canvas_items()
        self.status.set(f"Loaded card: {os.path.basename(pdf_path)}")

    def save(self):
        if not self.rois:
            messagebox.showwarning("Save", "No regions to save")
            return
        fields: Dict[str, List[float]] = {}
        for roi in self.rois:
            fields[roi.label] = roi.as_norm(self.iw, self.ih)
        tpl = {
            "dpi": 400,
            "fields": fields,
        }
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, 'w') as f:
            json.dump(tpl, f, indent=2)
        self.status.set(f"Saved template to {self.out_path}")
        messagebox.showinfo("Saved", f"Saved template to\n{self.out_path}")

    def load_from_template(self, tpl_path: str):
        if not os.path.isfile(tpl_path):
            messagebox.showerror("Load", f"Template not found:\n{tpl_path}")
            return
        try:
            with open(tpl_path, 'r') as f:
                tpl = json.load(f)
        except Exception as e:
            messagebox.showerror("Load", f"Failed to read template:\n{e}")
            return
        fields = tpl.get('fields') or {}
        if not isinstance(fields, dict) or not fields:
            messagebox.showwarning("Load", "Template has no fields")
            return
        # Clear current and import
        self.clear_all()
        for label, arr in fields.items():
            try:
                x, y, w, h = arr
            except Exception:
                continue
            # If values look normalized (<= 1.0), scale to pixels
            if max(x, y, w, h) <= 1.0:
                ix = int(round(x * self.iw))
                iy = int(round(y * self.ih))
                iw = int(round(w * self.iw))
                ih = int(round(h * self.ih))
            else:
                ix, iy, iw, ih = int(x), int(y), int(w), int(h)
            if iw < 1 or ih < 1:
                continue
            self.rois.append(ROI(label=label, x=ix, y=iy, w=iw, h=ih))
        self.rebuild_canvas_items()
        self.status.set(f"Loaded {len(self.rois)} regions from template")

    def load_dialog(self):
        # Use previously set output path if a sibling template exists
        guess = self.out_path if os.path.isfile(self.out_path) else os.path.join(os.path.dirname(self.out_path) or '.', 'items_template.json')
        path = simpledialog.askstring("Load Template", "Path to template JSON", initialvalue=guess, parent=self)
        if path:
            self.load_from_template(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tkinter ROI editor for card templates")
    ap.add_argument("--input", required=True, help="Sample card PDF path")
    ap.add_argument("--output", required=True, help="Output template JSON path")
    ap.add_argument("--template", help="Existing template JSON to load and prepopulate")
    ap.add_argument("--dpi", type=int, default=400, help="Render DPI for editing")
    args = ap.parse_args()

    img, _ = render_card(args.input, dpi=args.dpi)
    app = Editor(img=img, source_path=args.input, out_path=args.output)
    if args.template:
        app.load_from_template(args.template)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
