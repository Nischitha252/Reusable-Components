#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_legends_regions_box_aware.py

Adds box-aware cropping:
- If a rectangular boundary surrounds the Legends area, crop that box precisely.
- Else, use heading→next-heading (ABBREVIATIONS/NOTES/HOLDS/REFERENCES) logic.
- OCR fallback for scanned/image-only PDFs.

Tested with: vector drawings (searchable) and scanned pages.
"""

from __future__ import annotations
import os, re, io
from dataclasses import dataclass
from typing import List, Tuple, Optional

import fitz  # PyMuPDF

# ----------------------- configuration / dictionaries ------------------------

HEADER_TERMS = {"LEGEND", "LEGENDS"}  # normalized tokens
STOP_TERMS   = {"ABBREVIATION", "ABBREVIATIONS", "NOTES", "HOLD", "HOLDS", "REFERENCE", "REFERENCES"}

BOX_MIN_AREA_FRAC   = 0.05   # candidate rectangle must be at least 1% of page area
BOX_MAX_AREA_FRAC   = 0.55   # and at most 50% of page area (avoid whole-page border)
BOX_MIN_ASPECT      = 0.65   # h/w >= 0.65 → prefer tall-ish panels (tune if needed)
HEADER_TOP_FRACTION = 0.50   # header should lie within top 35% of the detected box

# ----------------------------- helpers ---------------------------------------

@dataclass
class Word:
    x0: float; y0: float; x1: float; y1: float; text: str

def _norm_token(t: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", t.upper())

def _load_words(page: fitz.Page) -> List[Word]:
    raw = page.get_text("words") or []
    return [Word(w[0], w[1], w[2], w[3], w[4]) for w in raw]

def _find_first_header(words: List[Word]) -> Optional[Word]:
    cands = [w for w in words if _norm_token(w.text) in HEADER_TERMS]
    cands.sort(key=lambda w: (w.y0, w.x0))
    return cands[0] if cands else None

def _find_next_stop(words: List[Word], y_after: float) -> Optional[Word]:
    cands = [w for w in words if w.y0 > y_after - 1.0 and _norm_token(w.text) in STOP_TERMS]
    cands.sort(key=lambda w: (w.y0, w.x0))
    return cands[0] if cands else None

def _union_rect(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects: return None
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u

# -------------------------- image / cv utilities -----------------------------

def _render_page_to_pix(page: fitz.Page, dpi: int) -> fitz.Pixmap:
    scale = dpi / 72.0
    return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

def _pix_to_pil(pm: fitz.Pixmap):
    from PIL import Image
    mode = "RGB" if pm.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pm.width, pm.height], pm.samples)
    return img

def _pil_to_pix(img) -> fitz.Pixmap:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    doc = fitz.open(stream=buf.read(), filetype="png")
    pm = doc[0].get_pixmap(alpha=False)
    doc.close()
    return pm

def _page_to_numpy(page: fitz.Page, dpi: int):
    import numpy as np
    pm = _render_page_to_pix(page, dpi)
    img = _pix_to_pil(pm)
    arr = np.asarray(img)
    return arr, pm.width, pm.height, (dpi/72.0)

# --------------------- BOX-AWARE detector (OpenCV) ---------------------------

def _detect_box_containing_header(arr, header_xy_img: Tuple[int,int]) -> Optional[Tuple[int,int,int,int]]:
    """
    Detect the rectangular panel that contains the header point (cx, cy) in image coords.
    Returns (x, y, w, h) in image coords or None.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    H, W = arr.shape[:2]
    cx, cy = header_xy_img

    # Preprocess for sharp lines (works for both vector renders and scans)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Slight blur to suppress noise; then strong edge detection
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 150)

    # Close gaps in borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = W * H
    candidates = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < BOX_MIN_AREA_FRAC * page_area:     # too small
            continue
        if area > BOX_MAX_AREA_FRAC * page_area:     # too big (likely outer frame)
            continue

        # polygonal approximation to prefer rectangles
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if w == 0:  # guard
            continue
        aspect = h / float(w)
        if aspect < BOX_MIN_ASPECT:                  # too flat → unlikely legends panel
            continue

        # header point must be inside the rectangle
        if not (x <= cx <= x + w and y <= cy <= y + h):
            continue

        # the header should be near the top portion of the box (not at the bottom)
        if h > 0 and (cy - y) / float(h) > HEADER_TOP_FRACTION:
            continue

        # rectangularity: filled area / bbox area
        rect_area = area
        cnt_area = cv2.contourArea(c)
        rectangularity = cnt_area / max(1.0, rect_area)

        candidates.append(((x, y, w, h), rectangularity, area))

    if not candidates:
        return None

    # pick the smallest-area candidate (tends to be the tight legends box)
    # tie-breaker: higher rectangularity
    candidates.sort(key=lambda t: (t[2], -t[1]))
    x, y, w, h = candidates[0][0]

    # inset a few px to avoid drawing the border lines in the crop
    inset = max(4, int(min(w, h) * 0.01))
    x2 = max(0, x + inset); y2 = max(0, y + inset)
    w2 = max(1, w - 2 * inset); h2 = max(1, h - 2 * inset)
    return (x2, y2, w2, h2)

# ------------------- Text-first legends crop (no box) ------------------------

def _crop_legends_region_text(page: fitz.Page, dpi: int = 240) -> Optional[fitz.Pixmap]:
    words = _load_words(page)
    if not words:
        return None

    header = _find_first_header(words)
    if not header:
        return None

    # BOX-AWARE path first: render and try to detect rectangle containing header
    arr, W, H, scale = _page_to_numpy(page, dpi)
    header_cx = int(((header.x0 + header.x1) / 2.0) * scale)
    header_cy = int(((header.y0 + header.y1) / 2.0) * scale)
    box = _detect_box_containing_header(arr, (header_cx, header_cy))
    if box is not None:
        x, y, w, h = box
        # Crop directly from already-rendered image for speed
        import numpy as np
        crop = arr[y:y+h, x:x+w, :]
        from PIL import Image
        img = Image.fromarray(crop)
        return _pil_to_pix(img)

    # ---- fallback to heading→next-heading logic (no box) ----
    stop = _find_next_stop(words, y_after=header.y1)
    page_w, page_h = page.rect.width, page.rect.height
    y_top = header.y1 + 4
    y_bot = min(stop.y0 - 4, page_h) if stop else min(header.y1 + 0.35 * page_h, page_h)
    x_left_limit = max(0, header.x0 - 10)

    band_words = [w for w in words if (w.y0 >= y_top and w.y1 <= y_bot and w.x0 >= x_left_limit)]
    if len(band_words) < 5:
        x_left_limit = max(0, header.x0 - 40)
        band_words = [w for w in words if (w.y0 >= y_top and w.y1 <= y_bot and w.x0 >= x_left_limit)]

    text_rects = [fitz.Rect(w.x0, w.y0, max(w.x1, header.x1 + 120), w.y1) for w in band_words]
    union = _union_rect(text_rects)
    if not union or union.height < 15 or union.width < 80:
        union = fitz.Rect(x_left_limit, y_top, page_w - 8, y_bot)

    margin = 12
    clip = fitz.Rect(
        max(0, union.x0 - margin),
        max(0, union.y0 - margin),
        min(page_w, union.x1 + margin),
        min(page_h, union.y1 + margin),
    )
    scale = dpi / 72.0
    return page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)

# --------------------------- OCR fallback ------------------------------------

def _ocr_header_bbox_on_image(img) -> Optional[Tuple[int,int,int,int]]:
    """Return (x,y,w,h) of 'LEGEND(S)' on a PIL image via Tesseract, else None."""
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return None
    data = pytesseract.image_to_data(img, output_type=Output.DICT, lang="eng")
    hits = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt: continue
        if _norm_token(txt) in HEADER_TERMS:
            hits.append((data["left"][i], data["top"][i], data["width"][i], data["height"][i]))
    if not hits:
        return None
    # top-most
    return sorted(hits, key=lambda t: (t[1], t[0]))[0]

def _crop_legends_region_ocr(page: fitz.Page, dpi: int = 260) -> Optional[fitz.Pixmap]:
    pm = _render_page_to_pix(page, dpi)
    img = _pix_to_pil(pm)

    # 1) Find header via OCR
    hb = _ocr_header_bbox_on_image(img)
    if hb is None:
        return None

    # 2) Try to detect box that contains the header
    import numpy as np
    arr = np.asarray(img)
    x, y, w, h = hb
    cx = x + w // 2
    cy = y + h // 2
    box = _detect_box_containing_header(arr, (cx, cy))
    if box is not None:
        bx, by, bw, bh = box
        crop = arr[by:by+bh, bx:bx+bw, :]
        from PIL import Image
        out = Image.fromarray(crop)
        return _pil_to_pix(out)

    # 3) If no box detected, fallback to previous OCR vertical slice logic
    #    Crop below header to ~35% of page height, right side
    H, W = arr.shape[:2]
    y_top = y + h + 6
    y_bot = min(H, int(y_top + 0.35 * H))
    x0   = max(0, x - 25); x1 = W - 5
    from PIL import Image
    out = Image.fromarray(arr[y_top:y_bot, x0:x1, :])
    return _pil_to_pix(out)

# ------------------------------ orchestrator ---------------------------------

def extract_legends_regions(pdf_path: str, output_dir: str, use_ocr_if_needed: bool = True, dpi: int = 260) -> List[Tuple[int, str]]:
    """
    Returns a list of (page_number, saved_path) for each legends crop found.
    - Tries text+box path first.
    - Falls back to OCR path if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Tuple[int, str]] = []

    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)

            pm = _crop_legends_region_text(page, dpi=dpi)
            if pm is None and use_ocr_if_needed:
                pm = _crop_legends_region_ocr(page, dpi=dpi)

            if pm is not None:
                out_path = os.path.join(output_dir, f"legends_p{i+1}.png")
                pm.save(out_path)
                results.append((i + 1, out_path))

    return results

# ------------------------------- no CLI --------------------------------------

def main():
    # ===== EDIT THESE =====
    pdf_path   = r""
    output_dir = r""
    # ======================

    hits = extract_legends_regions(pdf_path, output_dir, use_ocr_if_needed=True, dpi=260)
    if hits:
        print("Legends panels saved:")
        for pg, pth in hits:
            print(f"  • Page {pg}: {pth}")
    else:
        print("No legends panels found. If this is a scanned PDF, keep OCR enabled and ensure Tesseract+Poppler are installed.")

if __name__ == "__main__":
    main()