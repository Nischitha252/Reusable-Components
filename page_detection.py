#!/usr/bin/env python3
"""
find_legends_pages.py

Usage:
  python find_legends_pages.py --pdf /path/to/file.pdf
  # Optional flags:
  #   --regex "\\blegend(s)?\\b"   # custom regex (default already detects Legend/Legends)
  #   --term "Legends"             # simple term; internally becomes a safe whole-word regex
  #   --case-sensitive             # default is case-insensitive
  #   --ocr                        # OCR fallback if nothing found via text extraction
  #   --ocr-dpi 300                # DPI for OCR rendering (default 300)

Notes:
- Requires: pip install pymupdf
- OCR fallback (optional): pip install pdf2image pillow pytesseract
  Also install system binaries: Tesseract and Poppler.
"""

from __future__ import annotations
import re
from typing import List, Pattern

def make_pattern(term: str | None, regex: str | None, case_sensitive: bool) -> Pattern:
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.compile(regex, flags)
    # Build a whole-word regex for Legend/Legends if only a term is given
    if term is None or term.strip() == "":
        term = "legends"
    # If user passes something like "Legends", still match singular/plural cleanly:
    base = r"legend(s)?"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"\b{base}\b", flags)

def normalize_text(text: str) -> str:
    # Join hyphenations across line breaks: e.g., "Leg-\nends" -> "Legends"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace newlines with spaces so word boundaries behave predictably
    return text.replace("\r", " ").replace("\n", " ")

def find_pages_with_pattern(pdf_path: str, pattern: Pattern) -> List[int]:
    import fitz  # PyMuPDF
    pages: List[int] = []
    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text = normalize_text(text)
            if pattern.search(text):
                # Return 1-based page numbers for human-friendly output
                pages.append(i + 1)
    return pages

def find_pages_with_pattern_ocr(pdf_path: str, pattern: Pattern, dpi: int = 300) -> List[int]:
    # OCR fallback for scanned drawings (image-only pages)
    from pdf2image import convert_from_path
    import pytesseract

    pages: List[int] = []
    # Convert all pages; restrict if your docs are very large
    images = convert_from_path(pdf_path, dpi=dpi)
    for idx, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="eng")
        text = normalize_text(text or "")
        if pattern.search(text):
            pages.append(idx + 1)
    return pages

def main():
    # ==== USER SETTINGS (edit these) ====
    pdf_path = r""   # <-- change to your local PDF path
    case_sensitive = False                   # True to match case
    term = "Legends"                         # Simple term; matches whole word "Legend/Legends"
    custom_regex = None                      # e.g., r"\blegend(s)?\b" (overrides `term` if set)
    use_ocr_if_needed = True             # Set True for scanned PDFs (requires Tesseract + Poppler + libs)
    ocr_dpi = 300                            # DPI used if OCR is enabled
    # ====================================

    pattern = make_pattern(term, custom_regex, case_sensitive)

    try:
        text_hits = find_pages_with_pattern(pdf_path, pattern)
    except Exception as e:
        print(f"[ERROR] Failed to read PDF '{pdf_path}': {e}")
        return

    if text_hits:
        print("Matches found (text extraction):", ", ".join(map(str, text_hits)))
        return

    if use_ocr_if_needed:
        try:
            ocr_hits = find_pages_with_pattern_ocr(pdf_path, pattern, dpi=ocr_dpi)
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
            return
        if ocr_hits:
            print("Matches found (OCR):", ", ".join(map(str, ocr_hits)))
        else:
            print("No matches found (text or OCR).")
    else:
        print("No matches found via text extraction. "
              "If your PDF is scanned, set use_ocr_if_needed=True.")

if __name__ == "__main__":
    main()