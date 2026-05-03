"""PDF page-to-image conversion using PyMuPDF."""

import fitz  # PyMuPDF
from pathlib import Path


def pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def render_page_to_image(pdf_path: str, page_num: int, output_dir: str, zoom: float = 2.0) -> str:
    """Render a single PDF page to a JPEG image (1-indexed page_num).

    Returns the path to the saved image.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    image_path = out / f"page_{page_num:04d}.jpg"
    pix.save(str(image_path))
    doc.close()
    return str(image_path)


def render_all_pages(pdf_path: str, output_dir: str, zoom: float = 2.0) -> list[str]:
    """Render every page of a PDF to JPEG images. Returns list of image paths."""
    total = pdf_page_count(pdf_path)
    paths = []
    doc = fitz.open(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(total):
        page = doc[i]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = out / f"page_{i + 1:04d}.jpg"
        pix.save(str(image_path))
        paths.append(str(image_path))

    doc.close()
    return paths
