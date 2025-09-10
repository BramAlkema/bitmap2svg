# bitmap2svg/ocr_text.py
from __future__ import annotations
from typing import List, Tuple
import pytesseract
from PIL import Image

def extract_text(image: Image.Image) -> str:
    """Extract text from an image using OCR."""
    return pytesseract.image_to_string(image)

def extract_text_with_boxes(image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Extract text and bounding boxes from an image using OCR."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # Confidence threshold
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            results.append((data['text'][i], (x, y, x + w, y + h)))
    return results