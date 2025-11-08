from datetime import datetime

from dateutil import parser
from pytesseract import pytesseract

from date_utils import DateComponentClassifier

date_component = DateComponentClassifier()
pytesseract_config = "--oem 3 --psm 11"


def ocr(img):
    data = pytesseract.image_to_data(
        img,
        config=pytesseract_config,
        output_type=pytesseract.Output.DICT
    )
    words = []
    bboxes = []
    for txt, left, top, width, height in zip(
        data["text"], data["left"], data["top"], data["width"], data["height"]
    ):
        if txt.strip():
            words.append(txt)
            bboxes.append([left, top, left + width, top + height])
    return words, bboxes


def texts_to_numbers(texts):
    numbers = []
    for text in texts:
        text = text.replace(",", ".").strip().lower()
        try:
            if text.endswith("k"):
                text = text[:-1].strip()
                num = float(text) * 1e3
            elif text.endswith("m"):
                text = text[:-1].strip()
                num = float(text) * 1e6
            elif text.endswith("b"):
                text = text[:-1].strip()
                num = float(text) * 1e9
            elif text.endswith("%"):
                text = text[:-1].strip()
                num = float(text) / 100.0
            else:
                num = float(text)
            numbers.append(num)
        except ValueError:
            continue
    return numbers


def texts_to_datetimes(texts):
    index = []
    date_components = [date_component.classify(text) for text in texts]

    # Handle short format like '12.19', '12/19', '12-19' as month and year
    if all(dc == "year and month" for dc in date_components):
        sep = next((s for s in [".", "/", "-"] if s in texts[0]), None)
        for t in texts:
            m, y = t.split(sep)
            year = int(y)
            year += 2000 if year < 100 else 0
            index.append(datetime(year, int(m), 1))
        return index

    # Handle alternating month and day with years (e.g. ['Dec', '12', '2025', ...])
    if "year" in date_components and "month" in date_components:
        result = []
        first_year = next(
            int(t) for t, dc in zip(texts, date_components) if dc == "year"
        )
        year = None
        month = None
        for t, dc in zip(texts, date_components):
            if dc == "year":
                year = int(t)
                month = 1
                result.append(datetime(year, month, 1))
            elif dc == "month":
                month = parser.parse(t.replace("‘", "").replace("’", "")).month
                result.append(datetime(year if year else first_year - 1, month, 1))
            elif dc == "day":
                result.append(
                    datetime(
                        year if year else first_year - 1, month if month else 1, int(t)
                    )
                )
        return result

    # Fallback: parse as full date
    for text in texts:
        text = text.strip()
        try:
            extracted_date = parser.parse(text, fuzzy=True)
        except (ValueError, TypeError):
            extracted_date = None
        index.append(extracted_date)
    return index
