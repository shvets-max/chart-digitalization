import re
from datetime import datetime

import cv2
from dateutil import parser
from pytesseract import pytesseract

from date_utils import YEAR_REGEX, DateComponentClassifier

date_component = DateComponentClassifier()
pytesseract_config = "--oem 3 --psm 11"
OCR_UPSCALE_FACTOR = 3  # dashboard screenshots use ~7-10px tall axis/legend text,
# too small for tesseract to read reliably at native resolution
MIN_NATIVE_TOKEN_SIZE = 2  # px, at native (non-upscaled) resolution; upscaling can
# make a sub-pixel antialiasing artifact (e.g. a near-invisible legend line handle)
# large enough for tesseract to read as a spurious tiny glyph


def ocr(img):
    # INTER_LINEAR, not INTER_CUBIC: cubic's overshoot/ringing at sharp glyph
    # edges was measured to inflate tesseract's reported bounding boxes (e.g. a
    # text row growing several pixels taller than the glyphs themselves), which
    # then over-excludes real ink sitting just outside the true text -- linear
    # upscaling gives the same legibility gain without that side effect.
    upscaled = cv2.resize(
        img,
        None,
        fx=OCR_UPSCALE_FACTOR,
        fy=OCR_UPSCALE_FACTOR,
        interpolation=cv2.INTER_LINEAR,
    )
    data = pytesseract.image_to_data(
        upscaled, config=pytesseract_config, output_type=pytesseract.Output.DICT
    )
    words = []
    bboxes = []
    for txt, left, top, width, height in zip(
        data["text"], data["left"], data["top"], data["width"], data["height"]
    ):
        if not txt.strip():
            continue
        box = [
            round(v / OCR_UPSCALE_FACTOR)
            for v in (left, top, left + width, top + height)
        ]
        if (
            box[2] - box[0] < MIN_NATIVE_TOKEN_SIZE
            or box[3] - box[1] < MIN_NATIVE_TOKEN_SIZE
        ):
            continue
        words.append(txt)
        bboxes.append(box)
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
            numbers.append(None)
    return numbers


def texts_to_datetimes(texts):
    index = []
    date_components = [date_component.classify(text) for text in texts]

    # Handle short format like '12.19', '12/19', '12-19' (month, 2-digit year)
    # as well as '2019.12', '2019/12', '2019-12' (4-digit year, month) -- the
    # year is whichever part unambiguously matches a 4-digit year; when neither
    # part is unambiguous (both 2-digit, e.g. "12.19") the established
    # convention for this format is month first.
    if all(dc == "year and month" for dc in date_components):
        sep = next((s for s in [".", "/", "-"] if s in texts[0]), None)
        for t in texts:
            a, b = t.split(sep)
            if re.fullmatch(YEAR_REGEX, a):
                year_str, month_str = a, b
            elif re.fullmatch(YEAR_REGEX, b):
                year_str, month_str = b, a
            else:
                month_str, year_str = a, b
            year = int(year_str)
            year += 2000 if year < 100 else 0
            index.append(datetime(year, int(month_str), 1))
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
            else:
                result.append(None)
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
