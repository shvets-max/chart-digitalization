import re
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from dateutil import parser
from pytesseract import pytesseract

from src.date_utils import YEAR_REGEX, DateComponentClassifier

date_component = DateComponentClassifier()
PYTESSERACT_CONFIG = "--oem 3 --psm 11"
OCR_UPSCALE_FACTOR = 3  # dashboard screenshots use ~7-10px tall axis/legend text,
# too small for tesseract to read reliably at native resolution
MIN_NATIVE_TOKEN_SIZE = 2  # px, at native (non-upscaled) resolution; upscaling can
# make a sub-pixel antialiasing artifact (e.g. a near-invisible legend line handle)
# large enough for tesseract to read as a spurious tiny glyph

_YEAR_RE = re.compile(YEAR_REGEX)
_DUPLICATE_DECIMAL_RE = re.compile(r"\.{2,}")
_DATE_SEPARATORS = (".", "/", "-")
_SMART_QUOTES = str.maketrans("", "", "‘’")  # strip ' and '


def ocr(img: np.ndarray) -> tuple[list[str], list[list[int]]]:
    """OCR an image, returning recognized tokens and their (left, top, right, bottom) boxes."""
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
        upscaled, config=PYTESSERACT_CONFIG, output_type=pytesseract.Output.DICT
    )
    words: list[str] = []
    bboxes: list[list[int]] = []
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


_SUFFIX_MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9, "%": 1e-2}
# Digits OCR commonly confuses with a unit-suffix letter (e.g. "B" misread as
# "8": "15.5B" -> "15.58", "9B" -> "98"). Used to recover the suffix when a
# series is otherwise dominated by it.
_SUFFIX_DIGIT_CONFUSIONS = {"8": "b"}
_MIN_DOMINANT_SUFFIX_COUNT = 2  # ignore a single stray suffix as noise


def _normalize_number_text(text: str) -> str:
    text = text.replace(",", ".").strip().lower()
    # collapse a stray duplicate decimal separator, e.g. "10,.4B" -> "10..4b" -> "10.4b"
    return _DUPLICATE_DECIMAL_RE.sub(".", text)


def _parse_number_text(text: str) -> Optional[float]:
    """Parse a normalized OCR token into a float, honoring k/m/b/% suffixes."""
    for suffix, multiplier in _SUFFIX_MULTIPLIERS.items():
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)].strip()) * multiplier
            except ValueError:
                return None
    try:
        return float(text)
    except ValueError:
        return None


def texts_to_numbers(texts: list[str]) -> list[Optional[float]]:
    """Parse a series of OCR axis-label tokens (e.g. one axis' tick labels) into floats.

    Series-aware: if most tokens share a unit suffix (k/m/b/%), tokens ending
    in a digit OCR commonly confuses with that suffix's letter (e.g. "8" for
    "B") are corrected to use it, instead of being parsed as a wildly
    different plain number or dropped as unparsable.
    """
    normalized = [_normalize_number_text(t) for t in texts]

    suffix_counts: dict[str, int] = {}
    for text in normalized:
        for suffix in _SUFFIX_MULTIPLIERS:
            if text.endswith(suffix):
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
                break
    dominant_suffix = max(suffix_counts, key=suffix_counts.get, default=None)
    if dominant_suffix and suffix_counts[dominant_suffix] < _MIN_DOMINANT_SUFFIX_COUNT:
        dominant_suffix = None

    numbers: list[Optional[float]] = []
    for text in normalized:
        if (
            dominant_suffix
            and not text.endswith(dominant_suffix)
            and _SUFFIX_DIGIT_CONFUSIONS.get(text[-1:]) == dominant_suffix
        ):
            corrected = _parse_number_text(text[:-1] + dominant_suffix)
            if corrected is not None:
                numbers.append(corrected)
                continue
        numbers.append(_parse_number_text(text))
    return numbers


def texts_to_datetimes(texts: list[str]) -> list[Optional[datetime]]:
    """Parse a series of OCR axis-label tokens (e.g. one axis' tick labels) into datetimes."""
    date_components = [date_component.classify(text) for text in texts]

    # Handle short format like '12.19', '12/19', '12-19' (month, 2-digit year)
    # as well as '2019.12', '2019/12', '2019-12' (4-digit year, month) -- the
    # year is whichever part unambiguously matches a 4-digit year; when neither
    # part is unambiguous (both 2-digit, e.g. "12.19") the established
    # convention for this format is month first.
    if all(dc == "year and month" for dc in date_components):
        index: list[Optional[datetime]] = []
        sep = next((s for s in _DATE_SEPARATORS if s in texts[0]), None)
        for t in texts:
            a, b = t.split(sep)
            if _YEAR_RE.fullmatch(a):
                year_str, month_str = a, b
            elif _YEAR_RE.fullmatch(b):
                year_str, month_str = b, a
            else:
                month_str, year_str = a, b
            year = int(year_str)
            year += 2000 if year < 100 else 0
            index.append(datetime(year, int(month_str), 1))
        return index

    # Handle alternating month and day with years (e.g. ['Dec', '12', '2025', ...])
    if "year" in date_components and "month" in date_components:
        result: list[Optional[datetime]] = []
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
                month = parser.parse(t.translate(_SMART_QUOTES)).month
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
    index = []
    for text in texts:
        text = text.strip()
        try:
            extracted_date = parser.parse(text, fuzzy=True)
        except (ValueError, TypeError):
            extracted_date = None
        index.append(extracted_date)
    return index
