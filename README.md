# Chart Digitalization

## Overview
Extract time series data from chart images using OCR and image processing techniques in Python.
Leverage libraries such as Pytesseract, OpenCV, NumPy to identify chart regions, recognize text, detect axis scales and map values.

## Features

- Automatic detection of chart areas in images
- Optical Character Recognition (OCR) for axis labels and values
- Extraction of time series data points from line charts
- Web API + browser UI to upload an image and inspect the result on top of it

## Web API

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Open <http://127.0.0.1:8000/> to upload a chart image. The result view draws the
extracted series over the original picture, with optional overlays for the grid
and the numeric scale, a hover readout, and a CSV download. Interactive API docs
live at `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /api/charts` | Upload an image; returns the series with pixel + value coordinates |
| `GET /api/charts/{id}` | Re-fetch a processed chart |
| `GET /api/charts/{id}/ticks` | Grid lines / numeric scale, `source=generated\|detected` |
| `GET /api/charts/{id}/image` | The original uploaded image |
| `GET /api/charts/{id}/series.csv` | Extracted series as CSV |
| `DELETE /api/charts/{id}` | Drop a chart from the server |

Uploads are kept in memory only (the 20 most recent), so restarting the server
clears them.

## Library use

```python
from chart_extraction import extract_chart, extract_time_series

series = extract_time_series("chart.png")      # [(x_value, [y_value, ...]), ...]
result = extract_chart("chart.png")            # + pixel geometry, scales, grid
```
