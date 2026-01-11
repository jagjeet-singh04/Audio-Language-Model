# Audio Language Model (ALM) — Django App

Listen, think, and understand speech in a simple web app. This project transcribes uploaded audio using Whisper (via `faster-whisper`) and performs lightweight Natural Language Understanding (NLU) with spaCy and Vader Sentiment.

## Features
- Transcription: Faster-Whisper (`tiny.en` by default, CPU-friendly)
- Understanding: entities (NER), key phrases (noun chunks), sentiment (Vader), basic intent heuristics
- Clean UI: modern upload page and detailed results view

## Requirements
- Python 3.11 (recommended)
- Windows (PowerShell) or macOS/Linux
- Internet access (first run downloads Whisper and spaCy model)

Optional:
- GPU acceleration (CTranslate2 CUDA build) — not required for `tiny.en` on CPU

## Quick Start (Windows)
Run these from the outer project folder: `Audio-Classification-Deep-Learning-main`.

```powershell
# 1) Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# 2) Install dependencies (inner folder requirements)
& "$PWD\.venv\Scripts\python.exe" -m pip install -r "${PWD}\Audio-Classification-Deep-Learning-main\requirements.txt"

# 3) Download spaCy English model
& "$PWD\.venv\Scripts\python.exe" -m spacy download en_core_web_sm

# 4) Start the Django server (from inner app folder)
cd "${PWD}\Audio-Classification-Deep-Learning-main"
& "${PWD}\..\.venv\Scripts\python.exe" manage.py runserver
```

Then open http://127.0.0.1:8000 in your browser.

## Usage
- Upload a short speech audio file (e.g., WAV/MP3)
- Wait for processing; the overlay indicates progress
- Review the results: transcript, intent, sentiment, key phrases, and entities
- Use the “Copy” button to copy the transcript

## Project Structure (key files)
- `AudioClassification/alm.py` — ALM pipeline (Whisper + spaCy + Vader)
- `AudioClassification/views.py` — upload handling and pipeline integration
- `templates/home.html` — upload page (ALM-focused UI)
- `templates/result.html` — results page (transcript + NLU)
- `static/styles/style.css` — shared theme and layout
- `static/js/script.js` — UX helpers (overlay, copy-to-clipboard)

## Configuration Notes
- Static files load in development from `static/`
- Uploaded audio is saved under `media/`
- Default Whisper model: `tiny.en` (good for CPU). Larger models (`base.en`, etc.) improve accuracy at the cost of speed

### Changing the Whisper model
Edit the ALM pipeline to use a different size (example):
- In `AudioClassification/alm.py`, set `model_size='base.en'` when calling transcription or adjust the `_get_whisper()` default accordingly

## Troubleshooting
- spaCy model missing: run `python -m spacy download en_core_web_sm`
- First run downloads Whisper weights; ensure internet connectivity
- Slow CPU performance: keep `tiny.en` or switch to `compute_type="int8"` (default). GPU use requires CUDA-enabled CTranslate2
- Static files not loading in dev: confirm `STATIC_URL` and default Django static settings; no `collectstatic` needed for dev

## License
This repository contains code provided for educational purposes.# About the Project

Kaggle Notebook link - <https://www.kaggle.com/abishekas11/audio-classification-using-deep-learning>.<br>
The demo of the project is explained here - <https://youtu.be/hJvr1dyiOxM>.

## To run this project you need

- python3
- pip

## Steps for running this project

- git clone this project and extract it and open a terminal in that folder itself.
- Create a virtual environment by running the following command
  - ```python3 -m venv audio-venv``` (audio-venv is the name of the virtual environment)
- Activate the virtual environment by running the following command
  - ```source audio-venv/bin/activate```
- Now run the following command
  - ```pip install -r requirements.txt```
- Now run the following command to start the Django server
  - ```python3 manage.py run server```
- To stop the server, press Ctrl+Shift+C or Ctrl+Alt+C
