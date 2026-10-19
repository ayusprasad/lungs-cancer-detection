# Lung Cancer Classification Lab

[![CI](https://github.com/ayusprasad/lungs-cancer-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/lungs-cancer-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/API-Flask-000000?logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)

A local research application that demonstrates a tabular survey classifier and an LC25000 histopathology image classifier with a browser dashboard and JSON API.

> This is an educational research demo, not a medical device. Predictions are not diagnoses and require review by qualified clinicians.

## Capabilities

- Train and evaluate a logistic-regression model from the survey CSV.
- Train an EfficientNetB0 image classifier from a local LC25000 archive.
- Predict from survey fields or an uploaded JPG/PNG image.
- Review dataset status and model metrics through a local Flask API.
- Use either the browser dashboard or the smaller Streamlit demo.

## Architecture

```text
frontend/ (HTML, CSS, JavaScript)
  └─ Flask API (server.py)
       └─ lung_cancer_pipeline.py
            ├─ Tabular survey model (scikit-learn)
            ├─ Histopathology model (TensorFlow / Keras)
            ├─ artifacts/ (saved models)
            └─ logs/ (local runtime output)

app.py → optional Streamlit interface over the same pipeline
```

## Tech stack

Python 3.11+, Flask, Streamlit, pandas, scikit-learn, TensorFlow/Keras, pytest, Ruff, GitHub Actions.

## Run locally

### Prerequisites

- Python 3.11 or newer
- The tabular survey CSV (downloaded automatically if it is absent)
- The LC25000 image archive for image-model training

### Install

```bash
git clone https://github.com/ayusprasad/lungs-cancer-detection.git
cd lungs-cancer-detection
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install the application dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure image data

Copy `.env.example` to `.env` if you want to change the local API port. For image training, extract the LC25000 archive so its class folders are under:

```text
data/archive/lung_colon_image_set/
  Train and Validation Set/
    colon_aca/
    colon_n/
    lung_aca/
    lung_n/
  Test Set/
    colon_aca/
    colon_n/
    lung_aca/
    lung_n/
```

The image archive and generated training samples are excluded from Git. The app uses a deterministic sample from the training folders for its local training demo.

### Start the browser dashboard

```bash
python server.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Set `LUNG_APP_PORT` in `.env` to use another port.

The Flask API exposes `GET /api/health`, `GET /api/dataset`, `GET /api/metrics`, and training/prediction routes under `/api/train` and `/api/predict`. API errors return a JSON message and an HTTP status.

### Start the Streamlit demo

```bash
streamlit run app.py
```

## Quality checks

```bash
ruff check app.py server.py lung_cancer_pipeline.py train.py tests
pytest -q
python -m compileall -q app.py server.py lung_cancer_pipeline.py train.py tests
```

GitHub Actions runs the same lint, API tests, and compile checks on pushes and pull requests. It does not retrain the neural network.

## Privacy and security

Runtime logs are local and ignored by Git because prediction inputs can contain health-related information. Keep patient data and trained private data out of commits. No API credentials are required.
