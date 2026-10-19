from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from lung_cancer_pipeline import (
    IMAGE_CLASS_NAMES,
    LOCAL_IMAGE_ARCHIVE_ROOT,
    TABULAR_FEATURE_COLUMNS,
    dataset_summary,
    load_image_model,
    load_tabular_dataset,
    load_tabular_model,
    log_prediction,
    predict_image,
    predict_tabular,
    train_image_model,
    train_tabular_model,
)

load_dotenv()


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _int_value(payload: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(payload.get(key, default))))
    except (TypeError, ValueError):
        return default


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "lung-cancer-local-api",
            "tabular_model_ready": load_tabular_model() is not None,
            "image_model_ready": load_image_model() is not None,
            "image_classes": IMAGE_CLASS_NAMES,
            "image_archive": str(LOCAL_IMAGE_ARCHIVE_ROOT),
        }
    )


@app.get("/api/dataset")
def dataset():
    try:
        return jsonify({"ok": True, "data": dataset_summary()})
    except Exception:  # noqa: BLE001 - model and data libraries raise varied runtime errors.
        return _error("Could not inspect the local dataset.", 500)


@app.get("/api/metrics")
def metrics():
    tabular = load_tabular_model()
    image = load_image_model()
    return jsonify(
        {
            "ok": True,
            "tabular": tabular.get("metrics") if tabular else None,
            "image": image.get("metrics") if image else None,
        }
    )


@app.get("/api/logs/predictions")
def prediction_logs():
    from lung_cancer_pipeline import PREDICTION_LOG_PATH, _read_json

    return jsonify({"ok": True, "data": _read_json(PREDICTION_LOG_PATH, [])})


@app.post("/api/train/tabular")
def train_tabular():
    payload = request.get_json(silent=True) or {}
    sample_size = _int_value(payload, "sample_size", 0, 0, 10000)
    try:
        result = train_tabular_model(sample_size=sample_size or len(load_tabular_dataset()))
        return jsonify({"ok": True, "metrics": result["metrics"]})
    except Exception:  # noqa: BLE001 - model and data libraries raise varied runtime errors.
        return _error("Tabular training failed. Verify the dataset and model configuration.", 500)


@app.post("/api/train/image")
def train_image():
    payload = request.get_json(silent=True) or {}
    sample_per_class = _int_value(payload, "sample_per_class", 50, 2, 500)
    epochs = _int_value(payload, "epochs", 8, 1, 50)
    batch_size = _int_value(payload, "batch_size", 16, 2, 64)
    fine_tune = bool(payload.get("fine_tune", True))
    try:
        result = train_image_model(
            sample_per_class=sample_per_class,
            epochs=epochs,
            batch_size=batch_size,
            fine_tune=fine_tune,
        )
        return jsonify({"ok": True, "metrics": result["metrics"]})
    except Exception:  # noqa: BLE001 - model and data libraries raise varied runtime errors.
        return _error("Image training failed. Verify the image archive and model configuration.", 500)


@app.post("/api/predict/tabular")
def predict_tabular_route():
    bundle = load_tabular_model()
    if bundle is None:
        return _error("Train the tabular model before making a prediction.", 409)
    payload = request.get_json(silent=True) or {}
    missing = [column for column in TABULAR_FEATURE_COLUMNS if column not in payload]
    if missing:
        return _error(f"Missing tabular fields: {', '.join(missing)}")
    try:
        result = predict_tabular(bundle["model"], {column: payload[column] for column in TABULAR_FEATURE_COLUMNS})
        log_prediction("tabular", result)
        return jsonify({"ok": True, "result": result})
    except Exception:  # noqa: BLE001 - preprocessing and estimator errors vary by input.
        return _error("Tabular prediction failed. Check that the supplied values are valid.", 400)


@app.post("/api/predict/image")
def predict_image_route():
    bundle = load_image_model()
    if bundle is None:
        return _error("Train the image model before making a prediction.", 409)
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return _error("Upload a JPG, JPEG, or PNG image.")
    try:
        result = predict_image(bundle["model"], upload.stream, bundle["class_names"])
        result["filename"] = upload.filename
        log_prediction("image", result)
        return jsonify({"ok": True, "result": result})
    except Exception:  # noqa: BLE001 - image decoders and model runtimes raise varied errors.
        return _error("Image prediction failed. Upload a valid JPG, JPEG, or PNG image.", 400)


@app.errorhandler(413)
def too_large(_error):
    return _error("The uploaded image is too large.", 413)


if __name__ == "__main__":
    port = int(os.environ.get("LUNG_APP_PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
