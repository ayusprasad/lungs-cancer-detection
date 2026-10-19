from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
LOG_DIR = ROOT / "logs"

TABULAR_DATA_URL = "https://huggingface.co/datasets/nateraw/lung-cancer/resolve/main/survey%20lung%20cancer.csv?download=1"
TABULAR_DATA_PATH = DATA_DIR / "survey_lung_cancer.csv"
TABULAR_MODEL_PATH = ARTIFACT_DIR / "tabular_lung_cancer_model.joblib"

LOCAL_IMAGE_ARCHIVE_ROOT = DATA_DIR / "archive" / "lung_colon_image_set"
LOCAL_IMAGE_ROOT = LOCAL_IMAGE_ARCHIVE_ROOT / "Train and Validation Set"
LOCAL_IMAGE_TEST_ROOT = LOCAL_IMAGE_ARCHIVE_ROOT / "Test Set"
IMAGE_SAMPLE_ROOT = DATA_DIR / "lc25000_lung_sample"
IMAGE_MODEL_PATH = ARTIFACT_DIR / "lc25000_lung_model.keras"
IMAGE_METADATA_PATH = ARTIFACT_DIR / "lc25000_lung_model.json"
IMAGE_MODEL_NAME = "EfficientNetB0"
PREDICTION_LOG_PATH = LOG_DIR / "predictions_log.json"
TRAINING_LOG_PATH = LOG_DIR / "training_logs.json"
VALIDATION_LOG_PATH = LOG_DIR / "validation_logs.json"

TABULAR_FEATURE_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]

TABULAR_TARGET_COLUMN = "LUNG_CANCER"
IMAGE_CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n"]


def _ensure_parent_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent_dirs()
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def ensure_tabular_dataset() -> Path:
    _ensure_parent_dirs()
    if not TABULAR_DATA_PATH.exists():
        df = pd.read_csv(TABULAR_DATA_URL)
        df.to_csv(TABULAR_DATA_PATH, index=False)
    return TABULAR_DATA_PATH


def _normalize_tabular_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    required_columns = set(TABULAR_FEATURE_COLUMNS + [TABULAR_TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    gender_map = {"M": 1, "F": 0, "MALE": 1, "FEMALE": 0, 1: 1, 0: 0}
    target_map = {"YES": 1, "NO": 0, 1: 1, 0: 0}

    df["GENDER"] = df["GENDER"].astype(str).str.strip().str.upper().map(gender_map)
    df[TABULAR_TARGET_COLUMN] = (
        df[TABULAR_TARGET_COLUMN].astype(str).str.strip().str.upper().map(target_map)
    )

    for column in TABULAR_FEATURE_COLUMNS:
        if column in {"GENDER", "AGE"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    df = df.dropna(subset=TABULAR_FEATURE_COLUMNS + [TABULAR_TARGET_COLUMN]).reset_index(drop=True)
    return df


def load_tabular_dataset() -> pd.DataFrame:
    df = pd.read_csv(ensure_tabular_dataset())
    return _normalize_tabular_dataset(df)


def sample_tabular_frame(
    df: pd.DataFrame, sample_size: int = 50, random_state: int = 42
) -> pd.DataFrame:
    sample_size = max(10, min(int(sample_size), len(df)))
    if sample_size == len(df):
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[TABULAR_TARGET_COLUMN],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def build_tabular_model() -> Pipeline:
    numeric_features = ["AGE"]
    binary_features = [c for c in TABULAR_FEATURE_COLUMNS if c != "AGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "age",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "binary",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def train_tabular_model(sample_size: int = 50, random_state: int = 42) -> dict[str, Any]:
    df = load_tabular_dataset()
    sampled = sample_tabular_frame(df, sample_size=sample_size, random_state=random_state)

    X = sampled[TABULAR_FEATURE_COLUMNS].copy()
    y = sampled[TABULAR_TARGET_COLUMN].copy()
    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    test_size = max(1, min(10, len(sampled) // 5))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_tabular_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "sample_rows": len(sampled),
        "positive_rate": float(sampled[TABULAR_TARGET_COLUMN].mean()),
    }

    joblib.dump(
        {
            "model": model,
            "feature_columns": TABULAR_FEATURE_COLUMNS,
            "metrics": metrics,
        },
        TABULAR_MODEL_PATH,
    )

    return {
        "model": model,
        "metrics": metrics,
        "sampled": sampled,
        "test_frame": X_test.assign(
            actual=y_test.values,
            predicted=predictions,
            probability=probabilities,
        ),
    }


def load_tabular_model() -> dict[str, Any] | None:
    if not TABULAR_MODEL_PATH.exists():
        return None
    return joblib.load(TABULAR_MODEL_PATH)


def predict_tabular(model: Pipeline, features: dict[str, Any]) -> dict[str, float | int | str]:
    payload = pd.DataFrame([features], columns=TABULAR_FEATURE_COLUMNS)
    proba = float(model.predict_proba(payload)[0, 1])
    pred = int(model.predict(payload)[0])
    return {
        "prediction": "YES" if pred == 1 else "NO",
        "probability_yes": proba,
        "probability_no": 1.0 - proba,
    }


def ensure_lc25000_lung_sample(sample_per_class: int = 10) -> Path:
    _ensure_parent_dirs()
    if not LOCAL_IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Local LC25000 folder not found: {LOCAL_IMAGE_ROOT}. "
            "Point the app at your extracted dataset or place it there."
        )

    sample_per_class = max(2, sample_per_class)
    cached_manifest = _read_json(IMAGE_SAMPLE_ROOT / "manifest.json", {})
    if (
        IMAGE_SAMPLE_ROOT.exists()
        and cached_manifest.get("sample_per_class") == sample_per_class
        and cached_manifest.get("classes") == IMAGE_CLASS_NAMES
    ):
        return IMAGE_SAMPLE_ROOT

    if IMAGE_SAMPLE_ROOT.exists():
        shutil.rmtree(IMAGE_SAMPLE_ROOT)

    for class_name in IMAGE_CLASS_NAMES:
        (IMAGE_SAMPLE_ROOT / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in IMAGE_CLASS_NAMES:
        source_dir = LOCAL_IMAGE_ROOT / class_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {source_dir}")

        image_files = [
            p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not image_files:
            raise FileNotFoundError(f"No images found in: {source_dir}")

        selector = random.Random(42 + IMAGE_CLASS_NAMES.index(class_name))
        selected = image_files if len(image_files) <= sample_per_class else selector.sample(image_files, sample_per_class)
        for idx, image_path in enumerate(selected):
            out_path = IMAGE_SAMPLE_ROOT / class_name / f"{class_name}_{idx:04d}.jpg"
            with Image.open(image_path) as image:
                image.convert("RGB").save(out_path, format="JPEG", quality=95)

    manifest = {
        "sample_per_class": sample_per_class,
        "classes": IMAGE_CLASS_NAMES,
        "source": str(LOCAL_IMAGE_ROOT),
    }
    _write_json(IMAGE_SAMPLE_ROOT / "manifest.json", manifest)
    return IMAGE_SAMPLE_ROOT


def _build_image_datasets(
    sample_root: Path, image_size: tuple[int, int] = (224, 224), batch_size: int = 8
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        sample_root,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        class_names=IMAGE_CLASS_NAMES,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        sample_root,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        class_names=IMAGE_CLASS_NAMES,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds


def build_image_model(num_classes: int = len(IMAGE_CLASS_NAMES)) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        name="efficientnetb0",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="lc25000_lc_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_image_model(
    sample_per_class: int = 50,
    epochs: int = 8,
    batch_size: int = 16,
    fine_tune: bool = True,
    fine_tune_layers: int = 30,
) -> dict[str, Any]:
    sample_root = ensure_lc25000_lung_sample(sample_per_class=sample_per_class)
    train_ds, val_ds = _build_image_datasets(sample_root, batch_size=batch_size)

    model = build_image_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.25, patience=1, min_lr=1e-7
        ),
    ]

    epochs = max(1, int(epochs))
    head_epochs = epochs if not fine_tune or epochs == 1 else max(1, epochs // 2)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        callbacks=callbacks,
        verbose=0,
    )

    if fine_tune and epochs > head_epochs:
        base = model.get_layer("efficientnetb0")
        base.trainable = True
        fine_tune_layers = max(1, min(int(fine_tune_layers), len(base.layers)))
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=head_epochs,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )
        for key, values in fine_history.history.items():
            history.history.setdefault(key, []).extend(values)

    eval_loss, eval_accuracy = model.evaluate(val_ds, verbose=0)
    val_labels = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)
    val_probabilities = model.predict(val_ds, verbose=0)
    val_predictions = np.argmax(val_probabilities, axis=1)
    report = classification_report(
        val_labels,
        val_predictions,
        labels=list(range(len(IMAGE_CLASS_NAMES))),
        target_names=IMAGE_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "loss": float(eval_loss),
        "accuracy": float(eval_accuracy),
        "epochs_ran": len(history.history["loss"]),
        "sample_per_class": sample_per_class,
        "train_batches": tf.data.experimental.cardinality(train_ds).numpy(),
        "val_batches": tf.data.experimental.cardinality(val_ds).numpy(),
        "model": IMAGE_MODEL_NAME,
        "fine_tuned": bool(fine_tune and epochs > head_epochs),
        "fine_tune_layers": fine_tune_layers if fine_tune else 0,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            val_labels, val_predictions, labels=list(range(len(IMAGE_CLASS_NAMES)))
        ).tolist(),
    }

    model.save(IMAGE_MODEL_PATH, include_optimizer=False)
    training_rows = []
    for epoch_index in range(len(history.history["loss"])):
        training_rows.append(
            {
                "epoch": epoch_index + 1,
                "loss": float(history.history["loss"][epoch_index]),
                "accuracy": float(history.history["accuracy"][epoch_index]),
                "val_loss": float(history.history["val_loss"][epoch_index]),
                "val_accuracy": float(history.history["val_accuracy"][epoch_index]),
            }
        )
    _write_json(TRAINING_LOG_PATH, training_rows)
    _write_json(VALIDATION_LOG_PATH, metrics)
    _write_json(
        IMAGE_METADATA_PATH,
        {
            "class_names": IMAGE_CLASS_NAMES,
            "model": IMAGE_MODEL_NAME,
            "image_archive": str(LOCAL_IMAGE_ARCHIVE_ROOT),
            "image_source": str(LOCAL_IMAGE_ROOT),
            "sample_per_class": sample_per_class,
            "metrics": metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "model": model,
        "metrics": metrics,
        "class_names": IMAGE_CLASS_NAMES,
        "sample_root": sample_root,
    }


def load_image_model() -> dict[str, Any] | None:
    if not IMAGE_MODEL_PATH.exists() or not IMAGE_METADATA_PATH.exists():
        return None
    metadata = json.loads(IMAGE_METADATA_PATH.read_text(encoding="utf-8"))
    if (
        metadata.get("class_names") != IMAGE_CLASS_NAMES
        or metadata.get("metrics", {}).get("model") != IMAGE_MODEL_NAME
    ):
        return None
    model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    return {
        "model": model,
        "class_names": metadata.get("class_names", IMAGE_CLASS_NAMES),
        "metrics": metadata.get("metrics", {}),
    }


def dataset_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "tabular_rows": len(load_tabular_dataset()),
        "image_archive": str(LOCAL_IMAGE_ARCHIVE_ROOT),
        "image_source": str(LOCAL_IMAGE_ROOT),
        "image_test_source": str(LOCAL_IMAGE_TEST_ROOT),
        "image_classes": {},
    }
    for class_name in IMAGE_CLASS_NAMES:
        train_dir = LOCAL_IMAGE_ROOT / class_name
        test_dir = LOCAL_IMAGE_TEST_ROOT / class_name
        summary["image_classes"][class_name] = {
            "train": len(list(train_dir.glob("*"))) if train_dir.exists() else 0,
            "test": len(list(test_dir.glob("*"))) if test_dir.exists() else 0,
        }
    return summary


def log_prediction(modality: str, result: dict[str, Any]) -> None:
    logs = _read_json(PREDICTION_LOG_PATH, [])
    if not isinstance(logs, list):
        logs = []
    logs.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modality": modality,
            "result": result,
        }
    )
    _write_json(PREDICTION_LOG_PATH, logs[-200:])


def predict_image(model: tf.keras.Model, image_file: Any, class_names: list[str] | None = None) -> dict[str, Any]:
    class_names = class_names or IMAGE_CLASS_NAMES
    if hasattr(image_file, "read"):
        image_file.seek(0)
        image_file = BytesIO(image_file.read())
    image = tf.keras.utils.load_img(image_file, target_size=(224, 224))
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    probs = model.predict(array, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    return {
        "prediction": class_names[best_idx],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }
