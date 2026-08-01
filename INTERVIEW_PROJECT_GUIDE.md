# Lung Cancer Classification Lab

## Interview-Ready Project Explanation

This document explains the project as it is actually implemented in this repository. It covers the problem, data, architecture, model choices, training process, frontend, API, evaluation, advantages, disadvantages, limitations, and likely interview questions.

## 1. Short Project Summary

This is a local multimodal cancer-classification application with two separate machine-learning workflows:

1. A tabular machine-learning model that predicts a positive or negative lung-cancer survey label from patient questionnaire features.
2. A computer-vision model that classifies histopathology images from the local LC25000 archive into the four folder labels available in the dataset: `colon_aca`, `colon_n`, `lung_aca`, and `lung_n`.

The models are exposed through a Flask API and a browser dashboard. A user can train the models, upload an image, view the predicted class and probabilities, and test the tabular model through form inputs.

This is a research and learning prototype, not a clinical diagnostic system.

### One-minute interview answer

> I built a local multimodal cancer-classification application. For structured survey data, I used a scikit-learn preprocessing pipeline with balanced logistic regression because the dataset is small, mostly binary, and benefits from an interpretable baseline. For images, I used ImageNet-pretrained EfficientNetB0 transfer learning. I replaced its original classification head with a four-class softmax head matching the LC25000 folders, froze the base initially, and then fine-tuned the final 30 layers with a lower learning rate. I added controlled augmentation, early stopping, validation metrics, saved model artifacts, JSON training logs, prediction logging, and a Flask API with a responsive browser frontend. I tested the complete workflow locally with the actual image archive and an uploaded LC25000 image.

## 2. Problem Statement

The original notebook was designed for a hosted notebook environment. It used environment-specific paths and had fragile training cells. The goal was to convert it into a reproducible local application that:

- Reads the local LC25000 image archive.
- Reads and cleans the tabular lung-cancer survey data.
- Uses a traditional ML model for structured data.
- Uses a pretrained CNN for image data.
- Saves models so prediction does not require retraining every time.
- Provides a browser interface for training and prediction.
- Exposes predictable API endpoints.
- Records metrics and predictions.
- Runs on a normal local Windows machine.

## 3. Dataset Structure

The local image archive is:

```text
C:\Users\kumar\Desktop\lung cancer detection\data\archive\lung_colon_image_set
```

The application uses this structure:

```text
lung_colon_image_set/
├── Train and Validation Set/
│   ├── colon_aca/
│   ├── colon_n/
│   ├── lung_aca/
│   └── lung_n/
└── Test Set/
    ├── colon_aca/
    ├── colon_n/
    ├── lung_aca/
    └── lung_n/
```

The observed local counts are:

| Class | Training/validation folder | Test folder |
|---|---:|---:|
| `colon_aca` | 500 | 96 |
| `colon_n` | 499 | 96 |
| `lung_aca` | 500 | 108 |
| `lung_n` | 500 | 84 |
| **Total** | **1,999** | **384** |

The four class labels are used exactly as the folder names. This means the image model is not a four-subtype lung-only model: it contains two colon categories and two lung categories because those are the labels in the supplied LC25000 archive.

The tabular CSV is downloaded once from the configured URL if it is not already present and cached at:

```text
data/survey_lung_cancer.csv
```

The current cleaned tabular dataset contains 309 usable rows.

## 4. Architecture

```text
                         Local browser
                              |
                    frontend/index.html
                    frontend/script.js
                    frontend/style.css
                              |
                       Flask REST API
                         server.py
                              |
                  lung_cancer_pipeline.py
                    /                    \
       Tabular ML pipeline              Image CNN pipeline
       pandas + scikit-learn             TensorFlow + Keras
                    |                    |
       tabular_lung_cancer_model.joblib  lc25000_lung_model.keras
                    |                    |
             metrics and prediction logs in logs/
```

### Component responsibilities

| Component | Responsibility |
|---|---|
| `lung_cancer_pipeline.py` | Shared data loading, preprocessing, training, inference, persistence, and logging |
| `server.py` | Flask API and static frontend server |
| `frontend/index.html` | Dashboard structure and controls |
| `frontend/script.js` | API calls, tab switching, upload handling, and result rendering |
| `frontend/style.css` | Responsive visual design |
| `train.py` | Command-line training entry point |
| `app.py` | Older Streamlit interface retained for experimentation |
| `artifacts/` | Saved models and metadata |
| `logs/` | Training, validation, and prediction records |
| `.gitignore` | Keeps the large local image archive out of GitHub |

## 5. Why Two Different Models?

The project has two data modalities, so one algorithm is not appropriate for both.

### Tabular data

The survey contains values such as age, smoking, anxiety, wheezing, coughing, and chest pain. These are structured numeric or binary values. Traditional ML algorithms can work directly with this feature table.

### Image data

The histopathology images contain spatial patterns such as edges, textures, cell structures, and tissue arrangements. CNNs are designed to learn spatial features from images.

Separate models allow each modality to use an appropriate representation rather than forcing both inputs into one unsuitable model.

## 6. Tabular Machine-Learning Pipeline

### Features and target

The 15 input features are:

```text
GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING,
COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN
```

The target column is:

```text
LUNG_CANCER
```

### Cleaning and normalization

The pipeline:

1. Copies the DataFrame.
2. Removes whitespace from column names.
3. Maps gender values such as `M`, `F`, `MALE`, and `FEMALE` to numeric values.
4. Maps target values such as `YES` and `NO` to `1` and `0`.
5. Converts numeric fields with `pd.to_numeric(..., errors="coerce")`.
6. Drops rows missing required features or the target.

### Preprocessing

The `ColumnTransformer` separates the continuous age feature from binary questionnaire features.

For `AGE`:

- Missing values use median imputation.
- Values are standardized with `StandardScaler`.

For the binary features:

- Missing values use most-frequent imputation.

The preprocessing is inside the scikit-learn `Pipeline`, so training and prediction use the same transformations. This prevents a common deployment error where inference data is transformed differently from training data.

### Classifier: balanced logistic regression

The classifier is configured as:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
```

#### Why logistic regression?

- The dataset is small.
- Most features are binary questionnaire values.
- The output is binary.
- It provides probabilities with `predict_proba`.
- It is fast to train and deploy.
- Its coefficients can be inspected for interpretability.
- It creates a strong, explainable baseline before trying complex models.

#### Why balanced class weights?

The current tabular data has a positive rate of approximately 87.4%. A model that always predicts the majority class could obtain deceptively high accuracy. `class_weight="balanced"` increases the influence of the less frequent class when calculating training loss.

### Tabular evaluation result

The current full-data saved run used:

- 309 usable rows.
- 299 training rows.
- 10 holdout rows.
- 80.0% accuracy.
- 100.0% precision.
- 77.8% recall.
- 87.5% F1 score.

The holdout is very small, so these results demonstrate the pipeline but should not be presented as clinically reliable performance.

### Tabular prediction flow

1. The browser collects form values.
2. JavaScript converts them into JSON.
3. The browser sends the JSON to `/api/predict/tabular`.
4. Flask checks that all expected fields are present.
5. The saved pipeline preprocesses the values.
6. Logistic regression returns a class and probability.
7. The API returns `YES` or `NO` and both probabilities.
8. The prediction is appended to `logs/predictions_log.json`.

## 7. Image CNN Pipeline

### Image source

The code explicitly uses:

```text
data/archive/lung_colon_image_set/Train and Validation Set
```

The class order is fixed explicitly:

```python
IMAGE_CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n"]
```

Explicit ordering prevents class-index mismatches between training, saved metadata, and prediction.

### Why transfer learning?

The local compute environment and short training requirement make transfer learning more appropriate than training a CNN from random initialization. The pretrained feature extractor already knows useful low-level visual patterns. The project only needs to adapt higher-level features to the LC25000 classes.

### CNN choice: EfficientNetB0

The model uses ImageNet-pretrained EfficientNetB0 without its original classifier:

```python
tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
```

#### Why EfficientNetB0?

- It is directly supported by TensorFlow/Keras.
- It offers a good accuracy-to-compute balance.
- B0 is practical on a local CPU compared with larger variants.
- `include_top=False` allows a task-specific classification head.
- It is more capable than a minimal CNN while remaining lightweight enough for local testing.

#### Why not train from scratch?

Training from scratch would require the model to learn low-level visual features from the local dataset. That would take longer and increase overfitting risk, especially for the short sample run.

#### Why not a larger model?

EfficientNetB3, ResNet50, or larger architectures may perform better with enough data and compute, but they use more memory, take longer to train, and are less convenient for a local demo. EfficientNetB0 is a sensible first transfer-learning baseline.

### Image preprocessing and augmentation

Images are resized to `224 x 224` and loaded with TensorFlow's `image_dataset_from_directory`.

Training augmentation includes:

- Random horizontal flip.
- Small random rotation.
- Small random zoom.

The model also includes EfficientNet preprocessing. Keeping preprocessing in the model graph reduces the chance that training and inference use different image scaling.

The TensorFlow dataset pipeline uses:

- A fixed 20% validation split.
- Seed 42.
- Caching.
- Training shuffle.
- Prefetching with `tf.data.AUTOTUNE`.

### CNN classification head

The ImageNet classification head is replaced with:

1. Global average pooling.
2. Dropout at rate 0.2.
3. A dense layer with four softmax outputs.

Global average pooling reduces feature maps to a compact vector. Dropout reduces overfitting. Softmax returns four probabilities whose sum is one.

### Two-stage training

#### Stage 1: train the new head

The EfficientNet base is frozen. The new four-class head learns the mapping from pretrained features to LC25000 labels. The initial Adam learning rate is `1e-3`.

#### Stage 2: fine-tune selected layers

For multi-epoch runs:

- The EfficientNet base is made trainable.
- Only the final 30 layers are unfrozen.
- Batch-normalization layers remain frozen for stability.
- Adam is recompiled with learning rate `1e-5`.

The lower fine-tuning learning rate protects useful pretrained features from being overwritten by large updates.

### Callbacks

The image pipeline uses:

- `EarlyStopping` on validation accuracy with patience 2.
- `restore_best_weights=True`.
- `ReduceLROnPlateau` on validation loss.

Early stopping prevents unnecessary training after validation performance stops improving. Reducing the learning rate helps the model make smaller updates when progress plateaus.

### Actual image training run

The current saved model used:

| Setting | Value |
|---|---:|
| Images per class | 50 |
| Total sampled images | 200 |
| Training images | 160 |
| Validation images | 40 |
| Requested epochs | 8 |
| Effective epochs | 6, due to early stopping |
| Batch size | 16 |
| Fine-tuned layers | Final 30 layers |
| Model | ImageNet-pretrained EfficientNetB0 |
| Validation accuracy | 92.5% |

The current model is therefore trained on a reproducible sample from the actual archive, not on the entire image archive. This was chosen to keep local training practical. To use up to all 500 available images in each class:

```powershell
python train.py --image-sample-per-class 500 --image-epochs 8 --batch-size 16
```

That configuration takes substantially longer on CPU.

### Image evaluation

The saved validation report includes:

- Overall accuracy.
- Per-class precision.
- Per-class recall.
- Per-class F1 score.
- Support count.
- Confusion matrix.

The current report shows strong performance on the sampled colon classes and some confusion between `lung_aca` and `lung_n`. This is why per-class metrics and the confusion matrix are important instead of reporting accuracy alone.

### Image prediction flow

1. The user chooses a JPG or PNG.
2. The browser previews the image.
3. The browser sends multipart form data to `/api/predict/image`.
4. Flask checks that a compatible trained model exists.
5. The upload stream is converted to `BytesIO`.
6. The image is resized to `224 x 224`.
7. EfficientNet returns four softmax probabilities.
8. The highest probability becomes the predicted class.
9. The API returns the class and all probabilities.
10. The result is written to `logs/predictions_log.json`.

## 8. Flask API Design

The API is implemented in `server.py`. It serves both the static frontend and JSON endpoints from the same local origin, so normal local use does not need cross-origin configuration.

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the browser dashboard |
| `GET` | `/api/health` | Reports API and saved-model readiness |
| `GET` | `/api/dataset` | Reports local paths and class counts |
| `GET` | `/api/metrics` | Returns saved tabular and image metrics |
| `GET` | `/api/logs/predictions` | Returns recent prediction records |
| `POST` | `/api/train/tabular` | Trains and saves the tabular model |
| `POST` | `/api/train/image` | Trains and saves the image CNN |
| `POST` | `/api/predict/tabular` | Predicts from structured inputs |
| `POST` | `/api/predict/image` | Predicts from an uploaded image |

### API error handling

The API returns JSON errors with HTTP status codes instead of exposing raw Python tracebacks to the browser. It handles missing model artifacts, missing tabular fields, missing image uploads, invalid image input, missing dataset folders, and uploads larger than the 10 MB limit.

### Why Flask?

Flask is small, familiar, and sufficient for a local synchronous API. It makes the separation between model logic and web logic clear. The same architecture could later be moved to FastAPI or a production WSGI server.

## 9. Frontend Design

The main frontend is plain HTML, CSS, and JavaScript. This was intentional because this is a local ML application and does not need a large frontend build system.

### Frontend features

- Tabular model tab.
- Image CNN tab.
- Local API status indicator.
- Dataset row and class counts.
- Exact image archive path display.
- Training controls.
- Tabular input form.
- Image file chooser and preview.
- Prediction result cards.
- Probability bars for all four image classes.
- Responsive layout.
- Research and medical-use disclaimer.

### Why plain JavaScript instead of React?

Advantages:

- No Node.js build step.
- Fewer dependencies.
- Easy to run on Windows.
- Direct, understandable API calls.
- Small debugging surface.

Disadvantages:

- Less component reuse.
- More manual DOM updates.
- No formal state-management layer.
- A larger product would benefit from React, Vue, or another structured framework.

For this project size, plain JavaScript keeps setup and deployment simple.

## 10. Storage and Reproducibility

### Saved artifacts

```text
artifacts/
├── lc25000_lung_model.keras
├── lc25000_lung_model.json
├── tabular_lung_cancer_model.joblib
└── lung_cancer_model.joblib
```

The Keras file stores the image CNN. The image JSON stores class names, training configuration, metrics, image source, and timestamp. The joblib file stores the complete tabular scikit-learn pipeline and feature metadata.

### Logs

```text
logs/
├── training_logs.json
├── validation_logs.json
└── predictions_log.json
```

`training_logs.json` stores epoch-level loss and accuracy. `validation_logs.json` stores final metrics and the confusion matrix. `predictions_log.json` stores recent image and tabular predictions with UTC timestamps.

### Reproducibility controls

- Fixed seed for tabular splitting.
- Fixed seed for image sampling.
- Explicit image class order.
- Explicit tabular feature order.
- Saved preprocessing inside the model pipeline.
- Saved model metadata.
- Saved training and validation results.

TensorFlow may still produce tiny numerical differences on different hardware because CPU instructions and floating-point operations can vary.

## 11. Why These Tools Were Chosen

### Python

Python has mature libraries for data processing, classical ML, deep learning, image loading, APIs, and local automation.

### pandas

Used for CSV loading, column normalization, numeric conversion, missing-value handling, and prediction DataFrames.

### scikit-learn

Used for tabular preprocessing, logistic regression, train/test splitting, and metrics.

### TensorFlow/Keras

Used for image datasets, transfer learning, augmentation, callbacks, model saving, and inference.

### Pillow

Used to validate and convert sampled images into RGB JPEG files.

### Flask

Used to expose the local REST API and serve the static dashboard.

### Joblib and Keras model files

Joblib is appropriate for scikit-learn pipelines. Keras format is appropriate for saving and loading the TensorFlow image model.

## 12. Alternatives Considered

### Tabular alternatives

#### Random Forest

Pros:

- Captures nonlinear feature interactions.
- Works well with mixed tabular features.
- Does not depend strongly on feature scaling.

Cons:

- Less directly interpretable than logistic regression.
- Can overfit a small dataset.
- Probabilities may need calibration.

#### XGBoost or LightGBM

Pros:

- Often powerful on tabular data.
- Models nonlinear relationships well.

Cons:

- Adds dependency and tuning complexity.
- May be excessive for 309 rows.
- More difficult to explain as a first baseline.

#### Neural network for tabular data

Pros:

- Can model nonlinear relationships.
- Fits a deep-learning workflow.

Cons:

- Unnecessary for this small dataset.
- More sensitive to tuning and overfitting.
- Less interpretable.

### Image alternatives

#### CNN from scratch

Pros:

- Full architectural control.
- No dependence on ImageNet features.

Cons:

- Needs more data and compute.
- More likely to overfit a short sample.
- Takes longer to reach a useful baseline.

#### MobileNetV2

Pros:

- Very lightweight.
- Fast inference.
- Good for low-resource devices.

Cons:

- May have lower feature capacity than EfficientNetB0.

The earlier quick demo used MobileNetV2, but the final implementation uses EfficientNetB0 because the project plan requested a stronger pretrained CNN baseline.

#### ResNet50

Pros:

- Strong and widely understood architecture.
- Good transfer-learning baseline.

Cons:

- Heavier than EfficientNetB0.
- More memory and CPU cost for this local application.

#### Larger EfficientNet variants

Pros:

- More capacity.
- Potentially stronger with enough data.

Cons:

- Longer training.
- More memory usage.
- Larger model artifact.
- Less suitable for quick local testing.

### Frontend alternatives

#### Streamlit

Pros:

- Very fast to build.
- Excellent for data-science prototypes.
- Python-only workflow.

Cons:

- Rerun-based interaction can be less predictable.
- Less separation between UI and API.
- Less representative of a conventional web application.

Streamlit remains in `app.py` as a fallback demo, but Flask plus static JavaScript is the main application architecture.

#### FastAPI

Pros:

- Strong request validation.
- Automatic OpenAPI documentation.
- Good modern API ecosystem.

Cons:

- More setup than needed for this local synchronous workload.

Flask was sufficient for the current scope.

## 13. Project Advantages

### Technical advantages

- Supports two common healthcare data modalities.
- Uses a suitable model family for each modality.
- Uses transfer learning to reduce image training cost.
- Keeps preprocessing in reusable pipelines.
- Saves models for later inference.
- Has explicit API boundaries.
- Provides a simple upload workflow.
- Includes metrics and prediction logging.
- Uses deterministic sampling and class ordering.
- Runs locally without cloud deployment.
- Keeps the large dataset archive out of GitHub.

### Interview advantages

The project demonstrates data cleaning, feature preprocessing, class-imbalance awareness, classical ML, deep learning, transfer learning, augmentation, evaluation, model persistence, REST API development, frontend integration, error handling, reproducibility, and responsible communication of medical limitations.

## 14. Project Disadvantages and Limitations

### Dataset limitations

- The saved image model uses 50 images per class for the short local run, not all 1,999 training images.
- The image validation set contains only 40 images in the saved run.
- The separate test folder is discovered and counted but is not currently used as the final evaluation set.
- The tabular holdout contains only 10 rows.
- Metrics can change significantly with a different split or larger sample.
- The dataset may not represent real clinical populations.

### Modeling limitations

- The four image classes combine colon and lung categories, so this is not a pure lung-only model.
- A softmax probability is not automatically a calibrated clinical risk score.
- Histopathology images can contain scanner, staining, or preparation patterns that do not generalize.
- No external validation dataset is used.
- No explainability method such as Grad-CAM is included yet.
- No confidence threshold or abstention policy is implemented.
- The tabular model predicts a survey label, not confirmed pathology.

### Engineering limitations

- The Flask development server is for local use, not production deployment.
- The training endpoint is synchronous, so the browser waits while training runs.
- There is no authentication because the app is local-only.
- There is no database; logs are JSON files.
- Automated unit and integration tests have not yet been added.
- CPU training is slow for large image samples.

## 15. Improvements I Would Make Next

1. Evaluate the image model on the untouched `Test Set` and report test metrics separately from validation metrics.
2. Train with all available training images after the pipeline is stable.
3. Use stratified cross-validation for the tabular model instead of one very small holdout.
4. Add tests for preprocessing, API validation, model loading, and prediction responses.
5. Add Grad-CAM heatmaps to show which image regions influenced a prediction.
6. Add calibrated probabilities and an `uncertain` result for low-confidence cases.
7. Add a lung-only binary mode using only `lung_aca` and `lung_n` when that is the required scope.
8. Add background jobs so the API remains responsive during long training.
9. Add model versioning instead of overwriting one artifact.
10. Add Docker packaging or a production WSGI server for deployment.
11. Validate corrupted files, image dimensions, and class-folder completeness.
12. Add experiment tracking for hyperparameters and metrics.

## 16. Common Interview Questions and Answers

### What is the main contribution of this project?

I converted a notebook-style experiment into an end-to-end local ML application. It includes data ingestion, preprocessing, training, evaluation, saved models, a REST API, a browser frontend, image upload, and logging.

### Why logistic regression for tabular data?

The dataset is small and mostly binary. Logistic regression is fast, interpretable, produces probabilities, and provides a strong baseline before trying more complex models. Balanced class weights address the skewed target distribution.

### Why EfficientNetB0 for images?

It is a pretrained CNN with a good accuracy-to-compute balance. It is small enough for local CPU experimentation but has stronger transfer-learning capacity than a minimal CNN. I removed its ImageNet head and attached a four-class LC25000 head.

### What does `include_top=False` mean?

It removes the original ImageNet classifier. The convolutional feature extractor remains, and I add a new global pooling, dropout, and four-class softmax head.

### Why freeze the base first?

The new head must learn the task-specific mapping without immediately changing every pretrained feature. Freezing first makes training stable. Fine-tuning selected final layers then allows adaptation to histopathology.

### Why use a lower learning rate for fine-tuning?

The pretrained features are already useful. A large learning rate could destroy them. The lower `1e-5` learning rate makes smaller updates and reduces catastrophic forgetting.

### Why keep batch normalization frozen?

Small datasets and small batches can create unstable batch-normalization statistics. Keeping those layers frozen makes fine-tuning more stable.

### How did you address overfitting?

I used augmentation, dropout, a frozen base, limited fine-tuning, early stopping, and learning-rate reduction. Overfitting remains a limitation because the saved validation set is small.

### What is validation versus test data?

Validation data is used during development to monitor performance and stop training. A final test set should remain untouched until the end. This pipeline uses a split from the training folder for validation and reports the separate test-folder counts, but final test evaluation should be added before making stronger claims.

### Can you call 92.5% medical-grade accuracy?

No. It is 92.5% on a small 40-image validation subset from a sampled local run. It is a development metric, not clinical evidence. A real medical system needs representative data, external validation, calibration, explainability, clinician review, and regulatory controls.

### Why are colon classes in a lung-cancer project?

The supplied LC25000 archive contains both colon and lung folders. The current classifier faithfully predicts the four actual folder labels. If the product requirement is strictly lung-only prediction, I would train a separate binary or lung-only classifier using `lung_aca` and `lung_n` and make that scope explicit in the UI.

### How does the browser communicate with the model?

The frontend sends JSON for tabular inputs and multipart form data for images. Flask validates the request, calls shared pipeline functions, and returns JSON containing predictions, probabilities, or errors.

### How did you make training reproducible?

I fixed random seeds for sampling and splits, defined class order explicitly, saved preprocessing inside the model pipeline, and wrote model metadata and logs.

### What happens if a model is not trained?

The health endpoint reports model readiness. A prediction endpoint returns a clear HTTP 409 error instructing the user to train the relevant model first. The frontend displays the readiness state.

### Why keep the image archive out of GitHub?

The archive is large local data and is not required to clone the code. The application expects it at a known local path and creates a small training sample. Excluding it keeps the repository manageable and avoids unnecessary redistribution of dataset files.

## 17. Honest Limitations Answer

> The main limitation is that this is a local research prototype, not a clinically validated tool. The saved image model used a 50-per-class sample and a 40-image validation set, and the tabular model used a 10-row holdout. The separate test folder is available but should be used for final untouched evaluation. The labels also include both colon and lung classes, so I would build a lung-only mode for a narrower clinical claim. My next steps would be full-dataset training, independent test evaluation, cross-validation, calibration, Grad-CAM explanations, uncertainty thresholds, and automated tests.

## 18. Recommended Interview Demonstration

1. Start the application with `start_lung_app.bat` or `python server.py`.
2. Open `http://127.0.0.1:5000`.
3. Show that the dashboard reports the local archive path.
4. Show the four detected image classes.
5. Show the saved EfficientNetB0 validation metrics.
6. Open the image tab and upload a sample image.
7. Show the predicted class and all four probabilities.
8. Explain ImageNet pretraining versus LC25000 fine-tuning.
9. Open the tabular tab and show the questionnaire features.
10. Explain the preprocessing pipeline and balanced logistic regression.
11. Show that models and logs are stored under `artifacts/` and `logs/`.
12. Finish by explaining limitations and future improvements.

## 19. Important Terminology

| Term | Meaning in this project |
|---|---|
| Transfer learning | Reusing pretrained CNN features and adapting them to LC25000 classes |
| Fine-tuning | Unfreezing selected pretrained layers and training them with a small learning rate |
| Feature extractor | The convolutional part of EfficientNetB0 before the new classifier head |
| Softmax | Converts four output scores into probabilities that sum to one |
| Stratified split | A split that attempts to preserve class proportions |
| Early stopping | Stops training when monitored validation performance stops improving |
| Precision | Of predicted positives, the proportion that are correct |
| Recall | Of actual positives, the proportion detected by the model |
| F1 score | Harmonic mean of precision and recall |
| Confusion matrix | Table showing actual versus predicted class counts |
| Artifact | A saved model or metadata file used after training |

## 20. Final Project Positioning

The strongest accurate way to present this project is:

> This is an end-to-end local machine-learning prototype that demonstrates the complete lifecycle from data loading and preprocessing to model training, evaluation, persistence, API design, frontend integration, and responsible limitation reporting. It intentionally uses an interpretable tabular baseline and a pretrained CNN for image data because the two modalities require different representations, and transfer learning is more appropriate than training a large image model from scratch on limited local compute.

That description is technically defensible, honest about the current validation setup, and gives a clear path for future improvement.
