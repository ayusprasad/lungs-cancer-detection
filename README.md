# Lung Cancer Classification Lab

This is a local browser application for the tabular survey model and the LC25000 image model. It uses scikit-learn logistic regression for the tabular data and pretrained TensorFlow EfficientNetB0 transfer learning for the four image folders in your archive.

Dataset location:

`data/archive/lung_colon_image_set/Train and Validation Set`

Quick start:

```powershell
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:5000` in your browser. In the dashboard, train the tabular model and then train the image CNN. The default image run uses 50 images per class, 8 epochs, and fine-tunes the final EfficientNetB0 layers. Reduce those controls for a faster smoke test.

On Windows, you can also double-click `start_lung_app.bat`; it starts the local API and opens the upload dashboard automatically.

For command-line training:

```powershell
python train.py --image-sample-per-class 50 --image-epochs 8
```

Training metrics are saved in `logs/training_logs.json` and `logs/validation_logs.json`. Predictions are appended to `logs/predictions_log.json`; trained models are saved under `artifacts/`.

The older Streamlit demo and notebook remain available:

```powershell
streamlit run app.py
jupyter notebook local_lung_cancer_demo.ipynb
```

This is a research/demo tool, not a medical diagnostic system. All predictions require clinical review.
