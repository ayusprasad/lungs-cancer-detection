from __future__ import annotations

from io import BytesIO

import streamlit as st

from lung_cancer_pipeline import (
    IMAGE_CLASS_NAMES,
    LOCAL_IMAGE_ROOT,
    load_image_model,
    load_tabular_dataset,
    load_tabular_model,
    predict_image,
    predict_tabular,
    train_image_model,
    train_tabular_model,
)

st.set_page_config(page_title="Lung Cancer Local Demo", layout="wide")


def _tabular_defaults() -> dict[str, int]:
    return {
        "GENDER": 1,
        "AGE": 60,
        "SMOKING": 1,
        "YELLOW_FINGERS": 1,
        "ANXIETY": 1,
        "PEER_PRESSURE": 1,
        "CHRONIC DISEASE": 1,
        "FATIGUE": 2,
        "ALLERGY": 1,
        "WHEEZING": 1,
        "ALCOHOL CONSUMING": 1,
        "COUGHING": 1,
        "SHORTNESS OF BREATH": 1,
        "SWALLOWING DIFFICULTY": 1,
        "CHEST PAIN": 1,
    }


def _render_tabular_inputs() -> dict[str, int]:
    defaults = _tabular_defaults()
    st.subheader("Patient Inputs")
    cols = st.columns(3)
    values: dict[str, int] = {}

    with cols[0]:
        values["GENDER"] = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        values["AGE"] = st.slider("Age", 20, 90, defaults["AGE"])
        values["SMOKING"] = st.selectbox("Smoking", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["YELLOW_FINGERS"] = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["ANXIETY"] = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")

    with cols[1]:
        values["PEER_PRESSURE"] = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["CHRONIC DISEASE"] = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["FATIGUE"] = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=1)
        values["ALLERGY"] = st.selectbox("Allergy", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["WHEEZING"] = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")

    with cols[2]:
        values["ALCOHOL CONSUMING"] = st.selectbox("Alcohol Consuming", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["COUGHING"] = st.selectbox("Coughing", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["SHORTNESS OF BREATH"] = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["SWALLOWING DIFFICULTY"] = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        values["CHEST PAIN"] = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")

    return values


def _render_image_inputs() -> BytesIO | None:
    st.subheader("Upload a Lung Histopathology Image")
    upload = st.file_uploader("Choose a JPEG or PNG", type=["jpg", "jpeg", "png"])
    if upload is None:
        return None
    return BytesIO(upload.read())


st.title("Lung Cancer Local Demo")
st.write("This app keeps the notebook short: a fast tabular model and a tiny LC25000 lung image model.")

tab1, tab2 = st.tabs(["Tabular CSV", "LC25000 Images"])

with tab1:
    st.subheader("Tabular Survey Model")
    try:
        df = load_tabular_dataset()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not load the tabular dataset: {error}")
        st.info("Confirm that the survey CSV is available, then reload the app.")
        st.stop()
    left, right = st.columns([1, 1])
    with left:
        st.write(f"Rows: **{len(df)}**")
        st.dataframe(df.head(8), use_container_width=True)
        sample_size = st.slider("Tabular training rows", 20, 50, 50)
        train_tabular = st.button("Train tabular model", type="primary")
    with right:
        st.info("The tabular model is the cleaned version of the original CSV path, kept short for local testing.")

    if train_tabular or "tabular_bundle" not in st.session_state:
        try:
            if train_tabular:
                with st.spinner("Training tabular model on a small sample..."):
                    st.session_state.tabular_bundle = train_tabular_model(sample_size=sample_size)
            else:
                loaded = load_tabular_model()
                if loaded is not None:
                    st.session_state.tabular_bundle = loaded
        except (EOFError, OSError, RuntimeError, ValueError) as error:
            st.session_state.tabular_bundle = None
            st.error(f"The tabular model is unavailable: {error}")

    bundle = st.session_state.get("tabular_bundle")
    if bundle:
        metrics = bundle.get("metrics", {})
        if metrics:
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
            cols[1].metric("Precision", f"{metrics['precision']:.2%}")
            cols[2].metric("Recall", f"{metrics['recall']:.2%}")
            cols[3].metric("F1", f"{metrics['f1']:.2%}")

            st.caption(
                f"Trained on {metrics['train_rows']} rows, evaluated on {metrics['test_rows']} rows from a {metrics['sample_rows']}-row subset."
            )
        else:
            st.info("Loaded a saved tabular model. Train once to view fresh metrics and the holdout table.")

        features = _render_tabular_inputs()
        if st.button("Predict tabular risk"):
            try:
                result = predict_tabular(bundle["model"], features)
            except (KeyError, TypeError, ValueError) as error:
                st.error(f"Could not make a tabular prediction: {error}")
            else:
                if result["prediction"] == "YES":
                    st.error(f"Prediction: {result['prediction']} probability={result['probability_yes']:.2%}")
                else:
                    st.success(f"Prediction: {result['prediction']} probability={result['probability_yes']:.2%}")
        test_frame = bundle.get("test_frame")
        if test_frame is not None:
            st.dataframe(test_frame, width="stretch")
    else:
        st.info("Train the tabular model to enable predictions.")

with tab2:
    st.subheader("LC25000 Lung Image Model")
    st.write(f"Classes: {', '.join(IMAGE_CLASS_NAMES)}")
    st.caption(f"First-time training samples only a tiny subset from {LOCAL_IMAGE_ROOT} and caches it locally.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        image_sample_per_class = st.slider("Images per class", 5, 20, 10)
        image_epochs = st.slider("Image epochs", 1, 5, 3)
        train_images = st.button("Train image model", type="primary")
    with col_b:
        st.info("This uses your local LC25000 archive with 4 folders: colon_aca, colon_n, lung_aca, and lung_n.")

    if train_images or "image_bundle" not in st.session_state:
        try:
            if train_images:
                with st.spinner("Training the LC25000 lung image model..."):
                    st.session_state.image_bundle = train_image_model(
                        sample_per_class=image_sample_per_class,
                        epochs=image_epochs,
                    )
            else:
                loaded = load_image_model()
                if loaded is not None:
                    st.session_state.image_bundle = loaded
        except (EOFError, OSError, RuntimeError, ValueError) as error:
            st.session_state.image_bundle = None
            st.error(f"The image model is unavailable: {error}")

    image_bundle = st.session_state.get("image_bundle")
    if image_bundle:
        metrics = image_bundle["metrics"]
        cols = st.columns(3)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
        cols[1].metric("Epochs", str(metrics["epochs_ran"]))
        cols[2].metric("Sample / class", str(metrics["sample_per_class"]))

        image_file = _render_image_inputs()
        if image_file is not None:
            st.image(image_file, caption="Uploaded image", width="stretch")
            if st.button("Predict image class"):
                try:
                    image_file.seek(0)
                    result = predict_image(image_bundle["model"], image_file, image_bundle["class_names"])
                except (OSError, TypeError, ValueError) as error:
                    st.error(f"Could not process this image: {error}")
                else:
                    st.write(f"Prediction: **{result['prediction']}**")
                    st.bar_chart(result["probabilities"])
    else:
        st.info("Train the image model to enable LC25000 predictions.")
