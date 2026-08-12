
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="Road Traffic Accident Detection",
    page_icon="🚗",
    layout="wide"
)


# Load trained YOLOv8 model
MODEL_PATH = Path(__file__).parent / "best.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


# Class names
CLASS_NAMES = {
    0: "No Accident",
    1: "Minor Accident",
    2: "Moderate Accident",
    3: "Severe Accident",
    4: "Totaled Vehicle",
}


# Application title
st.title("🚗 Road Traffic Accident Detection")

st.write(
    "YOLOv8-based object detection system for detecting "
    "road traffic accidents and classifying accident severity."
)

st.divider()


# Upload image
uploaded_file = st.file_uploader(
    "Upload a road traffic image",
    type=["jpg", "jpeg", "png"]
)


# Confidence threshold
confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)


# Detection
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

    # Run YOLOv8
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Create annotated image
    annotated_image = result.plot()

    with col2:
        st.subheader("Detection Result")
        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

    st.divider()

    st.subheader("Detection Details")

    if result.boxes is not None and len(result.boxes) > 0:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            class_name = CLASS_NAMES.get(
                class_id,
                f"Class {class_id}"
            )

            st.write(
                f"**{class_name}** — "
                f"Confidence: **{confidence_score:.2%}**"
            )

    else:
        st.info("No objects detected.")
