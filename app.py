from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Traffic Accident Detection",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 🎯 Detection")

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.caption(
    f"Current threshold: {confidence:.2f}"
)

st.sidebar.divider()

st.sidebar.markdown("### 🤖 Model")

st.sidebar.write("YOLOv8n")
st.sidebar.write("Custom trained model")
st.sidebar.write("5 detection classes")


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = Path(__file__).parent / "best.pt"


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"Model file not found: {MODEL_PATH}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "No Accident",
    1: "Minor Accident",
    2: "Moderate Accident",
    3: "Severe Accident",
    4: "Totaled Vehicle",
}


# ============================================================
# HEADER
# ============================================================

st.title("🚗 Road Traffic Accident Detection")

st.write(
    "YOLOv8-based object detection system for detecting "
    "road traffic accidents and classifying accident severity."
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a road traffic image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# DETECTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    with st.spinner("Detecting accidents..."):

        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            verbose=False
        )


    result = results[0]


    # --------------------------------------------------------
    # CREATE ANNOTATED IMAGE
    # --------------------------------------------------------

    annotated_image = result.plot()


    # ========================================================
    # DISPLAY IMAGES
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader("📷 Original Image")

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # DETECTION RESULT
    # --------------------------------------------------------

    with col2:

        st.subheader("📊 Detection Result")

        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )


    # ========================================================
    # DETECTION DETAILS
    # ========================================================

    st.divider()

    st.subheader("🔍 Detection Details")


    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        detections = []


        # ----------------------------------------------------
        # EXTRACT DETECTIONS
        # ----------------------------------------------------

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence_score = float(
                box.conf[0]
            )

            class_name = CLASS_NAMES.get(
                class_id,
                f"Class {class_id}"
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence_score
                }
            )


        # ----------------------------------------------------
        # DISPLAY DETECTIONS
        # ----------------------------------------------------

        for index, detection in enumerate(
            detections,
            start=1
        ):

            class_name = detection["class"]

            confidence_score = detection["confidence"]

            st.write(
                f"**{index}. {class_name}** — "
                f"Confidence: **{confidence_score:.2%}**"
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.divider()

        st.subheader("📋 Detection Summary")

        st.write(
            f"**Total objects detected:** "
            f"{len(detections)}"
        )

        detected_classes = [
            detection["class"]
            for detection in detections
        ]

        st.write(
            "**Detected classes:** "
            + ", ".join(detected_classes)
        )


    # ========================================================
    # NO DETECTION
    # ========================================================

    else:

        st.info(
            "No objects detected above the selected "
            "confidence threshold."
        )

else:

    st.info(
        "Upload an image above to start accident detection."
    )