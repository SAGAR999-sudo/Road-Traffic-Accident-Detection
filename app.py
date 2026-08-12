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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 🎨 Theme")

theme = st.sidebar.selectbox(
    "Choose theme",
    ["Day Mode", "Dark Mode"],
    index=0
)


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

st.sidebar.markdown("### ℹ️ Model")

st.sidebar.write("YOLOv8n")
st.sidebar.write("5 detection classes")


# ============================================================
# THEME COLORS
# ============================================================

if theme == "Day Mode":

    background_color = "#FFFFFF"
    text_color = "#111111"
    secondary_text = "#555555"
    panel_background = "#F8F9FA"
    border_color = "#555555"
    accent_color = "#2563EB"

else:

    background_color = "#0E1117"
    text_color = "#FFFFFF"
    secondary_text = "#BBBBBB"
    panel_background = "#1B1F27"
    border_color = "#888888"
    accent_color = "#60A5FA"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ======================================================
       MAIN APPLICATION
    ====================================================== */

    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}


    /* ======================================================
       TEXT
    ====================================================== */

    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
    }}

    p, span, label {{
        color: {text_color};
    }}


    /* ======================================================
       MAIN PANELS
    ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 2px solid {border_color} !important;
        border-radius: 12px !important;
        background-color: {panel_background} !important;
        padding: 12px !important;
    }}


    /* ======================================================
       PANEL HOVER
    ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: {accent_color} !important;
    }}


    /* ======================================================
       SIDEBAR
    ====================================================== */

    [data-testid="stSidebar"] {{
        background-color: {panel_background};
    }}


    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {text_color} !important;
    }}


    /* ======================================================
       DETECTION BOX
    ====================================================== */

    .detection-box {{
        border: 2px solid {border_color};
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        background-color: {panel_background};
    }}


    /* ======================================================
       INFO BOX
    ====================================================== */

    .info-box {{
        border: 2px solid {border_color};
        border-radius: 10px;
        padding: 15px;
        background-color: {panel_background};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "best.pt"


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

col1, col2 = st.columns(2)


with col1:

    with st.container(border=True):

        st.subheader("📤 Input Image")

        uploaded_file = st.file_uploader(
            "Upload a road traffic image",
            type=["jpg", "jpeg", "png"]
        )


with col2:

    with st.container(border=True):

        st.subheader("📊 Model Detection Output")

        if uploaded_file is None:

            st.info(
                "Upload an image in the left panel "
                "to trigger inference."
            )


# ============================================================
# DETECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")


    # --------------------------------------------------------
    # YOLO PREDICTION
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # ANNOTATED IMAGE
    # --------------------------------------------------------

    annotated_image = result.plot()


    # --------------------------------------------------------
    # DISPLAY IMAGES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        with st.container(border=True):

            st.subheader("📤 Input Image")

            st.image(
                image,
                use_container_width=True
            )


    with col2:

        with st.container(border=True):

            st.subheader("📊 Model Detection Output")

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


    if result.boxes is not None and len(result.boxes) > 0:

        detections = []


        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence_score = float(box.conf[0])

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

        for detection in detections:

            st.markdown(
                f"""
                <div class="detection-box">

                <strong>🚨 {detection["class"]}</strong>

                <br><br>

                Confidence:
                <strong>
                {detection["confidence"]:.2%}
                </strong>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader("📋 Detection Summary")

        st.write(
            f"**Total objects detected:** "
            f"{len(detections)}"
        )

        detected_classes = [
            d["class"]
            for d in detections
        ]

        st.write(
            "**Detected classes:** "
            + ", ".join(detected_classes)
        )


    else:

        st.warning(
            "⚠️ No objects detected above the selected "
            "confidence threshold."
        )