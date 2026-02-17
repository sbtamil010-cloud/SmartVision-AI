import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("🎯 Object Detection - YOLOv8")

MODEL_PATH = "models/detection/best.pt"

@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

model = load_yolo()

confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        results = model(tmp.name, conf=confidence)

    result_image = results[0].plot()

    st.image(result_image, caption="Detection Result", use_column_width=True)
