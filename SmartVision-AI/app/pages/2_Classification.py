import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("🖼️ Image Classification")

MODEL_PATH = "models/classification/resnet50.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_NAMES = [
    "car","truck","bus","motorcycle","bicycle","airplane",
    "person","traffic light","stop sign","bench",
    "dog","cat","horse","bird","cow","elephant",
    "bottle","cup","bowl","pizza","cake",
    "chair","couch","bed","potted plant"
]

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    top5_idx = predictions.argsort()[-5:][::-1]

    st.subheader("🔝 Top 5 Predictions")

    for idx in top5_idx:
        st.write(f"{CLASS_NAMES[idx]} → {predictions[idx]*100:.2f}%")

    st.success(f"Predicted: {CLASS_NAMES[np.argmax(predictions)]}")
