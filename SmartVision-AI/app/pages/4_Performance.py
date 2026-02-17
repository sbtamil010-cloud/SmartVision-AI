import streamlit as st
import pandas as pd

st.title("📊 Model Performance Comparison")

data = {
    "Model": ["VGG16","ResNet50","MobileNetV2","EfficientNetB0"],
    "Accuracy": [0.83,0.88,0.85,0.91],
    "Inference Time (ms)": [150,100,50,80]
}

df = pd.DataFrame(data)

st.subheader("Accuracy Comparison")
st.bar_chart(df.set_index("Model")["Accuracy"])

st.subheader("Inference Speed Comparison")
st.bar_chart(df.set_index("Model")["Inference Time (ms)"])

st.dataframe(df)
