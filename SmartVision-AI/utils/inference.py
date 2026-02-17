import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO


def load_classification_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_classification(model, image, class_names):
    image = cv2.resize(image, (224,224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    return predicted_class, confidence, preds


def load_yolo_model(model_path):
    return YOLO(model_path)


def run_yolo_inference(model, image_path, conf=0.5):
    results = model(image_path, conf=conf)
    return results
