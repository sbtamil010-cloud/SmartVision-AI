import cv2
import numpy as np
import os


def preprocess_image_for_classification(image, target_size=(224, 224)):
    """
    Resize and normalize image for CNN classification.
    """
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def crop_object(image, bbox):
    """
    Crop bounding box region from image.
    bbox format: [x, y, w, h]
    """
    x, y, w, h = bbox
    crop = image[int(y):int(y+h), int(x):int(x+w)]
    return crop


def coco_to_yolo_format(bbox, img_width, img_height):
    """
    Convert COCO bbox [x,y,w,h] to YOLO format
    """
    x, y, w, h = bbox

    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    w = w / img_width
    h = h / img_height

    return x_center, y_center, w, h


def save_yolo_annotation(file_path, class_id, bbox_yolo):
    """
    Save YOLO formatted annotation to .txt file
    """
    with open(file_path, "a") as f:
        f.write(f"{class_id} {' '.join(map(str, bbox_yolo))}\n")
