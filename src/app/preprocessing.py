import cv2
from src.preprocessing.validation import (
    detect_plate,
    crop_plate,
    resize_plate,
    mask_plate,
)
from src.config import ValidationConfig


def preprocess_for_model(image):
    config = ValidationConfig()

    plate_detected, circle = detect_plate(image)

    if not plate_detected:
        return None, {"error": "No plate detected"}

    cropped, cropped_circle, _ = crop_plate(image, circle, config)

    resized, resized_circle = resize_plate(cropped, cropped_circle, config)

    masked = mask_plate(resized, resized_circle)

    return masked, {
        "plate_detected": True,
        "circle": circle,
    }