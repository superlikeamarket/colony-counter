from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.app.preprocessing import preprocess_for_model
from src.ml.inference import draw_predictions, run_inference_on_image


def predict_colonies(
    image: np.ndarray,
    model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Full colony-counting pipeline for app use.

    Steps:
    1. Validate input image
    2. Preprocess image (plate detect -> crop -> resize -> mask)
    3. Run YOLO inference
    4. Draw visualization
    5. Classify TNTC

    Args:
        image: Input image as a NumPy array. Can be RGB or BGR.
        model: Loaded ultralytics YOLO model.
        config: Inference config dict. Expected keys:
            - conf
            - iou
            - imgsz
            - max_det (optional)
            - tntc_threshold (optional, default 300)

    Returns:
        Dict with:
            success: bool
            error: str | None
            original_image: np.ndarray | None
            processed_image: np.ndarray | None
            visualization: np.ndarray | None
            pred_count: int | None
            pred_label: str | None
            is_tntc: bool | None
            boxes: np.ndarray | list
            centers: list[tuple[int, int]]
            preprocess_info: dict
    """
    if image is None:
        return _error_result("Input image is None")

    if not isinstance(image, np.ndarray):
        return _error_result("Input image must be a NumPy array")

    if image.ndim != 3 or image.shape[2] != 3:
        return _error_result("Input image must have shape (H, W, 3)")

    original_bgr = _to_bgr(image)

    processed_image, preprocess_info = preprocess_for_model(original_bgr)

    if processed_image is None:
        error_message = preprocess_info.get("error", "Preprocessing failed")
        return _error_result(
            error_message,
            original_image=original_bgr,
            preprocess_info=preprocess_info,
        )

    pred_count, boxes, inference_image = run_inference_on_image(
        model=model,
        image=processed_image,
        config=config,
    )

    if inference_image is None:
        inference_image = processed_image.copy()

    visualization = draw_predictions(
        image=inference_image,
        boxes=boxes,
        pred_count=pred_count,
    )

    centers = _boxes_to_centers(boxes)

    tntc_threshold = int(config.get("tntc_threshold", 300))
    is_tntc = pred_count >= tntc_threshold
    pred_label = "TNTC" if is_tntc else "countable"

    return {
        "success": True,
        "error": None,
        "original_image": original_bgr,
        "processed_image": processed_image,
        "visualization": visualization,
        "pred_count": int(pred_count),
        "pred_label": pred_label,
        "is_tntc": is_tntc,
        "boxes": boxes if boxes is not None else [],
        "centers": centers,
        "preprocess_info": preprocess_info,
    }


def _boxes_to_centers(boxes: np.ndarray | list) -> list[tuple[int, int]]:
    """Convert xyxy boxes to integer center points."""
    if boxes is None or len(boxes) == 0:
        return []

    centers: list[tuple[int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        centers.append((cx, cy))

    return centers


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Best-effort conversion to BGR for OpenCV-based preprocessing/inference.
    Assumes PIL/Streamlit uploads are usually RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _error_result(
    message: str,
    original_image: np.ndarray | None = None,
    preprocess_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "error": message,
        "original_image": original_image,
        "processed_image": None,
        "visualization": None,
        "pred_count": None,
        "pred_label": None,
        "is_tntc": None,
        "boxes": [],
        "centers": [],
        "preprocess_info": preprocess_info or {},
    }