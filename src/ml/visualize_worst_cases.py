import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

MODEL_PATH = "local/runs/detect/train5/weights/best.pt"

IMAGE_DIR = Path("local/data/dataset_yolo_counting/images/test")
OUTPUT_DIR = Path("local/outputs/ml_evaluation/test_worst_cases_clean")

CONF = 0.25
IOU = 0.5
IMGSZ = 1024
MAX_DET = 1000


def draw_centers(image, boxes):
    overlay = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(overlay, (cx, cy), 2, (255, 0, 0), 3)

    # optional: blend for softer look
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("local/outputs/ml_evaluation/yolo_final_predictions.csv")

    # df = df[
    #     (df["conf_threshold"] == CONF) &
    #     (df["iou_threshold"] == IOU) &
    #     (df["image_size"] == IMGSZ)
    # ]

    df["error"] = df["pred_count"] - df["true_count"]
    df["abs_error"] = df["error"].abs()

    worst = df.sort_values("abs_error", ascending=False).head(10)

    model = YOLO(MODEL_PATH)

    for _, row in worst.iterrows():
        image_name = row["image_name"]
        image_path = IMAGE_DIR / image_name

        # run inference (NO saving)
        results = model.predict(
            source=str(image_path),
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            save=False,
            verbose=False
        )[0]

        image = cv2.imread(str(image_path))

        clean = draw_centers(image, results.boxes)

        # add small title (optional but useful)
        text = f"pred={row['pred_count']} true={row['true_count']}"
        cv2.putText(
            clean,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        cv2.imwrite(
            str(OUTPUT_DIR / image_name),
            clean
        )

        print(f"Saved clean visualization: {image_name}")


if __name__ == "__main__":
    main()