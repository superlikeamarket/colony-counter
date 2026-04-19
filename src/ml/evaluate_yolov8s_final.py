from pathlib import Path
import math
import pandas as pd
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

MODEL_PATH = Path("local/runs/detect/yolov8s_1024_cpu_resume/weights/best.pt")
IMAGE_DIR = Path("local/data/dataset_yolo_counting/images/val")
LABEL_DIR = Path("local/data/dataset_yolo_counting/labels/val")

OUTPUT_DIR = Path("local/outputs/ml_evaluation")
PREDICTIONS_CSV = OUTPUT_DIR / "yolov8s_final_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "yolov8s_final_metrics.csv"
TNTC_CONFUSION_CSV = OUTPUT_DIR / "yolov8s_tntc_confusion_matrix.csv"

# Best settings from your grid search
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.3
IMAGE_SIZE = 1024
MAX_DET = 1000

# Lab rule
TNTC_THRESHOLD = 300

# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# HELPERS
# =========================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def count_true_boxes(label_path: Path) -> int:
    """
    Count valid YOLO rows in a label file.
    Each valid row = one colony.
    """
    if not label_path.exists():
        return 0

    count = 0
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                count += 1
    return count


def count_predicted_boxes(result) -> int:
    """
    Count predicted boxes from one Ultralytics result object.
    """
    if result.boxes is None:
        return 0
    return len(result.boxes)


def classify_tntc(count: int, threshold: int = TNTC_THRESHOLD) -> str:
    """
    Lab-style classification:
    <= threshold -> countable
    > threshold  -> TNTC
    """
    return "TNTC" if count > threshold else "countable"


def compute_regression_metrics(df: pd.DataFrame) -> dict:
    """
    Compute count-regression metrics.
    """
    df = df.copy()
    df["error"] = df["pred_count"] - df["true_count"]
    df["abs_error"] = df["error"].abs()

    nonzero_mask = df["true_count"] != 0
    df["relative_error"] = pd.NA
    df.loc[nonzero_mask, "relative_error"] = (
        df.loc[nonzero_mask, "abs_error"] / df.loc[nonzero_mask, "true_count"]
    )

    metrics = {
        "num_images": len(df),
        "mae": df["abs_error"].mean(),
        "rmse": math.sqrt((df["error"] ** 2).mean()),
        "median_ae": df["abs_error"].median(),
        "mean_relative_error": df["relative_error"].dropna().mean(),
    }
    return metrics


def compute_tntc_metrics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Compute lab-style TNTC metrics and confusion matrix.
    """
    df = df.copy()

    df["true_class"] = df["true_count"].apply(classify_tntc)
    df["pred_class"] = df["pred_count"].apply(classify_tntc)
    df["correct_classification"] = df["true_class"] == df["pred_class"]

    accuracy = df["correct_classification"].mean()

    # confusion matrix counts
    true_countable_pred_countable = len(
        df[(df["true_class"] == "countable") & (df["pred_class"] == "countable")]
    )
    true_countable_pred_tntc = len(
        df[(df["true_class"] == "countable") & (df["pred_class"] == "TNTC")]
    )
    true_tntc_pred_countable = len(
        df[(df["true_class"] == "TNTC") & (df["pred_class"] == "countable")]
    )
    true_tntc_pred_tntc = len(
        df[(df["true_class"] == "TNTC") & (df["pred_class"] == "TNTC")]
    )

    confusion_df = pd.DataFrame(
        [
            {
                "true_label": "countable",
                "pred_countable": true_countable_pred_countable,
                "pred_tntc": true_countable_pred_tntc,
            },
            {
                "true_label": "TNTC",
                "pred_countable": true_tntc_pred_countable,
                "pred_tntc": true_tntc_pred_tntc,
            },
        ]
    )

    # TNTC as positive class
    tp = true_tntc_pred_tntc
    fp = true_countable_pred_tntc
    fn = true_tntc_pred_countable

    precision_tntc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_tntc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "tntc_threshold": TNTC_THRESHOLD,
        "tntc_accuracy": accuracy,
        "tntc_precision": precision_tntc,
        "tntc_recall": recall_tntc,
    }

    return metrics, confusion_df


# =========================
# MAIN
# =========================

def main() -> None:
    ensure_output_dir()

    model = YOLO(str(MODEL_PATH))
    image_paths = get_image_paths(IMAGE_DIR)

    rows = []

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            max_det=MAX_DET,
            save=False,
            verbose=False
        )

        result = results[0]
        pred_count = count_predicted_boxes(result)
        true_count = count_true_boxes(label_path)

        rows.append(
            {
                "image_name": image_path.name,
                "pred_count": pred_count,
                "true_count": true_count,
            }
        )

        print(f"{image_path.name}: pred={pred_count}, true={true_count}")

    df = pd.DataFrame(rows)

    # Add regression columns
    df["error"] = df["pred_count"] - df["true_count"]
    df["abs_error"] = df["error"].abs()

    nonzero_mask = df["true_count"] != 0
    df["relative_error"] = pd.NA
    df.loc[nonzero_mask, "relative_error"] = (
        df.loc[nonzero_mask, "abs_error"] / df.loc[nonzero_mask, "true_count"]
    )

    # Add TNTC columns
    df["true_class"] = df["true_count"].apply(classify_tntc)
    df["pred_class"] = df["pred_count"].apply(classify_tntc)
    df["correct_classification"] = df["true_class"] == df["pred_class"]

    # Save detailed per-image results
    df.to_csv(PREDICTIONS_CSV, index=False)

    # Summary metrics
    regression_metrics = compute_regression_metrics(df)
    tntc_metrics, confusion_df = compute_tntc_metrics(df)

    summary_metrics = {
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "max_det": MAX_DET,
        **regression_metrics,
        **tntc_metrics,
    }

    metrics_df = pd.DataFrame([summary_metrics])
    metrics_df.to_csv(METRICS_CSV, index=False)
    confusion_df.to_csv(TNTC_CONFUSION_CSV, index=False)

    # Print regression metrics
    print("\n=== YOLO COUNT METRICS ===")
    for key in ["num_images", "mae", "rmse", "median_ae", "mean_relative_error"]:
        print(f"{key}: {summary_metrics[key]}")

    # Print TNTC metrics
    print("\n=== YOLO TNTC METRICS ===")
    for key in ["tntc_threshold", "tntc_accuracy", "tntc_precision", "tntc_recall"]:
        print(f"{key}: {summary_metrics[key]}")

    # Print confusion matrix
    print("\n=== TNTC CONFUSION MATRIX ===")
    print(confusion_df.to_string(index=False))

    # Print worst cases
    worst = df.sort_values("abs_error", ascending=False).head(10)
    print("\n=== WORST 10 IMAGES ===")
    print(
        worst[
            [
                "image_name",
                "pred_count",
                "true_count",
                "error",
                "abs_error",
                "pred_class",
                "true_class",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved detailed predictions to: {PREDICTIONS_CSV}")
    print(f"Saved summary metrics to: {METRICS_CSV}")
    print(f"Saved TNTC confusion matrix to: {TNTC_CONFUSION_CSV}")


if __name__ == "__main__":
    main()