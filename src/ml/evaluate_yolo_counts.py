from pathlib import Path
import math
import pandas as pd
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

MODEL_PATH = Path("local/runs/detect/train5/weights/best.pt")
IMAGE_DIR = Path("local/data/dataset_yolo_counting/images/val")
LABEL_DIR = Path("local/data/dataset_yolo_counting/labels/val")

OUTPUT_DIR = Path("local/outputs/ml_evaluation")
PREDICTIONS_CSV = OUTPUT_DIR / "yolo_count_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "yolo_count_metrics.csv"

CONF_THRESHOLDS = [0.25]
IOU_THRESHOLDS = [0.5]
IMAGE_SIZES = [1024]

MAX_DET = 1000
SAVE_PREDICTION_TXT = False
VERBOSE_PREDICT = False


# =========================
# HELPERS
# =========================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_image_paths(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def count_true_boxes(label_path: Path) -> int:
    """
    Count ground-truth boxes in a YOLO label file.
    Each non-empty valid line = one colony.
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


def evaluate_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Compute per-image errors and aggregate metrics for one parameter setting.
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
    return df, metrics


def evaluate_one_setting(
    model: YOLO,
    image_paths: list[Path],
    conf_threshold: float,
    iou_threshold: float,
    image_size: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Run YOLO counting evaluation for one (conf, iou, imgsz) setting.
    """
    rows = []

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            max_det=MAX_DET,
            save=False,
            save_txt=SAVE_PREDICTION_TXT,
            verbose=VERBOSE_PREDICT
        )

        result = results[0]
        pred_count = count_predicted_boxes(result)
        true_count = count_true_boxes(label_path)

        rows.append({
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "image_size": image_size,
            "image_name": image_path.name,
            "pred_count": pred_count,
            "true_count": true_count,
        })

    df = pd.DataFrame(rows)
    df, metrics = evaluate_predictions(df)

    metrics["conf_threshold"] = conf_threshold
    metrics["iou_threshold"] = iou_threshold
    metrics["image_size"] = image_size

    return df, metrics


# =========================
# MAIN
# =========================

def main() -> None:
    ensure_output_dir()

    model = YOLO(str(MODEL_PATH))
    image_paths = get_image_paths(IMAGE_DIR)

    all_predictions = []
    all_metrics = []

    total_runs = (
        len(CONF_THRESHOLDS) *
        len(IOU_THRESHOLDS) *
        len(IMAGE_SIZES)
    )
    run_idx = 0

    for image_size in IMAGE_SIZES:
        for iou_threshold in IOU_THRESHOLDS:
            for conf_threshold in CONF_THRESHOLDS:
                run_idx += 1
                print(
                    f"\n[{run_idx}/{total_runs}] "
                    f"Evaluating imgsz={image_size}, "
                    f"iou={iou_threshold:.2f}, "
                    f"conf={conf_threshold:.2f}"
                )

                df, metrics = evaluate_one_setting(
                    model=model,
                    image_paths=image_paths,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    image_size=image_size,
                )

                all_predictions.append(df)
                all_metrics.append(metrics)

                print(
                    f"MAE={metrics['mae']:.3f}, "
                    f"RMSE={metrics['rmse']:.3f}, "
                    f"MedianAE={metrics['median_ae']:.3f}, "
                    f"MRE={metrics['mean_relative_error']:.4f}"
                )

                worst = df.sort_values("abs_error", ascending=False).head(5)
                print("Worst 5 images:")
                print(
                    worst[
                        ["image_name", "pred_count", "true_count", "error", "abs_error"]
                    ].to_string(index=False)
                )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    metrics_df = metrics_df.sort_values(
        by=["mae", "rmse", "mean_relative_error"],
        ascending=[True, True, True]
    )

    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\n=== TOP 10 SETTINGS BY MAE ===")
    print(metrics_df.head(10).to_string(index=False))

    best = metrics_df.iloc[0]
    print("\n=== BEST SETTING ===")
    print(best.to_string())


if __name__ == "__main__":
    main()