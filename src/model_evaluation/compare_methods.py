import pandas as pd
import math


def compute_metrics(df):
    df = df.copy()
    df["error"] = df["pred_count"] - df["true_count"]
    df["abs_error"] = df["error"].abs()

    mae = df["abs_error"].mean()
    rmse = math.sqrt((df["error"] ** 2).mean())
    median_ae = df["abs_error"].median()

    rel = df[df["true_count"] > 0]["abs_error"] / df[df["true_count"] > 0]["true_count"]
    mean_relative_error = rel.mean()

    return mae, rmse, median_ae, mean_relative_error


# Load both methods
yolo = pd.read_csv("local/outputs/ml_evaluation/yolo_grid_predictions.csv")
contour = pd.read_csv("local/outputs/ml_evaluation/contour_predictions.csv")

# YOLO CSV contains multiple thresholds, filter best one:
yolo = yolo[
    (yolo["conf_threshold"] == 0.25) &
    (yolo["iou_threshold"] == 0.5) &
    (yolo["image_size"] == 1024)
]

# Compute metrics
yolo_metrics = compute_metrics(yolo)
contour_metrics = compute_metrics(contour)

# Build comparison table
comparison = pd.DataFrame([
    {
        "Method": "Contour",
        "MAE": contour_metrics[0],
        "RMSE": contour_metrics[1],
        "Median AE": contour_metrics[2],
        "Rel Error": contour_metrics[3],
    },
    {
        "Method": "YOLO (optimized)",
        "MAE": yolo_metrics[0],
        "RMSE": yolo_metrics[1],
        "Median AE": yolo_metrics[2],
        "Rel Error": yolo_metrics[3],
    }
])

print("\n=== FINAL COMPARISON ===")
print(comparison)

comparison.to_csv("local/outputs/ml_evaluation/final_comparison.csv", index=False)