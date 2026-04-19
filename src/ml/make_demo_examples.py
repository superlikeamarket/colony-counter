from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

MODEL_PATH = Path("local/runs/detect/train5/weights/best.pt")
PREDICTIONS_CSV = Path("local/outputs/ml_evaluation/yolo_test_predictions.csv")
IMAGE_DIR = Path("local/data/dataset_yolo_counting/images/test")
OUTPUT_DIR = Path("assets/example_outputs")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
IMAGE_SIZE = 1024
MAX_DET = 1000

DOT_RADIUS = 3
DOT_COLOR = (255, 0, 0)   # blue in OpenCV BGR
DOT_THICKNESS = -1        # filled circle

TEXT_COLOR = (0, 0, 0)
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2


# =========================
# HELPERS
# =========================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["pred_count"] - df["true_count"]
    df["abs_error"] = df["error"].abs()
    return df


def pick_best_by_range(df: pd.DataFrame, low: int, high: int | None = None) -> pd.Series | None:
    """
    Pick the image with the lowest absolute error in a count range.
    If high is None, select true_count >= low.
    """
    if high is None:
        subset = df[df["true_count"] >= low].copy()
    else:
        subset = df[(df["true_count"] >= low) & (df["true_count"] < high)].copy()

    if subset.empty:
        return None

    subset = subset.sort_values(["abs_error", "true_count"], ascending=[True, True])
    return subset.iloc[0]


def pick_worst_undercount(df: pd.DataFrame) -> pd.Series | None:
    subset = df[df["error"] < 0].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("error", ascending=True)  # most negative first
    return subset.iloc[0]


def pick_worst_overcount(df: pd.DataFrame) -> pd.Series | None:
    subset = df[df["error"] > 0].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("error", ascending=False)  # most positive first
    return subset.iloc[0]


def select_demo_examples(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """
    Return up to 6 representative examples.
    """
    chosen = []
    used_names = set()

    candidates = [
        ("best_sparse", pick_best_by_range(df, 0, 25)),
        ("best_medium", pick_best_by_range(df, 50, 150)),
        ("best_dense_countable", pick_best_by_range(df, 150, 300)),
        ("best_tntc", pick_best_by_range(df, 300, None)),
        ("worst_undercount", pick_worst_undercount(df)),
        ("worst_overcount", pick_worst_overcount(df)),
    ]

    for label, row in candidates:
        if row is None:
            continue
        image_name = row["image_name"]
        if image_name in used_names:
            continue
        chosen.append((label, row))
        used_names.add(image_name)

    return chosen


def draw_prediction_centers(image, boxes) -> any:
    output = image.copy()

    if boxes is None:
        return output

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(
            output,
            (cx, cy),
            DOT_RADIUS,
            DOT_COLOR,
            DOT_THICKNESS
        )

    return output


def add_header_text(image, title: str, subtitle: str) -> any:
    output = image.copy()

    cv2.putText(
        output,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        subtitle,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    return output


# =========================
# MAIN
# =========================

def main() -> None:
    ensure_output_dir()

    df = pd.read_csv(PREDICTIONS_CSV)
    df = add_error_columns(df)

    chosen = select_demo_examples(df)

    if not chosen:
        print("No examples selected.")
        return

    model = YOLO(str(MODEL_PATH))

    for label, row in chosen:
        image_name = row["image_name"]
        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

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
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        annotated = draw_prediction_centers(image, result.boxes)

        title = f"{label}"
        subtitle = (
            f"true={int(row['true_count'])} | "
            f"pred={int(row['pred_count'])} | "
            f"error={int(row['error'])}"
        )

        annotated = add_header_text(annotated, title, subtitle)

        out_path = OUTPUT_DIR / f"{label}_{image_name}"
        cv2.imwrite(str(out_path), annotated)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()