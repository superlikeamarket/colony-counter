from pathlib import Path
import pandas as pd

from src.config import CountingConfig
from src.preprocessing.counting import run_batch


# Contour pipeline should use the masked images it was designed for
MASKED_IMAGE_DIR = Path("local/outputs/validation/masked_plates")

# YOLO validation split defines which images belong to the validation set
YOLO_VAL_IMAGE_DIR = Path("local/data/dataset_yolo_counting/images/val")
YOLO_VAL_LABEL_DIR = Path("local/data/dataset_yolo_counting/labels/val")

OUTPUT_PATH = Path("local/outputs/ml_evaluation/contour_predictions.csv")


def count_true_boxes(label_path: Path) -> int:
    """
    Count valid YOLO rows in a ground-truth label file.
    Each valid row corresponds to one colony.
    """
    if not label_path.exists():
        return 0

    count = 0
    with open(label_path, "r") as f:
        for line in f:
            if len(line.strip().split()) == 5:
                count += 1
    return count


def get_val_image_stems() -> list[str]:
    """
    Read the YOLO validation image folder and return stems like:
    sp01_img01
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    stems = []

    for image_path in sorted(YOLO_VAL_IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() in exts:
            stems.append(image_path.stem)

    return stems


def build_masked_image_paths(val_stems: list[str]) -> list[Path]:
    """
    Convert YOLO val stems to masked image paths used by the contour pipeline:
    sp01_img01 -> outputs/validation/masked_plates/sp01_img01_masked.jpg
    """
    image_paths = []

    for stem in val_stems:
        matched = False

        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            candidate = MASKED_IMAGE_DIR / f"{stem}_masked{ext}"
            if candidate.exists():
                image_paths.append(candidate)
                matched = True
                break

        if not matched:
            print(f"WARNING: masked image not found for {stem}")

    return image_paths


def main() -> None:
    config = CountingConfig()
    config.save_to_csv = False
    config.save_debug_images = False

    val_stems = get_val_image_stems()
    masked_image_paths = build_masked_image_paths(val_stems)

    print(f"YOLO val images: {len(val_stems)}")
    print(f"Matched masked images: {len(masked_image_paths)}")

    rows = []

    for image_path in masked_image_paths:
        pred_df = run_batch([image_path], config)
        pred_count = int(pred_df.iloc[0]["colony_count"])

        # convert sp01_img01_masked.jpg -> sp01_img01
        original_stem = image_path.stem.replace("_masked", "")
        true_label_path = YOLO_VAL_LABEL_DIR / f"{original_stem}.txt"
        true_count = count_true_boxes(true_label_path)

        rows.append({
            "image_name": f"{original_stem}{image_path.suffix}",
            "pred_count": pred_count,
            "true_count": true_count
        })

        print(f"{original_stem}: pred={pred_count}, true={true_count}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    df["class_true"] = df["true_count"] > 300
    df["class_pred"] = df["pred_count"] > 300
    accuracy = (df["class_true"] == df["class_pred"]).mean()

    print(f"\nSaved contour predictions to: {OUTPUT_PATH}")
    print(f">300 classification accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()