from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import copy
import random
import math
import pandas as pd
import cv2

from src.config import CountingConfig
from src.preprocessing.counting import load_accepted_filenames, get_accepted_image_paths
from src.model_selection.evaluate_counting import evaluate_config


# =========================
# SETTINGS
# =========================

RANDOM_SEED = 42
N_TRIALS = 25

GROUND_TRUTH_EXCEL = Path("data/raw_dataset/images.xls")
RESULTS_CSV = Path("outputs/model_evaluation/tuning_results.csv")

# column names in ground-truth file
IMAGE_NAME_COL = "image_name"
TRUE_COUNT_COL = "number of CFUs"


# =========================
# HELPERS
# =========================

def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_ground_truth(excel_path: Path) -> pd.DataFrame:
    """
    Load ground-truth counts and keep only the needed columns.
    """
    df = pd.read_excel(excel_path)

    required_cols = {IMAGE_NAME_COL, TRUE_COUNT_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in ground-truth CSV: {missing}")

    return df[[IMAGE_NAME_COL, TRUE_COUNT_COL]].copy()


def make_base_config() -> CountingConfig:
    """
    Create a clean config for tuning.
    """
    config = CountingConfig()
    config.save_to_csv = False
    config.save_debug_images = False
    config.ensure_counting_dirs()
    return config


def sample_odd_kernel(rng: random.Random, choices: list[int]) -> tuple[int, int]:
    k = rng.choice(choices)
    return (k, k)


def sample_config(rng: random.Random) -> CountingConfig:
    """
    Sample one random configuration.
    """
    config = make_base_config()

    # threshold
    config.threshold.use_otsu = rng.choice([True, False])
    config.threshold.gaussian_kernel_size = sample_odd_kernel(rng, [3, 5, 7])

    # threshold value only matters when Otsu is False
    config.threshold.threshold_value = rng.randint(70, 190)

    # morphology
    config.morphology.morph_kernel_size = sample_odd_kernel(rng, [1, 3, 5])
    config.morphology.use_opening = rng.choice([True, False])
    config.morphology.use_closing = rng.choice([True, False])

    # watershed
    config.watershed.use_watershed = rng.choice([True, False])
    config.watershed.distance_metric = cv2.DIST_L2
    config.watershed.distance_mask_size = 5
    config.watershed.peak_threshold_fraction = rng.uniform(0.25, 0.45)
    config.watershed.marker_min_area = rng.randint(5, 30)

    # contour filtering
    config.contour.min_area = rng.randint(5, 800)
    config.contour.max_area = rng.randint(2000, 100000)

    # enforce valid min/max relation
    if config.contour.max_area <= config.contour.min_area:
        config.contour.max_area = config.contour.min_area + 1000

    config.contour.min_circularity = rng.uniform(0.2, 0.95)

    # mask
    config.mask.shrink_factor = rng.uniform(0.90, 0.99)

    return config


def flatten_config(config: CountingConfig) -> dict:
    """
    Flatten config fields into a row for the tuning-results table.
    """
    return {
        "gaussian_kernel_size": config.threshold.gaussian_kernel_size,
        "threshold_value": config.threshold.threshold_value,
        "use_otsu": config.threshold.use_otsu,
        "morph_kernel_size": config.morphology.morph_kernel_size,
        "use_opening": config.morphology.use_opening,
        "use_closing": config.morphology.use_closing,
        "use_watershed": config.watershed.use_watershed,
        "distance_metric": config.watershed.distance_metric,
        "distance_mask_size": config.watershed.distance_mask_size,
        "peak_threshold_fraction": config.watershed.peak_threshold_fraction,
        "marker_min_area": config.watershed.marker_min_area,
        "min_area": config.contour.min_area,
        "max_area": config.contour.max_area,
        "min_circularity": config.contour.min_circularity,
        "shrink_factor": config.mask.shrink_factor,
    }


def evaluate_one_config(
    config: CountingConfig,
    image_paths: list[Path],
    real_df: pd.DataFrame,
) -> dict:
    """
    Run evaluation and return a single flat result row.

    Supports both:
    - evaluate_config(...) -> metrics dict
    - evaluate_config(...) -> {"metrics": ..., "results_df": ...}
    """
    result = evaluate_config(config, image_paths, real_df)

    if isinstance(result, dict) and "metrics" in result:
        metrics = result["metrics"]
    else:
        metrics = result

    row = flatten_config(config)
    row.update(metrics)
    return row


# =========================
# MAIN TUNING LOOP
# =========================

def main() -> None:
    rng = random.Random(RANDOM_SEED)
    ensure_output_dir(RESULTS_CSV)

    # load ground truth
    real_df = load_ground_truth(GROUND_TRUTH_EXCEL)

    # collect accepted images from your validated pipeline
    base_config = make_base_config()
    accepted_filenames = load_accepted_filenames(base_config)
    image_paths = get_accepted_image_paths(accepted_filenames, base_config)

    print(f"Ground-truth rows: {len(real_df)}")
    print(f"Accepted images: {len(image_paths)}")
    print(f"Running {N_TRIALS} trials...")

    all_rows = []
    best_row = None
    best_score = math.inf

    for trial_idx in range(1, N_TRIALS + 1):
        config = sample_config(rng)

        try:
            row = evaluate_one_config(config, image_paths, real_df)
            row["trial"] = trial_idx
            all_rows.append(row)

            mae = row["mae"]
            if mae < best_score:
                best_score = mae
                best_row = row

            print(
                f"[{trial_idx}/{N_TRIALS}] "
                f"MAE={row['mae']:.3f}, "
                f"RMSE={row['rmse']:.3f}, "
                f"min_area={row['min_area']}, "
                f"max_area={row['max_area']}, "
                f"circ={row['min_circularity']:.3f}, "
                f"shrink={row['shrink_factor']:.3f}, "
                f"otsu={row['use_otsu']}"
            )

        except Exception as e:
            error_row = {
                "trial": trial_idx,
                "error_message": str(e),
                **flatten_config(config),
            }
            all_rows.append(error_row)
            print(f"[{trial_idx}/{N_TRIALS}] FAILED: {e}")

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(by="mae", ascending=True, na_position="last")
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\nSaved tuning results to:", RESULTS_CSV)

    if best_row is not None:
        print("\nBest configuration found:")
        for key, value in best_row.items():
            if key != "trial":
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()