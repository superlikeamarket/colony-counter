from src.config import CountingConfig
from src.preprocessing.counting import run_batch
from pathlib import Path
import pandas as pd


def evaluate_config(
        config: CountingConfig,image_paths: list[Path], real: pd.DataFrame
    ) -> dict:
    """
    Evaluate a set of configuration values based on mean absolute error, RMSE,
    median absolute error, mean relative error and % within +-5%

    Input:
    - config: a set of randomly chosen configuration values
    - image_paths: paths to the input masked images
    - real: dataframe with real colony counts and image name labels

    Output:
    - metrics: a dictionary with metrics values for this set of config values
    """
    # run counting on all images
    pred = run_batch(image_paths, config)

    # merge predictions with true counts
    df = pred.merge(
        real[["image_name", "number of CFUs"]],
        on="image_name",
        how="inner"
    )
    if len(df) != len(pred):
        raise ValueError("Some predictions did not match ground-truth labels.")

    # compute MAE, RMSE, relative error
    df["abs_error"] = (df["colony_count"] - df["number of CFUs"]).abs()
    df["error"] = df["colony_count"] - df["number of CFUs"]
    df["relative_error"] = df["abs_error"] / df["number of CFUs"].replace(0, pd.NA)

    metrics = {}
    metrics["mae"] = df["abs_error"].mean()
    metrics["rmse"] = (df["error"] ** 2).mean() ** 0.5
    metrics["median_ae"] = df["abs_error"].median()
    metrics["mre"] = df["relative_error"].mean()
    metrics["within_5_pc"] = (df["relative_error"]  <= 0.05).mean()

    # return metrics
    return {
        "metrics": metrics,
        "results_df": df
    }
