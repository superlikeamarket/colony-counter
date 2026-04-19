from pathlib import Path
import pandas as pd


def ensure_csv_with_header(path: Path, columns: list):
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)