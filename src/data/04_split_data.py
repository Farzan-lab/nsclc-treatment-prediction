"""
Phase 3.1 - Data Splitting (once and for all)

Two-step stratified split on TREATMENT_FINAL:
    step 1: hold out 15% as test        -> 917  patients
    step 2: from the remaining 85%,
            take 17% as validation       -> 871  patients
            rest is train                -> 4322 patients

Result: 68% train / 14% val / 15% test, stratified so the 3-class
proportions match across all splits. Run this ONCE; every model in
Phase 3 and 4 loads these exact files.

Usage:
    python src/data/04_split_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- adjust these two lines to your repo ------------------------------------
DATA_PATH = Path("C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv")   # <- your final 6,110-row dataset
OUT_DIR = Path("C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/splits")
# ----------------------------------------------------------------------------
TARGET = "TREATMENT_FINAL"
SEED = 42


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    df = load_dataset(DATA_PATH).reset_index(drop=True)

    assert TARGET in df.columns, f"'{TARGET}' not found in dataset"
    assert df.isna().sum().sum() == 0, "dataset still contains missing values"

    y = df[TARGET]

    # step 1: 15% test
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=y, random_state=SEED,
    )
    # step 2: 17% of the remaining -> validation
    train, val = train_test_split(
        train_val,
        test_size=0.17,
        stratify=train_val[TARGET],
        random_state=SEED,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        # row indices into the original dataset, for full reproducibility
        np.save(OUT_DIR / f"{name}_idx.npy", part.index.to_numpy())

    # ---- sanity report -----------------------------------------------------
    n = len(df)
    print("split sizes")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:<5} {len(part):>5}  ({len(part) / n:5.1%})")

    print("\nclass proportions per split (should be nearly identical):")
    header = "  " + "".join(f"{c:>16}" for c in sorted(y.unique()))
    print(header)
    for name, part in [("all", df), ("train", train), ("val", val), ("test", test)]:
        props = part[TARGET].value_counts(normalize=True)
        row = "".join(f"{props.get(c, 0):>16.3f}" for c in sorted(y.unique()))
        print(f"  {name:<5}{row}")


if __name__ == "__main__":
    main()
