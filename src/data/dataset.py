"""
Data loading and splitting for NSCLC dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Dict

# Feature groups — single source of truth for the whole project
CLINICAL_FEATURES = [
    'AGE', 'GENDER',
    'STAGE',
    'SMOKING_HISTORY',
    'TMB_NONSYNONYMOUS', 'MSI_SCORE_MANTIS', 'TUMOR_PURITY',
    'Adenocarcinoma', 'Squamous', 'Neuroendocrine', 'Other_Subtype',
    'BONE', 'CNS_BRAIN', 'LIVER', 'LUNG', 'LYMPH_NODES',
]

GENOMIC_FEATURES = [
    'TP53', 'EGFR', 'KRAS', 'KEAP1', 'STK11',
    'CDKN2A', 'RB1', 'SMARCA4', 'NF1', 'MET',
    'BRAF', 'PIK3CA', 'PTEN', 'RET', 'ALK',
    'ROS1', 'NTRK1', 'NTRK2', 'NTRK3', 'MYC',
    'FGFR1', 'FGFR2', 'FGFR3', 'ERBB2', 'ERBB3',
    'APC', 'CTNNB1', 'NOTCH1', 'NOTCH2', 'NOTCH3',
    'FAT1', 'FAT2', 'FAT3', 'FAT4', 'PTCH1',
    'SETD2', 'KDM6A', 'ARID1A', 'KMT2D', 'KMT2C',
    'SMAD4', 'SMAD2', 'TGFb1', 'TGFBR2', 'ACVR2A',
    'MDM2', 'MDM4', 'ATM', 'CHEK2', 'BRCA2',
]

TARGET_COL = 'TREATMENT_TYPE'
SURVIVAL_EVENT_COL = 'OS_EVENT'
SURVIVAL_TIME_COL = 'OS_MONTHS'

TREATMENT_MAP = {
    'Immunotherapy': 0,
    'Chemotherapy': 1,
    'Targeted': 2,
}


def load_dataset(processed_path: str) -> pd.DataFrame:
    """Load processed NSCLC dataset."""
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} patients, {df.shape[1]} columns")
    return df


def get_feature_cols(df: pd.DataFrame, include_genomic: bool = True) -> list:
    """Return available feature columns based on mode."""
    clinical = [c for c in CLINICAL_FEATURES if c in df.columns]
    if not include_genomic:
        return clinical
    genomic = [c for c in GENOMIC_FEATURES if c in df.columns]
    return clinical + genomic


def make_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    val_fold: int = 0,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Stratified split: train / val / test
    Test set is held out first, then CV folds on the rest.
    """
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COL],
        random_state=random_state,
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = list(skf.split(train_val, train_val[TARGET_COL]))
    train_idx, val_idx = folds[val_fold]

    return {
        'train': train_val.iloc[train_idx].reset_index(drop=True),
        'val':   train_val.iloc[val_idx].reset_index(drop=True),
        'test':  test.reset_index(drop=True),
    }
