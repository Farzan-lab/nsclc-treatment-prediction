# NSCLC Treatment Prediction via Cross-Modal Knowledge Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the implementation for our research on treatment prediction
in Non-Small Cell Lung Cancer (NSCLC) using cross-modal knowledge distillation.

**Core Idea:** Train with both clinical (16 features) and genomic (50 genes) data,
but perform inference using only cheap clinical features — making the model
accessible to patients without genomic sequencing.

**Dataset:** MSK-CHORD (Nature 2024) — 6,110 NSCLC patients

---

## Project Structure

```
nsclc-treatment-prediction/
│
├── data/                    # Data directory (gitignored)
│   ├── raw/                 # Original MSK-CHORD files
│   ├── processed/           # Phase 1 output: nsclc_final.csv
│   └── splits/              # Train/val/test splits
│
├── notebooks/               # Analysis notebooks
│   ├── 01_eda/              # Phase 2: Exploratory Data Analysis
│   ├── 02_baseline/         # Phase 3: Baseline models
│   ├── 03_cross_modal/      # Phase 4: Cross-modal architecture
│   └── 04_subgroup/         # Phase 5: Subgroup discovery
│
├── src/                     # Source code (importable modules)
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model architectures
│   ├── training/            # Training loops and losses
│   └── evaluation/          # Metrics and evaluation
│
├── experiments/             # Experiment results and logs
├── figures/                 # Publication-ready figures
├── tests/                   # Unit tests
├── configs/                 # Hyperparameter configs (YAML)
└── docs/                    # Documentation
```

---

## Research Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Data preparation (MSK-CHORD → nsclc_final.csv) | ✅ Done |
| Phase 2 | EDA & Statistical Analysis | 🔄 In Progress |
| Phase 3 | Baseline Models (clinical-only & clinical+genomic) | ⏳ Pending |
| Phase 4 | Cross-Modal Architecture + Knowledge Distillation | ⏳ Pending |
| Phase 5 | Subgroup Discovery | ⏳ Pending |
| Phase 6 | Validation & Integration | ⏳ Pending |
| Phase 7 | Final Analysis & Paper Writing | ⏳ Pending |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nsclc-treatment-prediction.git
cd nsclc-treatment-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Data

Data is from the **MSK-CHORD dataset** (Nguyen et al., Nature 2024).
Due to data access restrictions, raw data is not included in this repository.

**To reproduce:**
1. Download MSK-CHORD from [cBioPortal](https://www.cbioportal.org/)
2. Place files in `data/raw/`
3. Run Phase 1 preprocessing notebook

**Final dataset stats:**
- Patients: 6,110 NSCLC
- Clinical features: 16
- Genomic features: 50 (top NSCLC genes)
- Treatment classes: Immunotherapy / Chemotherapy / Targeted

---

## Key Results

*To be updated as experiments complete.*

| Model | Features (Inference) | Macro F1 | AUC |
|-------|---------------------|----------|-----|
| Logistic Regression | Clinical only | — | — |
| XGBoost | Clinical only | — | — |
| XGBoost | Clinical + Genomic | — | — |
| **Ours (Cross-Modal KD)** | **Clinical only** | **—** | **—** |

---

## Citation

```bibtex
@article{nsclc_kd_2025,
  title   = {Cross-Modal Knowledge Distillation for NSCLC Treatment Prediction},
  author  = {Your Name},
  journal = {TBD},
  year    = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
