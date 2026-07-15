# Phase 4 — Cross-Modal Architecture

## Notebooks (run in order)

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_architecture.ipynb` | Model definition and sanity checks |
| 02 | `02_training.ipynb` | Training loop and monitoring |
| 03 | `03_ablation.ipynb` | Ablation study (no distillation, no attention) |
| 04 | `04_results.ipynb` | Final comparison table |
| script | `train_generalized_hoffman.py` | Generalized distillation on logits + Hoffman-style intermediate feature alignment |

## Key Questions to Answer
- [ ] Does distillation improve over clinical-only baseline?
- [ ] What lambda (KD weight) is optimal?
- [ ] Which genomic features are distilled most?
