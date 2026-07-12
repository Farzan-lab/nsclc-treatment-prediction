# experiments/

## What is this folder?

The storage location for **all experiment results and trained models**.

Every time a model is trained, its results are saved in a uniquely named
subfolder inside this directory.

---

## Naming Convention

```
experiments/
├── 01_baseline_clinical_only/
├── 02_baseline_all_features/
├── 03_crossmodal_lambda05/
├── 04_crossmodal_lambda10/
├── 05_crossmodal_ablation_no_attention/
└── ...
```

**Naming format:**
```
{number}_{model_name}_{key_parameter}/
```

---

## Contents of Each Experiment Folder

```
experiments/03_crossmodal_lambda05/
├── config.yaml          ← Exact copy of the config used for this run
├── metrics.json         ← Final numerical results
├── training_log.csv     ← Loss and metrics recorded at every epoch
├── checkpoints/
│   ├── best_model.pt    ← Best model checkpoint (based on val loss)
│   └── last_model.pt    ← Checkpoint from the final epoch
└── predictions/
    ├── val_predictions.csv
    └── test_predictions.csv
```

---

## `metrics.json` — Structure

```json
{
  "experiment_name": "crossmodal_lambda05",
  "date": "2025-07-12",
  "val": {
    "accuracy": 0.762,
    "macro_f1": 0.748,
    "auc_ovr": 0.831
  },
  "test": {
    "accuracy": 0.751,
    "macro_f1": 0.739,
    "auc_ovr": 0.824
  },
  "config": {
    "lambda_kd": 0.5,
    "embed_dim": 64,
    "epochs_trained": 87
  }
}
```

---

## `training_log.csv` — Structure

```
epoch, train_loss, val_loss, train_f1, val_f1, lr
1,     1.0821,    1.0654,   0.412,    0.423,  0.001
2,     0.9234,    0.9187,   0.501,    0.498,  0.001
...
87,    0.4123,    0.4891,   0.761,    0.748,  0.0003
```

This file is used to plot the learning curve inside the notebooks.

---

## Important Notes

**Checkpoints are listed in `.gitignore`:**
`.pt` files are not pushed to GitHub due to their large size.
Only `metrics.json` and `config.yaml` are committed and pushed.

**Why do we copy the config?**
If `configs/cross_modal.yaml` is modified later, you can still know
exactly which settings were used for experiment 03.

**How to find the best experiment:**

```python
# comparison notebook
import json, glob

results = []
for path in glob.glob('experiments/*/metrics.json'):
    with open(path) as f:
        results.append(json.load(f))

# sort by test macro_f1 (descending)
results.sort(key=lambda x: x['test']['macro_f1'], reverse=True)
```
