# experiments/

## این پوشه چیست؟

محل ذخیره **نتایج تمام آزمایش‌ها و مدل‌های train شده**.

هر بار که یک مدل را train می‌کنیم، نتایجش در یک زیرپوشه
با نام منحصربه‌فرد اینجا ذخیره می‌شود.

---

## ساختار نام‌گذاری

```
experiments/
├── 01_baseline_clinical_only/
├── 02_baseline_all_features/
├── 03_crossmodal_lambda05/
├── 04_crossmodal_lambda10/
├── 05_crossmodal_ablation_no_attention/
└── ...
```

**قرارداد نام‌گذاری:**
```
{شماره}_{نام_مدل}_{پارامتر_کلیدی}/
```

---

## محتوای هر پوشه experiment

```
experiments/03_crossmodal_lambda05/
├── config.yaml          ← کپی دقیق config مورد استفاده
├── metrics.json         ← نتایج عددی نهایی
├── training_log.csv     ← loss و metrics در هر epoch
├── checkpoints/
│   ├── best_model.pt    ← بهترین مدل (بر اساس val loss)
│   └── last_model.pt    ← آخرین epoch
└── predictions/
    ├── val_predictions.csv
    └── test_predictions.csv
```

---

## فایل `metrics.json` — ساختار

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

## فایل `training_log.csv` — ساختار

```
epoch, train_loss, val_loss, train_f1, val_f1, lr
1,     1.0821,    1.0654,   0.412,    0.423,  0.001
2,     0.9234,    0.9187,   0.501,    0.498,  0.001
...
87,    0.4123,    0.4891,   0.761,    0.748,  0.0003
```

این فایل برای رسم learning curve در notebook استفاده می‌شود.

---

## نکات مهم

**checkpoints در `.gitignore` هستند:**
فایل‌های `.pt` به GitHub push نمی‌شوند چون حجیم هستند.
فقط `metrics.json` و `config.yaml` push می‌شوند.

**چرا config را کپی می‌کنیم؟**
اگر بعداً `configs/cross_modal.yaml` را تغییر دهید،
همچنان می‌دانید آزمایش ۰۳ با چه تنظیماتی اجرا شده.

**چطور بهترین experiment را پیدا کنیم؟**
```python
# notebook مقایسه
import json, glob

results = []
for path in glob.glob('experiments/*/metrics.json'):
    with open(path) as f:
        results.append(json.load(f))

# مرتب کردن بر اساس test macro_f1
results.sort(key=lambda x: x['test']['macro_f1'], reverse=True)
```
