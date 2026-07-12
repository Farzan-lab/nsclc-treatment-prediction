# src/evaluation/

## این پوشه چیست؟

همه کدهای مربوط به **ارزیابی و اندازه‌گیری عملکرد مدل‌ها** اینجاست.

---

## فایل‌های موجود

### `metrics.py` — معیارهای ارزیابی

**`compute_metrics(y_true, y_pred, y_prob)`**

```python
from src.evaluation.metrics import compute_metrics

metrics = compute_metrics(
    y_true = test_labels,
    y_pred = predicted_classes,
    y_prob = predicted_probabilities  # اختیاری
)

# خروجی:
# {
#   'accuracy':    0.76,
#   'macro_f1':    0.74,
#   'weighted_f1': 0.75,
#   'auc_ovr':     0.82
# }
```

**`print_metrics(metrics, title)`**

نتایج را به صورت خوانا چاپ می‌کند:
```
==================================================
  XGBoost — Clinical Only
==================================================
  accuracy             0.7612
  macro_f1             0.7438
  weighted_f1          0.7521
  auc_ovr              0.8203
```

---

## معیارها — توضیح

**Accuracy:**
درصد کل پیش‌بینی‌های صحیح.
مشکل: اگر dataset imbalanced باشد، گمراه‌کننده است.

**Macro F1:**
میانگین F1 هر کلاس بدون توجه به اندازه کلاس.
مهم‌ترین معیار برای dataset های imbalanced — همه کلاس‌ها وزن برابر دارند.
این عدد را در مقاله گزارش می‌دهیم.

**Weighted F1:**
میانگین F1 هر کلاس با وزن نسبت به اندازه کلاس.
برای نشان دادن عملکرد کلی مناسب است.

**AUC-OVR (One-vs-Rest):**
برای هر کلاس، مدل را در مقابل بقیه ارزیابی می‌کند.
بین ۰.۵ (تصادفی) و ۱.۰ (کامل).

---

## فایل‌هایی که اضافه می‌شوند

### `comparison.py` — مقایسه مدل‌ها (فاز ۳ و ۴)

```python
results = compare_models({
    'clinical_only_xgb': metrics_1,
    'all_features_xgb':  metrics_2,
    'cross_modal_ours':  metrics_3,
})
# جدول مقایسه‌ای می‌سازد
```

### `survival_eval.py` — ارزیابی Survival (فاز ۶)

```python
# آیا بیمارانی که درمان صحیح predict شد OS بهتری دارند؟
plot_kaplan_meier_by_prediction(df, model_predictions)
log_rank_test(correct_group, incorrect_group)
```

### `shap_analysis.py` — تفسیرپذیری (فاز ۳ و ۵)

```python
# کدام feature ها بیشترین تاثیر را دارند؟
shap_values = compute_shap(model, X_test)
plot_shap_summary(shap_values, feature_names)
plot_shap_per_class(shap_values, class_names)
```

### `decision_curve.py` — Clinical Utility (فاز ۶)

Decision Curve Analysis: آیا استفاده از مدل نسبت به
"همه را یک درمان بده" بهتر است؟
این برای مقاله پزشکی ضروری است.
