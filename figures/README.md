# figures/

## این پوشه چیست؟

محل ذخیره **تمام نمودارها و تصاویر** پروژه — از EDA تا figures نهایی مقاله.

---

## ساختار

```
figures/
├── eda/              ← نمودارهای فاز ۲ (EDA)
├── baseline/         ← نمودارهای فاز ۳
├── cross_modal/      ← نمودارهای فاز ۴
├── subgroup/         ← نمودارهای فاز ۵
├── validation/       ← نمودارهای فاز ۶
└── paper/            ← figures نهایی برای مقاله (high resolution)
```

---

## figures مورد انتظار به ترتیب فاز

### فاز ۲ — EDA
```
figures/eda/
├── distributions_numeric.png      ← histogram همه feature های عددی
├── distributions_categorical.png  ← bar chart feature های دسته‌ای
├── correlation_heatmap.png        ← heatmap همبستگی
├── treatment_comparison_box.png   ← boxplot هر feature به تفکیک درمان
├── kaplan_meier_treatment.png     ← KM curve سه گروه درمانی
└── tmb_distribution_by_treatment.png
```

### فاز ۳ — Baseline
```
figures/baseline/
├── model_comparison_bar.png       ← مقایسه accuracy/F1/AUC مدل‌ها
├── confusion_matrix_xgb.png       ← confusion matrix بهترین baseline
├── shap_clinical_only.png         ← SHAP برای clinical-only model
├── shap_all_features.png          ← SHAP برای model با همه features
└── gap_analysis.png               ← نمایش gap بین clinical vs all
```

### فاز ۴ — Cross-Modal
```
figures/cross_modal/
├── architecture_diagram.png       ← نمودار معماری (figure 1 مقاله)
├── training_curves.png            ← learning curves
├── ablation_results.png           ← مقایسه ablation study
├── lambda_sensitivity.png         ← تاثیر lambda_kd بر performance
└── attention_weights.png          ← visualization attention weights
```

### فاز ۵ — Subgroup
```
figures/subgroup/
├── umap_embeddings.png            ← UMAP از learned embeddings
├── cluster_kaplan_meier.png       ← KM per cluster (figure 2 مقاله)
├── cluster_heatmap.png            ← feature profile هر cluster
└── shap_per_cluster.png           ← SHAP به تفکیک subgroup
```

### paper/ — figures نهایی
```
figures/paper/
├── fig1_architecture.pdf          ← 300 DPI، vector format
├── fig2_kaplan_meier.pdf
├── fig3_results_table.pdf
├── fig4_shap_analysis.pdf
└── supplementary/
```

---

## استانداردهای figures مقاله

**رزولوشن:** حداقل 300 DPI برای journal submission

**فرمت:** PDF یا TIFF ترجیح داده می‌شود — PNG برای review قابل قبول است

**فونت:** حداقل 8pt در figure نهایی

**رنگ:** رنگ‌ها باید برای colorblind readers هم قابل تشخیص باشند
از پالت `seaborn colorblind` یا `matplotlib tab10` استفاده کنید

**اندازه:** معمولاً single-column (8.5cm) یا double-column (17.4cm)
بسته به journal

---

## نحوه ذخیره figure در notebook

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
# ... رسم نمودار ...

# ذخیره با کیفیت بالا
fig.savefig('figures/eda/kaplan_meier_treatment.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
```
