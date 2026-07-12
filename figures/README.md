# figures/

## What is this folder?

The storage location for **all plots and images** in the project — from EDA all the way to the final paper figures.

---

## Structure

```
figures/
├── eda/              ← Phase 2 plots (EDA)
├── baseline/         ← Phase 3 plots
├── cross_modal/      ← Phase 4 plots
├── subgroup/         ← Phase 5 plots
├── validation/       ← Phase 6 plots
└── paper/            ← Final high-resolution figures for the paper
```

---

## Expected Figures by Phase

### Phase 2 — EDA

```
figures/eda/
├── distributions_numeric.png      ← Histograms for all numeric features
├── distributions_categorical.png  ← Bar charts for categorical features
├── correlation_heatmap.png        ← Feature correlation heatmap
├── treatment_comparison_box.png   ← Boxplots per feature split by treatment group
├── kaplan_meier_treatment.png     ← Kaplan-Meier curves for the three treatment groups
└── tmb_distribution_by_treatment.png
```

### Phase 3 — Baseline

```
figures/baseline/
├── model_comparison_bar.png       ← Accuracy / F1 / AUC comparison across models
├── confusion_matrix_xgb.png       ← Confusion matrix for the best baseline model
├── shap_clinical_only.png         ← SHAP values for the clinical-only model
├── shap_all_features.png          ← SHAP values for the model trained on all features
└── gap_analysis.png               ← Visualization of the gap between clinical-only vs all features
```

### Phase 4 — Cross-Modal

```
figures/cross_modal/
├── architecture_diagram.png       ← Model architecture diagram (Figure 1 in the paper)
├── training_curves.png            ← Learning curves
├── ablation_results.png           ← Ablation study comparison
├── lambda_sensitivity.png         ← Effect of lambda_kd on model performance
└── attention_weights.png          ← Attention weight visualization
```

### Phase 5 — Subgroup

```
figures/subgroup/
├── umap_embeddings.png            ← UMAP projection of learned embeddings
├── cluster_kaplan_meier.png       ← Kaplan-Meier per cluster (Figure 2 in the paper)
├── cluster_heatmap.png            ← Feature profile heatmap per cluster
└── shap_per_cluster.png           ← SHAP values broken down by subgroup
```

### paper/ — Final Figures

```
figures/paper/
├── fig1_architecture.pdf          ← 300 DPI, vector format
├── fig2_kaplan_meier.pdf
├── fig3_results_table.pdf
├── fig4_shap_analysis.pdf
└── supplementary/
```

---

## Paper Figure Standards

**Resolution:** Minimum 300 DPI for journal submission

**Format:** PDF or TIFF is preferred — PNG is acceptable for the review stage

**Font size:** Minimum 8pt in the final figure

**Color:** Colors must be distinguishable for colorblind readers.
Use the `seaborn colorblind` palette or `matplotlib tab10`

**Dimensions:** Typically single-column (8.5 cm) or double-column (17.4 cm)
depending on the target journal

---

## How to Save a Figure in a Notebook

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
# ... plotting code ...

# Save with high quality
fig.savefig('figures/eda/kaplan_meier_treatment.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
```
