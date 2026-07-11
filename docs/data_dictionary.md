# Data Dictionary

## Clinical Features (16) — Available at Inference

| Feature | Type | Description |
|---------|------|-------------|
| AGE | Numeric | Age at diagnosis |
| GENDER | Binary | 0=Female, 1=Male |
| STAGE | Ordinal | Cancer stage (encoded) |
| SMOKING_HISTORY | Ordinal | Smoking status |
| TMB_NONSYNONYMOUS | Numeric | Tumor Mutational Burden |
| MSI_SCORE_MANTIS | Numeric | Microsatellite Instability |
| TUMOR_PURITY | Numeric | Tumor purity (0-1) |
| Adenocarcinoma | Binary | Subtype flag |
| Squamous | Binary | Subtype flag |
| Neuroendocrine | Binary | Subtype flag |
| Other_Subtype | Binary | Subtype flag |
| BONE | Binary | Bone metastasis |
| CNS_BRAIN | Binary | CNS/Brain metastasis |
| LIVER | Binary | Liver metastasis |
| LUNG | Binary | Lung metastasis |
| LYMPH_NODES | Binary | Lymph node metastasis |

## Genomic Features (50) — Training Only

Top 50 NSCLC driver genes from mutation data.
See `src/data/dataset.py` → `GENOMIC_FEATURES` for full list.

## Target Variable

| Column | Description |
|--------|-------------|
| TREATMENT_TYPE | 0=Immunotherapy, 1=Chemotherapy, 2=Targeted |
| OS_EVENT | Overall Survival event (1=death, 0=censored) |
| OS_MONTHS | Follow-up time in months |
