# src/data/

## این پوشه چیست؟

همه کدهای مربوط به **بارگذاری، تعریف، و تقسیم داده** اینجاست.

---

## فایل‌ها

### `dataset.py` — مهم‌ترین فایل پروژه

#### ۱. تعریف Feature ها — Single Source of Truth

```python
CLINICAL_FEATURES = [
    'AGE', 'GENDER', 'STAGE', 'SMOKING_HISTORY',
    'TMB_NONSYNONYMOUS', 'MSI_SCORE_MANTIS', 'TUMOR_PURITY',
    'Adenocarcinoma', 'Squamous', 'Neuroendocrine', 'Other_Subtype',
    'BONE', 'CNS_BRAIN', 'LIVER', 'LUNG', 'LYMPH_NODES',
]  # 16 feature — در inference استفاده می‌شود

GENOMIC_FEATURES = [
    'TP53', 'EGFR', 'KRAS', ...
]  # 50 feature — فقط در training استفاده می‌شود
```

**چرا مهم است:**
اگر اسم یک ستون تغییر کند، فقط همین‌جا را تغییر می‌دهیم.
تمام notebook ها از این لیست import می‌کنند و هیچ‌جا اسم feature
به صورت hard-code نوشته نمی‌شود.

---

#### ۲. `load_dataset(path)` — بارگذاری داده

```python
from src.data.dataset import load_dataset

df = load_dataset('data/processed/nsclc_final.csv')
# خروجی: DataFrame با 6,110 ردیف
```

کارهایی که انجام می‌دهد:
- چک می‌کند فایل وجود دارد
- CSV را می‌خواند
- تعداد بیماران و ستون‌ها را گزارش می‌دهد

---

#### ۳. `get_feature_cols(df, include_genomic)` — انتخاب feature ها

```python
# فقط clinical — برای inference
clinical_cols = get_feature_cols(df, include_genomic=False)

# همه feature ها — برای training
all_cols = get_feature_cols(df, include_genomic=True)
```

فقط ستون‌هایی را برمی‌گرداند که واقعاً در dataframe وجود دارند.

---

#### ۴. `make_splits(df, n_folds, val_fold, test_size)` — تقسیم داده

```python
from src.data.dataset import make_splits

splits = make_splits(df, n_folds=5, val_fold=0)
train_df = splits['train']
val_df   = splits['val']
test_df  = splits['test']
```

**نحوه کار:**
1. ابتدا 15% داده را به عنوان test جدا می‌کند (یکبار برای همیشه)
2. بقیه را با StratifiedKFold به 5 fold تقسیم می‌کند
3. fold مشخص‌شده را به عنوان validation برمی‌گرداند

**Stratified یعنی چه:**
نسبت سه کلاس درمان (Immunotherapy / Chemo / Targeted) در هر
split دقیقاً مثل کل dataset حفظ می‌شود.

---

#### ۵. ثابت‌های مهم

```python
TARGET_COL          = 'TREATMENT_TYPE'
SURVIVAL_EVENT_COL  = 'OS_EVENT'
SURVIVAL_TIME_COL   = 'OS_MONTHS'

TREATMENT_MAP = {
    'Immunotherapy': 0,
    'Chemotherapy':  1,
    'Targeted':      2,
}
```

---

## فایل‌هایی که در آینده اضافه می‌شوند

| فایل | فاز | کاربرد |
|------|-----|---------|
| `preprocessor.py` | ۳ | StandardScaler، log-transform |
| `torch_dataset.py` | ۴ | PyTorch Dataset class برای DataLoader |
| `augmentation.py` | ۴ | Data augmentation در صورت نیاز |
