# src/models/

## این پوشه چیست؟

تعریف **معماری تمام مدل‌های neural network** پروژه اینجاست.

الان خالی است — در **فاز ۴** پر می‌شود.

---

## فایل‌هایی که ساخته می‌شوند

### `encoders.py` — Encoder ها

**ClinicalEncoder:**
ورودی: ۱۶ feature کلینیکی
خروجی: یک embedding vector با بعد 64

```
Input (16) → Linear(16→64) → LayerNorm → ReLU → Dropout
           → Linear(64→128) → LayerNorm → ReLU → Dropout
           → Linear(128→64) → LayerNorm
Output (64)
```

**GenomicEncoder:**
ورودی: ۵۰ feature ژنتیکی (فقط در training)
خروجی: یک embedding vector با بعد 64

```
Input (50) → Linear(50→128) → BatchNorm → ReLU → Dropout
           → Linear(128→128) → BatchNorm → ReLU → Dropout
           → Linear(128→64) → BatchNorm
Output (64)
```

---

### `fusion.py` — Cross-Attention Fusion

ورودی: دو embedding (clinical و genomic)
خروجی: یک embedding ترکیبی

**نحوه کار:**
- Clinical embedding به عنوان Query
- Genomic embedding به عنوان Key و Value
- Multi-Head Attention با 4 head
- Clinical encoder یاد می‌گیرد کجا از genomic نگاه کند

فقط در training استفاده می‌شود.

---

### `cross_modal.py` — مدل کامل

ترکیب همه بخش‌ها:

```
Training mode:
  Clinical features → ClinicalEncoder → clinical_emb
  Genomic features  → GenomicEncoder  → genomic_emb
  (clinical_emb, genomic_emb) → CrossAttentionFusion → fused_emb
  fused_emb → ClassificationHead → treatment prediction

Inference mode:
  Clinical features → ClinicalEncoder → clinical_emb
  clinical_emb → ClassificationHead → treatment prediction
```

**Knowledge Distillation:**
در training، یک loss اضافه وجود دارد که ClinicalEncoder را مجبور
می‌کند خروجیش شبیه fused_emb باشد — حتی بدون genomic data.

---

### `baseline_models.py` — مدل‌های فاز ۳

Wrapper های scikit-learn:
- LogisticRegression با تنظیمات پروژه
- RandomForest با تنظیمات پروژه
- XGBoost با تنظیمات پروژه

---

## اصول طراحی

**۱. هر کلاس یک مسئولیت دارد**
ClinicalEncoder فقط encode می‌کند — نه classify، نه fusion.

**۲. Forward pass در training و inference فرق دارد**
```python
model(clinical, genomic, training=True)   # از fusion استفاده می‌کند
model(clinical, training=False)           # فقط clinical encoder
```

**۳. همه ابعاد از config می‌آیند**
هیچ عددی hard-code نیست — از `configs/cross_modal.yaml` خوانده می‌شود.
