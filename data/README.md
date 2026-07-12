# data/

## این پوشه چیست؟

محل نگهداری **تمام فایل‌های داده** پروژه.

⚠️ **این پوشه در `.gitignore` است — هیچ داده‌ای به GitHub push نمی‌شود.**
این هم به خاطر حریم خصوصی بیماران است، هم به خاطر حجم فایل‌ها.

---

## زیرپوشه‌ها

### `raw/` — داده خام اصلی

فایل‌های اصلی MSK-CHORD که مستقیم از cBioPortal دانلود شده‌اند.
**هرگز این فایل‌ها را تغییر ندهید.**

فایل‌های مورد انتظار:
```
data/raw/
├── data_clinical_patient.txt     ← اطلاعات بیمار
├── data_clinical_sample.txt      ← اطلاعات نمونه
├── data_mutations.txt            ← داده جهش‌های ژنتیکی
├── data_timeline_treatment.txt   ← تاریخچه درمان
├── data_timeline_status.txt      ← وضعیت بیمار در زمان
└── data_timeline_pdl1.txt        ← داده PDL1
```

---

### `processed/` — داده پردازش‌شده

خروجی **فاز ۱** (آماده‌سازی داده) اینجاست.

```
data/processed/
└── nsclc_final.csv    ← dataset نهایی، آماده برای مدل‌سازی
```

**مشخصات `nsclc_final.csv`:**
- ۶,۱۱۰ بیمار NSCLC
- ۶۶ ستون feature (16 کلینیکی + 50 ژنتیکی)
- ۳ ستون target (TREATMENT_TYPE، OS_EVENT، OS_MONTHS)
- بدون missing value

---

### `splits/` — تقسیم‌بندی داده

فایل‌های index که مشخص می‌کنند هر بیمار در کدام split است.
این فایل‌ها توسط `src/data/dataset.py` ساخته می‌شوند.

```
data/splits/
├── test_indices.npy          ← index های test set (ثابت، یکبار ساخته می‌شود)
├── fold_0_train_indices.npy
├── fold_0_val_indices.npy
├── fold_1_train_indices.npy
└── ...
```

**چرا index ها را ذخیره می‌کنیم؟**
برای اینکه همه مدل‌ها (baseline و cross-modal) دقیقاً روی
همان train/val/test بیماران ارزیابی شوند و مقایسه عادلانه باشد.

---

## نحوه راه‌اندازی (برای کسی که از صفر شروع می‌کند)

```bash
# ۱. دانلود MSK-CHORD از cBioPortal
# آدرس: https://www.cbioportal.org/study/summary?id=msk_chord_2024

# ۲. قرار دادن فایل‌ها در data/raw/

# ۳. اجرای notebook آماده‌سازی داده
jupyter notebook notebooks/00_data_preparation/

# ۴. خروجی در data/processed/nsclc_final.csv ذخیره می‌شود
```

---

## نکته مهم درباره داده

داده MSK-CHORD تحت شرایط استفاده cBioPortal است.
استفاده از این داده برای تحقیقات غیرتجاری مجاز است.
قبل از هر انتشاری، شرایط استفاده را مطالعه کنید:
https://www.cbioportal.org/terms
