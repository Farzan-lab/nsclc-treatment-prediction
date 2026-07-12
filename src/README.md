# src/

## این پوشه چیست؟

این پوشه **کد اصلی و قابل استفاده مجدد** پروژه را نگه می‌دارد.

تفاوت این پوشه با notebooks:
- کد اینجا **اجرا نمی‌شود** — فقط تعریف می‌شود
- کد اینجا **import می‌شود** — از داخل notebook ها
- کد اینجا **modular** است — هر بخش مستقل از بقیه کار می‌کند
- کد اینجا **قابل تست** است — unit test می‌توان نوشت

---

## ساختار

```
src/
├── data/          ← بارگذاری و پردازش داده
├── models/        ← تعریف معماری مدل‌ها
├── training/      ← حلقه‌های آموزش و loss functions
└── evaluation/    ← معیارهای ارزیابی
```

---

## پوشه‌ها

### `data/` — مدیریت داده
همه کدهای مربوط به خواندن، پردازش، و تقسیم داده.

**چه چیزی اینجاست:**
- تعریف نام feature ها (single source of truth)
- تابع بارگذاری dataset
- تابع تقسیم به train/val/test
- هر preprocessing که باید روی داده انجام شود

**چه چیزی اینجا نیست:**
- visualization (آن در notebooks است)
- تحلیل آماری (آن در notebooks است)

---

### `models/` — معماری مدل‌ها (فاز ۴)
تعریف کلاس‌های PyTorch برای:
- Clinical Encoder
- Genomic Encoder
- Cross-Attention Fusion
- مدل کامل با Knowledge Distillation

در فاز ۴ پر می‌شود.

---

### `training/` — آموزش مدل (فاز ۴)
- حلقه train و validation
- Loss functions (CE + Distillation)
- Early stopping
- ذخیره checkpoint

در فاز ۴ پر می‌شود.

---

### `evaluation/` — ارزیابی
- محاسبه Accuracy، F1، AUC
- گزارش دهی نتایج
- مقایسه مدل‌ها

---

## نحوه import در notebook

```python
# اضافه کردن src به Python path
import sys
sys.path.append('../../')  # از داخل notebooks/01_eda/

# import ماژول‌ها
from src.data.dataset import load_dataset, get_feature_cols, make_splits
from src.evaluation.metrics import compute_metrics, print_metrics
```

---

## قانون مهم

هر چیزی که بیش از یک بار استفاده می‌شود، باید اینجا باشد — نه در notebook.
مثلاً اگر در سه notebook مختلف داده را load می‌کنید، تابع load باید اینجا
تعریف شود و هر سه notebook آن را import کنند.
