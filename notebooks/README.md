# notebooks/

## این پوشه چیست؟

محل تمام **Jupyter Notebook های تحلیلی** پروژه.

تفاوت با `src/`:
- اینجا **اجرا** می‌شود — `src/` فقط تعریف می‌شود
- اینجا **visualization** است — نمودار، جدول، تفسیر
- اینجا **اکتشافی** است — سوال می‌پرسیم و جواب می‌گیریم

---

## ساختار

```
notebooks/
├── 01_eda/           ← فاز ۲: تحلیل اکتشافی داده
├── 02_baseline/      ← فاز ۳: مدل‌های baseline
├── 03_cross_modal/   ← فاز ۴: معماری cross-modal
└── 04_subgroup/      ← فاز ۵: کشف subgroup
```

---

## قرارداد نام‌گذاری notebook ها

```
{شماره}_{موضوع}.ipynb

مثال:
01_descriptive_stats.ipynb
02_distributions.ipynb
03_correlation.ipynb
```

شماره‌گذاری ترتیب اجرا را مشخص می‌کند.

---

## قرارداد داخل هر notebook

هر notebook باید این بخش‌ها را داشته باشد:

```
# 1. Imports و Setup
import sys
sys.path.append('../../')
from src.data.dataset import load_dataset, ...

# 2. بارگذاری داده
df = load_dataset('../../data/processed/nsclc_final.csv')

# 3. سوال تحقیقاتی
# این notebook چه سوالی را جواب می‌دهد؟

# 4. تحلیل
# ...

# 5. نتیجه‌گیری
# یافته‌های کلیدی چه بودند؟
# چه تصمیمی برای فاز بعد می‌گیریم؟
```

---

## نحوه اجرا

```bash
# از ریشه repository
jupyter notebook

# یا با JupyterLab
jupyter lab
```

**ترتیب اجرا:**
1. ابتدا تمام notebook های `01_eda/` را اجرا کنید
2. سپس `02_baseline/`
3. سپس `03_cross_modal/`
4. در نهایت `04_subgroup/`

---

## نکته مهم

Notebook های این پوشه به GitHub push می‌شوند — اما **output ها پاک شوند**
قبل از commit. این باعث می‌شود حجم repository کم بماند و اطلاعات
بیماران در خروجی‌ها leak نشود.

```bash
# پاک کردن output همه notebook ها قبل از commit
jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb
```
