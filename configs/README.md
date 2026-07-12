# configs/

## این پوشه چیست؟

این پوشه تمام **تنظیمات و hyperparameter های پروژه** را نگه می‌دارد.

هدف اصلی: **جدا کردن اعداد و تنظیمات از کد**.
به جای اینکه learning rate یا تعداد لایه‌ها را مستقیم داخل کد بنویسیم،
آن‌ها را اینجا تعریف می‌کنیم. این یعنی برای تغییر یک عدد، نیازی به
دست زدن به کد Python نیست.

---

## فایل‌ها

### `base.yaml` — تنظیمات پایه
تنظیماتی که **بین همه فازها مشترک** هستند:
- مسیر فایل داده
- پارامترهای تقسیم داده (تعداد fold، نسبت test، random seed)
- ابعاد feature ها
- مسیر ذخیره نتایج و figures

هر فایل config دیگری از این فایل ارث‌بری می‌کند.

---

### `baseline.yaml` — تنظیمات فاز ۳
تنظیمات مخصوص **مدل‌های baseline** (فاز ۳):
- Logistic Regression: مقادیر C برای grid search
- Random Forest: تعداد درخت‌ها، عمق
- XGBoost: learning rate، تعداد estimator، عمق

این فایل مشخص می‌کند که grid search چه مقادیری را امتحان کند.

---

### `cross_modal.yaml` — تنظیمات فاز ۴
تنظیمات مخصوص **معماری Cross-Modal** (فاز ۴):

**Clinical Encoder:**
- ابعاد لایه‌های hidden
- نرخ dropout
- نوع normalization (layer norm)

**Genomic Encoder:**
- ابعاد لایه‌های hidden
- نرخ dropout
- نوع normalization (batch norm)

**Fusion (Cross-Attention):**
- بعد embedding
- تعداد attention head
- dropout

**Knowledge Distillation:**
- temperature: کنترل نرمی soft labels
- lambda_kd: وزن distillation loss در مقابل classification loss

**Training:**
- تعداد epochs
- batch size
- learning rate و weight decay
- patience برای early stopping
- نوع scheduler

---

## نحوه استفاده در کد

```python
import yaml

# بارگذاری config
with open('configs/cross_modal.yaml', 'r') as f:
    config = yaml.safe_load(f)

# دسترسی به تنظیمات
lr = config['training']['learning_rate']
embed_dim = config['model']['fusion']['embed_dim']
lambda_kd = config['model']['distillation']['lambda_kd']
```

---

## چه زمانی این فایل‌ها را تغییر می‌دهیم؟

| موقعیت | فایل |
|--------|------|
| تغییر مسیر داده | `base.yaml` |
| تنظیم hyperparameter های baseline | `baseline.yaml` |
| تنظیم معماری neural network | `cross_modal.yaml` |
| اضافه کردن experiment جدید | یک فایل YAML جدید بسازید |

---

## نکته مهم

هرگز مقادیر حساس (API key، password) را در این فایل‌ها ننویسید.
برای آن‌ها از فایل `.env` استفاده کنید که gitignore شده است.
