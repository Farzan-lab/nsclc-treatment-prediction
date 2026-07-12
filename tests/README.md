# tests/

## این پوشه چیست؟

محل **unit test های پروژه** — کدهایی که صحت ماژول‌های `src/` را تست می‌کنند.

الان خالی است — در فازهای بعد پر می‌شود.

---

## چرا test مهم است؟

در پروژه تحقیقاتی، یک باگ کوچک می‌تواند نتایج کل مقاله را خراب کند.
مثلاً اگر data leakage (نشت اطلاعات test به train) وجود داشته باشد
و آن را پیدا نکنیم، نتایجمان unrealistic خواهد بود.

---

## فایل‌هایی که ساخته می‌شوند

### `test_dataset.py` — تست ماژول داده

```python
def test_split_no_leakage():
    """مطمئن می‌شود که هیچ بیماری در هم train و هم test نیست."""
    splits = make_splits(df)
    train_ids = set(splits['train']['PATIENT_ID'])
    test_ids  = set(splits['test']['PATIENT_ID'])
    assert len(train_ids & test_ids) == 0  # intersection خالی باشد

def test_split_stratification():
    """نسبت کلاس‌ها در train و test مشابه کل dataset است."""
    ...

def test_feature_cols_count():
    """تعداد clinical features دقیقاً 16 است."""
    cols = get_feature_cols(df, include_genomic=False)
    assert len(cols) == 16
```

---

### `test_metrics.py` — تست معیارها

```python
def test_perfect_prediction():
    """با پیش‌بینی کامل، همه معیارها باید 1.0 باشند."""
    y = [0, 1, 2, 0, 1]
    metrics = compute_metrics(y, y)
    assert metrics['accuracy'] == 1.0
    assert metrics['macro_f1'] == 1.0

def test_random_prediction():
    """پیش‌بینی تصادفی باید AUC نزدیک 0.5 داشته باشد."""
    ...
```

---

### `test_models.py` — تست معماری (فاز ۴)

```python
def test_output_shape():
    """خروجی مدل باید shape (batch_size, 3) داشته باشد."""
    model = CrossModalModel(config)
    x_clinical = torch.randn(32, 16)
    x_genomic  = torch.randn(32, 50)
    output = model(x_clinical, x_genomic)
    assert output.shape == (32, 3)

def test_inference_without_genomic():
    """مدل باید بدون genomic هم کار کند."""
    model = CrossModalModel(config)
    x_clinical = torch.randn(32, 16)
    output = model(x_clinical)  # بدون genomic
    assert output.shape == (32, 3)
```

---

## نحوه اجرای test ها

```bash
# اجرای همه test ها
pytest tests/

# اجرای یک فایل خاص
pytest tests/test_dataset.py

# با جزئیات بیشتر
pytest tests/ -v

# با coverage report
pytest tests/ --cov=src
```

---

## قانون

هر تابع مهمی که به `src/` اضافه می‌شود،
باید حداقل یک test در این پوشه داشته باشد.
