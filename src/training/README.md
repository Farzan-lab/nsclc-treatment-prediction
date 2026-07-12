# src/training/

## این پوشه چیست؟

همه کدهای مربوط به **آموزش مدل‌های neural network** اینجاست.

الان خالی است — در **فاز ۴** پر می‌شود.

---

## فایل‌هایی که ساخته می‌شوند

### `losses.py` — توابع Loss

**TreatmentClassificationLoss:**
Cross-Entropy معمولی برای پیش‌بینی سه کلاس درمان.

```python
ce_loss = CrossEntropyLoss()(predictions, true_labels)
```

**DistillationLoss:**
این loss قلب Knowledge Distillation است.
ClinicalEncoder را مجبور می‌کند خروجیش شبیه fused embedding باشد.

```python
# KL Divergence بین soft predictions دو مدل
kd_loss = KLDivLoss()(
    softmax(clinical_logits / temperature),
    softmax(fused_logits / temperature)
)
```

**temperature چیست:**
عدد بزرگ‌تر یعنی soft labels نرم‌تر — مدل اطلاعات بیشتری
از توزیع احتمال یاد می‌گیرد نه فقط از کلاس برنده.

**CombinedLoss:**
ترکیب دو loss بالا با وزن:
```
total_loss = ce_loss + lambda_kd × kd_loss
```
مقدار `lambda_kd` در `configs/cross_modal.yaml` تعریف شده.

---

### `trainer.py` — حلقه آموزش

**Trainer class:**
مدیریت کامل فرایند آموزش:

```python
trainer = Trainer(model, config)
trainer.fit(train_loader, val_loader)
results = trainer.evaluate(test_loader)
```

**چه کارهایی انجام می‌دهد:**
- حلقه epoch → batch → forward → loss → backward → update
- محاسبه metrics در هر epoch (train و val)
- Early stopping: اگر val loss بهبود نیافت، آموزش متوقف می‌شود
- ذخیره بهترین checkpoint
- لاگ کردن نتایج

**Early Stopping چیست:**
اگر بعد از `patience` epoch متوالی val loss کاهش نیافت،
آموزش متوقف می‌شود تا از overfitting جلوگیری شود.

---

### `schedulers.py` — Learning Rate Scheduling

**CosineAnnealingLR:**
Learning rate را به صورت cosine کاهش می‌دهد.
در ابتدا بزرگ است و تدریجاً کوچک می‌شود — یاددهی سریع در ابتدا،
تنظیم دقیق در انتها.

```
epoch 0:   lr = 0.001  (maximum)
epoch 50:  lr = 0.0005 (middle)
epoch 100: lr = 0.0001 (minimum)
```

---

### `checkpoint.py` — ذخیره و بارگذاری مدل

```python
# ذخیره بهترین مدل
save_checkpoint(model, optimizer, epoch, val_loss, path)

# بارگذاری برای inference
model = load_checkpoint(path)
```

چه چیزی ذخیره می‌شود:
- وزن‌های مدل
- وضعیت optimizer
- شماره epoch
- بهترین val loss
- config مورد استفاده

---

## جریان کلی آموزش

```
for epoch in range(max_epochs):

    # Training
    for batch in train_loader:
        clinical, genomic, labels = batch
        
        clinical_emb = clinical_encoder(clinical)
        genomic_emb  = genomic_encoder(genomic)
        fused_emb    = fusion(clinical_emb, genomic_emb)
        
        ce_loss = classification_loss(fused_emb, labels)
        kd_loss = distillation_loss(clinical_emb, fused_emb)
        loss = ce_loss + lambda_kd * kd_loss
        
        loss.backward()
        optimizer.step()
    
    # Validation (فقط clinical encoder)
    for batch in val_loader:
        clinical, labels = batch
        clinical_emb = clinical_encoder(clinical)
        predictions = classifier(clinical_emb)
        val_loss = ce_loss(predictions, labels)
    
    # Early stopping check
    if val_loss improved:
        save_checkpoint()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```
