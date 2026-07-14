"""
================================================================================
Phase 4 — Cross-Modal (Method 1: Proper Teacher-Student Distillation)
File: train_teacher_student.py
================================================================================

THREE-STAGE APPROACH:

    STAGE 1 — Train Teacher (clinical + genomic)
        Full model with both modalities.
        Only CE loss. Goal: strongest possible fused model (~0.65 target).
        This is our "teacher" with access to privileged genomic info.

    STAGE 2 — Freeze Teacher
        Teacher weights are locked. It becomes a fixed knowledge source.

    STAGE 3 — Train Student (clinical only) via distillation
        Student = clinical encoder + classifier only.
        Learns from BOTH:
          - true labels (CE loss)
          - teacher's soft predictions (KD loss)
        Teacher provides "dark knowledge" the student can't get from labels alone.

WHY THIS IS BETTER THAN JOINT TRAINING:
    In joint training, the teacher (fused path) and student (clinical path)
    train simultaneously — the teacher never becomes strong before the student
    starts mimicking it. Here, we build a strong teacher FIRST, then distill.
================================================================================
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.model import (
    ClinicalEncoder, GenomicEncoder, CrossAttentionFusion, ClassifierHead
)

SPLITS_DIR  = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/splits/')
RESULTS_DIR = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/experiments/04_teacher_student/')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ================================================================================
# CONFIG
# ================================================================================
CONFIG = {
    'embed_dim'       : 64,
    'n_heads'         : 4,
    'n_classes'       : 3,
    'dropout'         : 0.2,

    # Teacher training
    'teacher_epochs'  : 80,
    'teacher_lr'      : 0.0005,

    # Student training
    'student_epochs'  : 100,
    'student_lr'      : 0.0005,

    'batch_size'      : 64,
    'weight_decay'    : 0.0001,
    'patience'        : 25,

    # Distillation
    'lambda_kd'       : 0.7,    # weight of KD vs CE for student
    'temperature'     : 4.0,

    'random_state'    : 42,
}

# ================================================================================
# TEACHER MODEL — clinical + genomic + fusion
# ================================================================================
class TeacherModel(nn.Module):
    """Full model with access to genomic data. This is the knowledge source."""
    def __init__(self, clinical_dim, genomic_dim, embed_dim, n_heads,
                 n_classes, dropout):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(clinical_dim, embed_dim, dropout)
        self.genomic_encoder  = GenomicEncoder(genomic_dim, embed_dim, dropout)
        self.fusion           = CrossAttentionFusion(embed_dim, n_heads, dropout)
        self.classifier       = ClassifierHead(embed_dim, n_classes, dropout)

    def forward(self, x_clinical, x_genomic):
        clin_emb  = self.clinical_encoder(x_clinical)
        gen_emb   = self.genomic_encoder(x_genomic)
        fused_emb = self.fusion(clin_emb, gen_emb)
        return self.classifier(fused_emb)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ================================================================================
# STUDENT MODEL — clinical only
# ================================================================================
class StudentModel(nn.Module):
    """Clinical-only model. Learns from teacher via distillation."""
    def __init__(self, clinical_dim, embed_dim, n_classes, dropout):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(clinical_dim, embed_dim, dropout)
        self.classifier       = ClassifierHead(embed_dim, n_classes, dropout)

    def forward(self, x_clinical):
        emb = self.clinical_encoder(x_clinical)
        return self.classifier(emb)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ================================================================================
# DATASET
# ================================================================================
class NSCLCDataset(Dataset):
    def __init__(self, X_clin, y, X_gen=None):
        self.X_clin = torch.tensor(X_clin.values, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.long)
        self.X_gen  = torch.tensor(X_gen.values, dtype=torch.float32) if X_gen is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.X_gen is not None:
            return self.X_clin[idx], self.X_gen[idx], self.y[idx]
        return self.X_clin[idx], self.y[idx]

# ================================================================================
# EVALUATION
# ================================================================================
def eval_teacher(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_clin, x_gen, y in loader:
            x_clin, x_gen = x_clin.to(DEVICE), x_gen.to(DEVICE)
            logits = model(x_clin, x_gen)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labels.extend(y.numpy())
    return {
        'accuracy': round(accuracy_score(labels, preds), 4),
        'macro_f1': round(f1_score(labels, preds, average='macro'), 4),
        'per_class': f1_score(labels, preds, average=None).tolist(),
    }

def eval_student(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x_clin = batch[0].to(DEVICE)
            y      = batch[-1]
            logits = model(x_clin)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labels.extend(y.numpy())
    return {
        'accuracy': round(accuracy_score(labels, preds), 4),
        'macro_f1': round(f1_score(labels, preds, average='macro'), 4),
        'per_class': f1_score(labels, preds, average=None).tolist(),
    }

# ================================================================================
# MAIN
# ================================================================================
def main():
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])

    # ── Load Data ──────────────────────────────────────────────────────────────
    X_train_clin = pd.read_csv(SPLITS_DIR / 'X_train_clinical.csv')
    X_val_clin   = pd.read_csv(SPLITS_DIR / 'X_val_clinical.csv')
    X_test_clin  = pd.read_csv(SPLITS_DIR / 'X_test_clinical.csv')
    X_train_all  = pd.read_csv(SPLITS_DIR / 'X_train_all.csv')
    X_val_all    = pd.read_csv(SPLITS_DIR / 'X_val_all.csv')
    X_test_all   = pd.read_csv(SPLITS_DIR / 'X_test_all.csv')
    y_train = np.load(SPLITS_DIR / 'y_train.npy')
    y_val   = np.load(SPLITS_DIR / 'y_val.npy')
    y_test  = np.load(SPLITS_DIR / 'y_test.npy')

    clin_cols = X_train_clin.columns.tolist()
    gen_cols  = [c for c in X_train_all.columns if c not in clin_cols]
    X_train_gen = X_train_all[gen_cols]
    X_val_gen   = X_val_all[gen_cols]
    X_test_gen  = X_test_all[gen_cols]

    clinical_dim = len(clin_cols)
    genomic_dim  = len(gen_cols)

    train_ds = NSCLCDataset(X_train_clin, y_train, X_train_gen)
    val_ds   = NSCLCDataset(X_val_clin,   y_val,   X_val_gen)
    test_ds  = NSCLCDataset(X_test_clin,  y_test,  X_test_gen)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    print(f"Clinical: {clinical_dim}, Genomic: {genomic_dim}")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ════════════════════════════════════════════════════════════════════════════
    # STAGE 1: TRAIN TEACHER
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  STAGE 1: TRAINING TEACHER (clinical + genomic)")
    print(f"{'='*65}")

    teacher = TeacherModel(clinical_dim, genomic_dim, CONFIG['embed_dim'],
                           CONFIG['n_heads'], CONFIG['n_classes'],
                           CONFIG['dropout']).to(DEVICE)
    print(f"  Teacher parameters: {teacher.count_parameters():,}")

    optimizer = AdamW(teacher.parameters(), lr=CONFIG['teacher_lr'],
                      weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['teacher_epochs'], eta_min=1e-5)
    ce_loss   = nn.CrossEntropyLoss()

    best_teacher_f1 = 0
    patience_count  = 0

    print(f"\n  {'Epoch':>6} {'Loss':>10} {'Val F1':>8} {'Best':>7}")
    print(f"  {'─'*38}")

    for epoch in range(1, CONFIG['teacher_epochs'] + 1):
        teacher.train()
        losses = []
        for x_clin, x_gen, y in train_loader:
            x_clin, x_gen, y = x_clin.to(DEVICE), x_gen.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = teacher(x_clin, x_gen)
            loss = ce_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_m = eval_teacher(teacher, val_loader)
        is_best = val_m['macro_f1'] > best_teacher_f1
        if is_best:
            best_teacher_f1 = val_m['macro_f1']
            patience_count = 0
            torch.save(teacher.state_dict(), RESULTS_DIR / 'teacher.pt')
        else:
            patience_count += 1

        if epoch % 5 == 0 or is_best:
            marker = " ★" if is_best else ""
            print(f"  {epoch:>6} {np.mean(losses):>10.4f} "
                  f"{val_m['macro_f1']:>8.4f} {best_teacher_f1:>7.4f}{marker}")

        if patience_count >= CONFIG['patience']:
            print(f"  Early stop at epoch {epoch}")
            break

    # Load best teacher and evaluate on test
    teacher.load_state_dict(torch.load(RESULTS_DIR / 'teacher.pt'))
    teacher_test = eval_teacher(teacher, test_loader)
    print(f"\n  Teacher TEST macro_f1: {teacher_test['macro_f1']:.4f}")
    print(f"  (this is our knowledge source — should be close to XGBoost 0.65)")

    # ════════════════════════════════════════════════════════════════════════════
    # STAGE 2: FREEZE TEACHER
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  STAGE 2: FREEZING TEACHER")
    print(f"{'='*65}")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"  ✓ Teacher frozen — weights locked")

    # ════════════════════════════════════════════════════════════════════════════
    # STAGE 3: TRAIN STUDENT VIA DISTILLATION
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  STAGE 3: TRAINING STUDENT (clinical only, distilled)")
    print(f"{'='*65}")

    student = StudentModel(clinical_dim, CONFIG['embed_dim'],
                           CONFIG['n_classes'], CONFIG['dropout']).to(DEVICE)
    print(f"  Student parameters: {student.count_parameters():,}")

    optimizer = AdamW(student.parameters(), lr=CONFIG['student_lr'],
                      weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['student_epochs'], eta_min=1e-5)

    T          = CONFIG['temperature']
    lambda_kd  = CONFIG['lambda_kd']
    ce_loss    = nn.CrossEntropyLoss()
    kl_loss    = nn.KLDivLoss(reduction='batchmean')

    best_student_f1 = 0
    patience_count  = 0
    history = []

    print(f"\n  {'Epoch':>6} {'Loss':>10} {'Val F1':>8} {'Best':>7}")
    print(f"  {'─'*38}")

    for epoch in range(1, CONFIG['student_epochs'] + 1):
        student.train()
        losses = []
        for x_clin, x_gen, y in train_loader:
            x_clin, x_gen, y = x_clin.to(DEVICE), x_gen.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            # Student prediction (clinical only)
            student_logits = student(x_clin)

            # Teacher prediction (with genomic) — no gradient
            with torch.no_grad():
                teacher_logits = teacher(x_clin, x_gen)

            # CE loss: student learns true labels
            loss_ce = ce_loss(student_logits, y)

            # KD loss: student mimics teacher's soft predictions
            soft_student = F.log_softmax(student_logits / T, dim=1)
            soft_teacher = F.softmax(teacher_logits / T, dim=1)
            loss_kd = kl_loss(soft_student, soft_teacher) * (T ** 2)

            # Combined: balance between learning labels and mimicking teacher
            loss = (1 - lambda_kd) * loss_ce + lambda_kd * loss_kd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_m = eval_student(student, val_loader)
        is_best = val_m['macro_f1'] > best_student_f1
        if is_best:
            best_student_f1 = val_m['macro_f1']
            patience_count = 0
            torch.save(student.state_dict(), RESULTS_DIR / 'student.pt')
        else:
            patience_count += 1

        if epoch % 5 == 0 or is_best:
            marker = " ★" if is_best else ""
            print(f"  {epoch:>6} {np.mean(losses):>10.4f} "
                  f"{val_m['macro_f1']:>8.4f} {best_student_f1:>7.4f}{marker}")

        history.append({'epoch': epoch, 'val_f1': val_m['macro_f1']})

        if patience_count >= CONFIG['patience']:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Final Evaluation ────────────────────────────────────────────────────────
    student.load_state_dict(torch.load(RESULTS_DIR / 'student.pt'))
    student_test = eval_student(student, test_loader)

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")

    CLASS_NAMES = ['Immunotherapy', 'Chemotherapy', 'Targeted']
    floor, ceiling = 0.5584, 0.6555
    ours = student_test['macro_f1']
    gap_closed = (ours - floor) / (ceiling - floor) * 100

    print(f"\n  Teacher (clin+gen)  test F1: {teacher_test['macro_f1']:.4f}")
    print(f"  Student (clin only) test F1: {ours:.4f}")
    print(f"\n  Student per-class F1:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls:<20} {student_test['per_class'][i]:.4f}")

    print(f"\n  {'─'*50}")
    print(f"  Floor   (clinical baseline)  : {floor:.4f}")
    print(f"  Ceiling (all-features)       : {ceiling:.4f}")
    print(f"  Ours    (distilled student)  : {ours:.4f}")
    print(f"  Gap closed                   : {gap_closed:.1f}%")

    if ours > floor:
        print(f"\n  ✓ Improved over clinical baseline by {(ours-floor)*100:.2f} points")
    else:
        print(f"\n  ✗ Did not beat clinical baseline")

    results = {
        'config': CONFIG,
        'teacher_test': teacher_test,
        'student_test': student_test,
        'floor': floor, 'ceiling': ceiling,
        'gap_closed_pct': gap_closed,
        'history': history,
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved to experiments/04_teacher_student/")

    return results


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"\n  Time: {(time.time()-start)/60:.1f} min")
