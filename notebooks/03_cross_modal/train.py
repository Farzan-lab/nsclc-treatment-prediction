"""
================================================================================
Phase 4 — Cross-Modal Architecture
File: train.py
================================================================================

TRAINING PIPELINE:

    Loss = α × CE(fused_logits, y) + β × CE(clinical_logits, y) + λ × KD_Loss

    CE Loss:
        Standard cross-entropy on treatment labels.
        Applied to BOTH fused and clinical logits.
        This ensures both paths learn to predict treatment correctly.

    KD Loss (Knowledge Distillation):
        KL Divergence between soft probability distributions.
        Forces clinical_logits ≈ fused_logits.
        Temperature T softens the distributions — the model learns
        from relative similarities, not just the winning class.

        KD_Loss = T² × KLDiv(
            softmax(clinical_logits / T),
            softmax(fused_logits / T)
        )

        Why T² scaling?
            KL divergence of softened distributions is scaled by 1/T².
            Multiplying by T² restores the gradient magnitude to match CE loss.

RUN:
    python notebooks/03_cross_modal/train.py
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

# Add project root to path so we can import src/models/model.py
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.model import CrossModalModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR  = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/splits/')
RESULTS_DIR = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/experiments/04_cross_modal/')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
# Use GPU if available, otherwise CPU
# On most machines this will be CPU — still fast enough for this dataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ================================================================================
# SECTION 1: Configuration
# ================================================================================
# All hyperparameters in one place — easy to change and track
CONFIG = {
    # Model architecture
    'clinical_dim' : 18,
    'genomic_dim'  : 46,
    'embed_dim'    : 64,
    'n_heads'      : 4,
    'n_classes'    : 3,
    'dropout'      : 0.2,

    # Training — Staged approach
    'epochs'          : 120,
    'warmup_epochs'   : 30,     # Phase 1: train WITHOUT KD loss
                                 # model first learns to predict with genomic
                                 # Phase 2: add KD loss after warmup
    'batch_size'      : 64,
    'learning_rate'   : 0.0003,
    'weight_decay'    : 0.0001,
    'patience'        : 25,

    # Loss weights (applied only after warmup_epochs)
    'alpha'           : 1.0,    # CE on fused logits
    'beta'            : 0.5,    # CE on clinical logits
    'lambda_kd'       : 0.4,    # KD distillation loss
    'temperature'     : 3.0,

    # No class weighting — caused Immunotherapy distortion
    'class_weights'   : None,

    'random_state'    : 42,
}


# ================================================================================
# SECTION 2: Dataset Class
# ================================================================================
class NSCLCDataset(Dataset):
    """
    PyTorch Dataset for NSCLC treatment prediction.

    Wraps clinical features, genomic features, and labels into
    a format that DataLoader can iterate over in batches.

    Args:
        X_clinical : DataFrame or array of clinical features
        X_genomic  : DataFrame or array of genomic features (optional)
        y          : array of integer labels (0, 1, 2)
    """
    def __init__(self, X_clinical, y, X_genomic=None):
        # Convert to float32 tensors — PyTorch default for neural networks
        self.X_clinical = torch.tensor(
            X_clinical.values if hasattr(X_clinical, 'values') else X_clinical,
            dtype=torch.float32
        )
        self.y = torch.tensor(y, dtype=torch.long)

        self.X_genomic = None
        if X_genomic is not None:
            self.X_genomic = torch.tensor(
                X_genomic.values if hasattr(X_genomic, 'values') else X_genomic,
                dtype=torch.float32
            )

    def __len__(self):
        # Required by DataLoader — returns total number of samples
        return len(self.y)

    def __getitem__(self, idx):
        # Returns one sample at a time — DataLoader batches these automatically
        if self.X_genomic is not None:
            return self.X_clinical[idx], self.X_genomic[idx], self.y[idx]
        return self.X_clinical[idx], self.y[idx]


# ================================================================================
# SECTION 3: Loss Function
# ================================================================================
class CrossModalLoss(nn.Module):
    """
    Combined loss: CE (fused) + CE (clinical) + KD

    WHY THREE LOSS TERMS?

    CE on fused_logits:
        The primary training signal. Teaches the full model
        (clinical + genomic + attention) to predict treatment correctly.

    CE on clinical_logits:
        Ensures the clinical encoder ALONE also learns to predict treatment.
        Without this, the clinical encoder might only learn to produce embeddings
        that work well in the attention mechanism, but not on their own.

    KD Loss:
        Forces clinical_logits to mimic fused_logits (not just the label).
        This transfers "dark knowledge" — the relative confidence across all
        three classes, not just which class wins.

        Example:
            Label y = 0 (Immunotherapy)
            fused_logits  = [0.7, 0.2, 0.1]   ← 70% Immuno, some uncertainty
            clinical alone = [0.4, 0.3, 0.3]  ← uncertain without genomic

            CE loss only sees: predict class 0
            KD loss also sees: be 70% confident for class 0,
                               and learn that class 1 is slightly more likely
                               than class 2

        Temperature T=4 makes distributions softer:
            fused at T=4: softmax([0.7, 0.2, 0.1]/4) ≈ [0.38, 0.32, 0.30]
            More information in the "dark" probabilities for the model to learn from.
    """
    def __init__(self, alpha=1.0, beta=0.5, lambda_kd=0.5,
                 temperature=4.0, class_weights=None, device='cpu'):
        super().__init__()
        self.alpha       = alpha
        self.beta        = beta
        self.lambda_kd   = lambda_kd
        self.temperature = temperature

        # Weighted CE: give more weight to minority classes (Targeted)
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        # KL divergence: measures how much one distribution differs from another
        # reduction='batchmean': average over batch (standard for KD)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, fused_logits, clinical_logits, y):
        """
        Args:
            fused_logits   : (batch, 3) — from attention fusion
            clinical_logits: (batch, 3) — from clinical encoder only
            y              : (batch,) — true labels

        Returns:
            total_loss   : scalar
            loss_details : dict of individual loss components
        """
        T = self.temperature

        # CE loss on fused output — primary supervision signal
        ce_fused = self.ce_loss(fused_logits, y)

        # CE loss on clinical output — ensures clinical encoder learns independently
        ce_clinical = self.ce_loss(clinical_logits, y)

        # KD loss: soft targets from fused, soft predictions from clinical
        # log_softmax for student (clinical), softmax for teacher (fused)
        # KLDiv expects log-probabilities as input, probabilities as target
        soft_student = F.log_softmax(clinical_logits / T, dim=1)
        soft_teacher = F.softmax(fused_logits.detach() / T, dim=1)
        # .detach() is critical: we don't want gradients to flow back through
        # the teacher (fused) path during KD loss computation

        kd_loss = self.kl_div(soft_student, soft_teacher) * (T ** 2)

        # Combine all losses with their weights
        total = (self.alpha * ce_fused +
                 self.beta  * ce_clinical +
                 self.lambda_kd * kd_loss)

        return total, {
            'total'       : total.item(),
            'ce_fused'    : ce_fused.item(),
            'ce_clinical' : ce_clinical.item(),
            'kd_loss'     : kd_loss.item(),
        }


# ================================================================================
# SECTION 4: Evaluation Function
# ================================================================================
def evaluate_model(model, loader, device, mode='inference'):
    """
    Evaluate model on a DataLoader.

    mode='inference': uses only clinical features (production mode)
    mode='training':  uses fused logits (measures training performance)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x_clin, x_gen, y = batch
                x_clin = x_clin.to(device)
                x_gen  = x_gen.to(device)
                y      = y.to(device)
            else:
                x_clin, y = batch
                x_clin = x_clin.to(device)
                y      = y.to(device)
                x_gen  = None

            # Always predict using clinical-only path (inference mode)
            logits = model(x_clin, x_genomic=None)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'accuracy' : round(accuracy_score(all_labels, all_preds), 4),
        'macro_f1' : round(f1_score(all_labels, all_preds, average='macro'), 4),
        'per_class': f1_score(all_labels, all_preds, average=None).tolist(),
    }


# ================================================================================
# SECTION 5: Training Loop
# ================================================================================
def train(config):
    torch.manual_seed(config['random_state'])
    np.random.seed(config['random_state'])

    # ── Load Data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")

    X_train_clin = pd.read_csv(SPLITS_DIR / 'X_train_clinical.csv')
    X_val_clin   = pd.read_csv(SPLITS_DIR / 'X_val_clinical.csv')
    X_test_clin  = pd.read_csv(SPLITS_DIR / 'X_test_clinical.csv')

    X_train_all  = pd.read_csv(SPLITS_DIR / 'X_train_all.csv')
    X_val_all    = pd.read_csv(SPLITS_DIR / 'X_val_all.csv')
    X_test_all   = pd.read_csv(SPLITS_DIR / 'X_test_all.csv')

    y_train = np.load(SPLITS_DIR / 'y_train.npy')
    y_val   = np.load(SPLITS_DIR / 'y_val.npy')
    y_test  = np.load(SPLITS_DIR / 'y_test.npy')

    # Extract genomic columns (all columns not in clinical)
    clinical_cols = X_train_clin.columns.tolist()
    all_cols      = X_train_all.columns.tolist()
    genomic_cols  = [c for c in all_cols if c not in clinical_cols]

    X_train_gen = X_train_all[genomic_cols]
    X_val_gen   = X_val_all[genomic_cols]
    X_test_gen  = X_test_all[genomic_cols]

    config['clinical_dim'] = len(clinical_cols)
    config['genomic_dim']  = len(genomic_cols)

    print(f"  Clinical features : {len(clinical_cols)}")
    print(f"  Genomic features  : {len(genomic_cols)}")
    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ── DataLoaders ─────────────────────────────────────────────────────────────
    train_ds = NSCLCDataset(X_train_clin, y_train, X_train_gen)
    val_ds   = NSCLCDataset(X_val_clin,   y_val,   X_val_gen)
    test_ds  = NSCLCDataset(X_test_clin,  y_test,  X_test_gen)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256,
                              shuffle=False, num_workers=0)

    # ── Model ───────────────────────────────────────────────────────────────────
    model = CrossModalModel(
        clinical_dim=config['clinical_dim'],
        genomic_dim=config['genomic_dim'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        n_classes=config['n_classes'],
        dropout=config['dropout'],
    ).to(DEVICE)

    print(f"\n  Model parameters: {model.count_parameters():,}")

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────────
    criterion = CrossModalLoss(
        alpha=config['alpha'],
        beta=config['beta'],
        lambda_kd=config['lambda_kd'],
        temperature=config['temperature'],
        class_weights=config['class_weights'],
        device=DEVICE,
    )

    # AdamW: Adam optimizer with decoupled weight decay
    # Better regularization than standard Adam for neural networks
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # Cosine annealing: lr starts at learning_rate, smoothly decreases to 0
    # Helps fine-tune in later epochs without oscillating around the optimum
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-5,
    )

    # ── Training Loop ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING — {config['epochs']} epochs, patience={config['patience']}")
    print(f"{'='*65}")
    print(f"\n  {'Epoch':>6} {'Train Loss':>12} {'Val F1':>8} {'Best':>6} {'LR':>10}")
    print(f"  {'─'*50}")

    history        = []
    best_val_f1    = 0.0
    best_epoch     = 0
    patience_count = 0

    for epoch in range(1, config['epochs'] + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_losses = []

        # Staged training:
        # Phase 1 (warmup): train only with CE on fused — no KD pressure
        # Phase 2 (after warmup): add KD loss so clinical learns from fused
        use_kd = epoch > config['warmup_epochs']

        for batch in train_loader:
            x_clin, x_gen, y = batch
            x_clin = x_clin.to(DEVICE)
            x_gen  = x_gen.to(DEVICE)
            y      = y.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass — training mode uses both encoders
            fused_logits, clinical_logits = model(x_clin, x_gen)

            # Phase 1: only CE on fused (KD disabled)
            # Phase 2: full loss including KD
            if not use_kd:
                ce_loss_fn = nn.CrossEntropyLoss()
                loss = ce_loss_fn(fused_logits, y)
                loss_details = {'total': loss.item(), 'ce_fused': loss.item(),
                                'ce_clinical': 0, 'kd_loss': 0}
            else:
                loss, loss_details = criterion(fused_logits, clinical_logits, y)

            # Backward pass — compute gradients
            loss.backward()

            # Gradient clipping: prevents exploding gradients
            # If gradient norm > 1.0, scale it down to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss_details['total'])

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────────
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        val_f1      = val_metrics['macro_f1']
        avg_loss    = np.mean(epoch_losses)
        current_lr  = scheduler.get_last_lr()[0]

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience_count = 0
            # Save best model checkpoint
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_f1'     : val_f1,
                'config'     : config,
            }, RESULTS_DIR / 'best_model.pt')
        else:
            patience_count += 1

        # Log every 5 epochs
        if epoch % 5 == 0 or is_best:
            best_marker = " ★" if is_best else ""
            kd_marker   = " [KD]" if use_kd else " [warm]"
            print(f"  {epoch:>6} {avg_loss:>12.4f} {val_f1:>8.4f} "
                  f"{best_val_f1:>6.4f} {current_lr:>10.6f}{best_marker}{kd_marker}")

        history.append({
            'epoch'   : epoch,
            'loss'    : avg_loss,
            'val_f1'  : val_f1,
            'lr'      : current_lr,
        })

        # Early stopping
        if patience_count >= config['patience']:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {config['patience']} epochs)")
            break

    # ── Final Evaluation ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL EVALUATION (best model from epoch {best_epoch})")
    print(f"{'='*65}")

    # Load best checkpoint
    checkpoint = torch.load(RESULTS_DIR / 'best_model.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])

    val_metrics  = evaluate_model(model, val_loader,  DEVICE)
    test_metrics = evaluate_model(model, test_loader, DEVICE)

    CLASS_NAMES = ['Immunotherapy', 'Chemotherapy', 'Targeted']

    print(f"\n  {'Metric':<20} {'Val':>8} {'Test':>8}")
    print(f"  {'─'*38}")
    print(f"  {'accuracy':<20} {val_metrics['accuracy']:>8.4f} "
          f"{test_metrics['accuracy']:>8.4f}")
    print(f"  {'macro_f1':<20} {val_metrics['macro_f1']:>8.4f} "
          f"{test_metrics['macro_f1']:>8.4f}")

    print(f"\n  Per-class F1 (test — clinical only inference):")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls:<20} {test_metrics['per_class'][i]:.4f}")

    # ── Gap Analysis ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  COMPARISON WITH BASELINES")
    print(f"{'='*65}")

    floor   = 0.5584   # clinical-only XGBoost from Phase 3
    ceiling = 0.6555   # all-features LR from Phase 3
    ours    = test_metrics['macro_f1']
    gap_closed = (ours - floor) / (ceiling - floor) * 100 if ceiling > floor else 0

    print(f"\n  Floor   (clinical only baseline) : {floor:.4f}")
    print(f"  Ceiling (all features baseline)  : {ceiling:.4f}")
    print(f"  Ours    (cross-modal distillation): {ours:.4f}")
    print(f"\n  Gap closed: {gap_closed:.1f}%")

    if ours > ceiling:
        print(f"  ★ Exceeded ceiling! Better than all-features baseline.")
    elif ours > floor + (ceiling - floor) * 0.5:
        print(f"  ✓ Closed more than 50% of the gap — strong result")
    elif ours > floor:
        print(f"  ⚡ Improved over floor but gap closure is modest")
    else:
        print(f"  ✗ Did not improve over clinical-only baseline")

    # ── Save Results ────────────────────────────────────────────────────────────
    results = {
        'config'      : config,
        'best_epoch'  : best_epoch,
        'best_val_f1' : best_val_f1,
        'val_metrics' : val_metrics,
        'test_metrics': test_metrics,
        'floor'       : floor,
        'ceiling'     : ceiling,
        'gap_closed_pct': gap_closed,
        'history'     : history,
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to experiments/04_cross_modal/")
    print(f"✓ Best model saved to experiments/04_cross_modal/best_model.pt")

    return results


# ================================================================================
# MAIN
# ================================================================================
if __name__ == '__main__':
    start = time.time()
    results = train(CONFIG)
    elapsed = time.time() - start
    print(f"\n  Total training time: {elapsed/60:.1f} minutes")
