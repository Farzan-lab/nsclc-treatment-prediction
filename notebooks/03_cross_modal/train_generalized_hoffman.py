"""
Phase 4 - Generalized Distillation + Hoffman-style Feature Alignment.

This experiment tests whether a clinical-only student can benefit from a
privileged clinical+genomic teacher at training time.

Loss:
    L = (1 - lambda) * CE(student_logits, y)
        + lambda * (KD_T(student_logits, teacher_logits)
                    + feature_weight * MSE(norm(student_emb), norm(teacher_fused_emb)))

The feature alignment term follows the spirit of Hoffman et al.'s hallucination
loss: the inference-time stream is encouraged to match an intermediate
representation from a privileged-modality stream.
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.model import (  # noqa: E402
    ClassifierHead,
    ClinicalEncoder,
    CrossAttentionFusion,
    GenomicEncoder,
)


ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "experiments" / "04_generalized_hoffman"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Immunotherapy", "Chemotherapy", "Targeted"]


DEFAULT_CONFIG = {
    "embed_dim": 64,
    "n_heads": 4,
    "n_classes": 3,
    "dropout": 0.2,
    "teacher_epochs": 80,
    "student_epochs": 100,
    "teacher_lr": 5e-4,
    "student_lr": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "eval_batch_size": 256,
    "patience": 20,
    "temperature": 4.0,
    "feature_weight": 1.0,
    "lambda_grid": [0.1, 0.25, 0.5, 0.75, 0.9],
    "seeds": [7, 13, 21, 42, 101],
}


class NSCLCDataset(Dataset):
    def __init__(self, x_clinical, y, x_genomic=None):
        self.x_clinical = torch.tensor(x_clinical.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.x_genomic = (
            torch.tensor(x_genomic.values, dtype=torch.float32)
            if x_genomic is not None
            else None
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.x_genomic is None:
            return self.x_clinical[idx], self.y[idx]
        return self.x_clinical[idx], self.x_genomic[idx], self.y[idx]


class TeacherModel(nn.Module):
    def __init__(self, clinical_dim, genomic_dim, config):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(
            clinical_dim, config["embed_dim"], config["dropout"]
        )
        self.genomic_encoder = GenomicEncoder(
            genomic_dim, config["embed_dim"], config["dropout"]
        )
        self.fusion = CrossAttentionFusion(
            config["embed_dim"], config["n_heads"], config["dropout"]
        )
        self.classifier = ClassifierHead(
            config["embed_dim"], config["n_classes"], config["dropout"]
        )

    def forward(self, x_clinical, x_genomic, return_features=False):
        clinical_emb = self.clinical_encoder(x_clinical)
        genomic_emb = self.genomic_encoder(x_genomic)
        fused_emb = self.fusion(clinical_emb, genomic_emb)
        logits = self.classifier(fused_emb)
        if not return_features:
            return logits
        return {
            "logits": logits,
            "clinical_emb": clinical_emb,
            "genomic_emb": genomic_emb,
            "fused_emb": fused_emb,
        }


class StudentModel(nn.Module):
    def __init__(self, clinical_dim, config):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(
            clinical_dim, config["embed_dim"], config["dropout"]
        )
        self.classifier = ClassifierHead(
            config["embed_dim"], config["n_classes"], config["dropout"]
        )

    def forward(self, x_clinical, return_features=False):
        emb = self.clinical_encoder(x_clinical)
        logits = self.classifier(emb)
        if not return_features:
            return logits
        return {"logits": logits, "emb": emb}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_CONFIG["seeds"])
    parser.add_argument(
        "--lambda-grid", nargs="+", type=float, default=DEFAULT_CONFIG["lambda_grid"]
    )
    parser.add_argument("--teacher-epochs", type=int, default=DEFAULT_CONFIG["teacher_epochs"])
    parser.add_argument("--student-epochs", type=int, default=DEFAULT_CONFIG["student_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--feature-weight", type=float, default=DEFAULT_CONFIG["feature_weight"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed seed runs from results_partial.json or results.json.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    x_train_clin = pd.read_csv(SPLITS_DIR / "X_train_clinical.csv")
    x_val_clin = pd.read_csv(SPLITS_DIR / "X_val_clinical.csv")
    x_test_clin = pd.read_csv(SPLITS_DIR / "X_test_clinical.csv")
    x_train_all = pd.read_csv(SPLITS_DIR / "X_train_all.csv")
    x_val_all = pd.read_csv(SPLITS_DIR / "X_val_all.csv")
    x_test_all = pd.read_csv(SPLITS_DIR / "X_test_all.csv")
    y_train = np.load(SPLITS_DIR / "y_train.npy")
    y_val = np.load(SPLITS_DIR / "y_val.npy")
    y_test = np.load(SPLITS_DIR / "y_test.npy")

    clinical_cols = x_train_clin.columns.tolist()
    genomic_cols = [col for col in x_train_all.columns if col not in clinical_cols]

    data = {
        "x_train_clin": x_train_clin,
        "x_val_clin": x_val_clin,
        "x_test_clin": x_test_clin,
        "x_train_gen": x_train_all[genomic_cols],
        "x_val_gen": x_val_all[genomic_cols],
        "x_test_gen": x_test_all[genomic_cols],
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "clinical_dim": len(clinical_cols),
        "genomic_dim": len(genomic_cols),
    }
    return data


def make_loaders(data, config, seed, include_genomic):
    if include_genomic:
        train_ds = NSCLCDataset(data["x_train_clin"], data["y_train"], data["x_train_gen"])
        val_ds = NSCLCDataset(data["x_val_clin"], data["y_val"], data["x_val_gen"])
        test_ds = NSCLCDataset(data["x_test_clin"], data["y_test"], data["x_test_gen"])
    else:
        train_ds = NSCLCDataset(data["x_train_clin"], data["y_train"])
        val_ds = NSCLCDataset(data["x_val_clin"], data["y_val"])
        test_ds = NSCLCDataset(data["x_test_clin"], data["y_test"])

    generator = torch.Generator()
    generator.manual_seed(seed)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=0,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
        ),
    }


def metrics_from_predictions(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "per_class": [round(x, 4) for x in per_class],
    }


def evaluate_teacher(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_clin, x_gen, y in loader:
            logits = model(x_clin.to(DEVICE), x_gen.to(DEVICE))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.numpy())
    return metrics_from_predictions(np.array(labels), np.array(preds))


def evaluate_student(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x_clin = batch[0].to(DEVICE)
            y = batch[-1]
            logits = model(x_clin)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.numpy())
    return metrics_from_predictions(np.array(labels), np.array(preds))


def fit_with_early_stopping(model, loaders, config, epochs, lr, train_step, eval_fn):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_state = None
    best_val = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss, details = train_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(details)
        scheduler.step()

        val_metrics = eval_fn(model, loaders["val"])
        val_f1 = val_metrics["macro_f1"]
        mean_loss = {
            key: float(np.mean([entry[key] for entry in losses]))
            for key in losses[0].keys()
        }
        history.append(
            {
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "lr": scheduler.get_last_lr()[0],
                **mean_loss,
            }
        )

        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1

        if stale_epochs >= config["patience"]:
            break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "history": history,
    }


def train_teacher(data, config, seed):
    teacher = TeacherModel(data["clinical_dim"], data["genomic_dim"], config).to(DEVICE)
    loaders = make_loaders(data, config, seed, include_genomic=True)
    ce_loss = nn.CrossEntropyLoss()

    def train_step(batch):
        x_clin, x_gen, y = batch
        x_clin = x_clin.to(DEVICE)
        x_gen = x_gen.to(DEVICE)
        y = y.to(DEVICE)
        logits = teacher(x_clin, x_gen)
        loss = ce_loss(logits, y)
        return loss, {"loss": loss.item()}

    fit = fit_with_early_stopping(
        teacher,
        loaders,
        config,
        config["teacher_epochs"],
        config["teacher_lr"],
        train_step,
        evaluate_teacher,
    )
    return teacher, loaders, fit


def train_clinical_baseline(data, config, seed):
    student = StudentModel(data["clinical_dim"], config).to(DEVICE)
    loaders = make_loaders(data, config, seed, include_genomic=False)
    ce_loss = nn.CrossEntropyLoss()

    def train_step(batch):
        x_clin, y = batch
        x_clin = x_clin.to(DEVICE)
        y = y.to(DEVICE)
        logits = student(x_clin)
        loss = ce_loss(logits, y)
        return loss, {"loss": loss.item()}

    fit = fit_with_early_stopping(
        student,
        loaders,
        config,
        config["student_epochs"],
        config["student_lr"],
        train_step,
        evaluate_student,
    )
    return student, loaders, fit


def train_distilled_student(data, config, seed, teacher, lambda_value):
    student = StudentModel(data["clinical_dim"], config).to(DEVICE)
    loaders = make_loaders(data, config, seed, include_genomic=True)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    temperature = config["temperature"]
    feature_weight = config["feature_weight"]

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    def train_step(batch):
        x_clin, x_gen, y = batch
        x_clin = x_clin.to(DEVICE)
        x_gen = x_gen.to(DEVICE)
        y = y.to(DEVICE)

        student_out = student(x_clin, return_features=True)
        with torch.no_grad():
            teacher_out = teacher(x_clin, x_gen, return_features=True)

        loss_ce = ce_loss(student_out["logits"], y)
        loss_kd = kl_loss(
            F.log_softmax(student_out["logits"] / temperature, dim=1),
            F.softmax(teacher_out["logits"] / temperature, dim=1),
        ) * (temperature**2)
        loss_feat = F.mse_loss(
            F.normalize(student_out["emb"], p=2, dim=1),
            F.normalize(teacher_out["fused_emb"], p=2, dim=1),
        )

        privileged_loss = loss_kd + feature_weight * loss_feat
        loss = (1.0 - lambda_value) * loss_ce + lambda_value * privileged_loss
        details = {
            "loss": loss.item(),
            "ce": loss_ce.item(),
            "kd": loss_kd.item(),
            "feature_align": loss_feat.item(),
        }
        return loss, details

    fit = fit_with_early_stopping(
        student,
        loaders,
        config,
        config["student_epochs"],
        config["student_lr"],
        train_step,
        evaluate_student,
    )
    return student, loaders, fit


def summarize(values):
    values = [float(v) for v in values]
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "n": len(values),
    }


def build_summary(seed_results, lambda_grid):
    baseline_tests = [run["baseline"]["test"]["macro_f1"] for run in seed_results]
    baseline_vals = [run["baseline"]["val"]["macro_f1"] for run in seed_results]
    teacher_tests = [run["teacher"]["test"]["macro_f1"] for run in seed_results]

    lambda_summary = {}
    for lambda_value in lambda_grid:
        key = str(lambda_value)
        val_scores = [
            run["distilled"][key]["val"]["macro_f1"]
            for run in seed_results
        ]
        test_scores = [
            run["distilled"][key]["test"]["macro_f1"]
            for run in seed_results
        ]
        lambda_summary[key] = {
            "val_macro_f1": summarize(val_scores),
            "test_macro_f1": summarize(test_scores),
        }

    selected_lambda = max(
        lambda_summary,
        key=lambda key: lambda_summary[key]["val_macro_f1"]["mean"],
    )
    selected_tests = [
        run["distilled"][selected_lambda]["test"]["macro_f1"]
        for run in seed_results
    ]
    selected_vals = [
        run["distilled"][selected_lambda]["val"]["macro_f1"]
        for run in seed_results
    ]
    deltas = [
        distilled - baseline
        for distilled, baseline in zip(selected_tests, baseline_tests)
    ]

    return {
        "teacher_test_macro_f1": summarize(teacher_tests),
        "baseline_val_macro_f1": summarize(baseline_vals),
        "baseline_test_macro_f1": summarize(baseline_tests),
        "lambda_summary": lambda_summary,
        "selected_lambda": float(selected_lambda),
        "selected_val_macro_f1": summarize(selected_vals),
        "selected_test_macro_f1": summarize(selected_tests),
        "selected_minus_baseline_test_macro_f1": summarize(deltas),
    }


def run_is_complete(run, lambda_grid):
    if "seed" not in run or "teacher" not in run or "baseline" not in run:
        return False
    distilled = run.get("distilled", {})
    return all(str(lambda_value) in distilled for lambda_value in lambda_grid)


def load_resume_runs(lambda_grid):
    for path in [RESULTS_DIR / "results_partial.json", RESULTS_DIR / "results.json"]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        runs = payload.get("runs", [])
        complete_runs = [
            run for run in runs
            if run_is_complete(run, lambda_grid)
        ]
        if complete_runs:
            return complete_runs, path
    return [], None


def print_run_line(seed, teacher_test, baseline_test, distilled_results):
    distilled_text = "  ".join(
        [
            f"lambda={key}: val {value['val']['macro_f1']:.4f}, test {value['test']['macro_f1']:.4f}"
            for key, value in distilled_results.items()
        ]
    )
    print(
        f"Seed {seed}: teacher test {teacher_test['macro_f1']:.4f}, "
        f"baseline test {baseline_test['macro_f1']:.4f}"
    )
    print(f"  {distilled_text}")


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "seeds": args.seeds,
            "lambda_grid": args.lambda_grid,
            "teacher_epochs": args.teacher_epochs,
            "student_epochs": args.student_epochs,
            "patience": args.patience,
            "temperature": args.temperature,
            "feature_weight": args.feature_weight,
            "batch_size": args.batch_size,
        }
    )

    print(f"Device: {DEVICE}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Seeds: {config['seeds']}")
    print(f"Lambda grid: {config['lambda_grid']}")
    print(f"T={config['temperature']}, feature_weight={config['feature_weight']}")

    data = load_data()
    print(
        "Data: "
        f"clinical={data['clinical_dim']}, genomic={data['genomic_dim']}, "
        f"train={len(data['y_train'])}, val={len(data['y_val'])}, test={len(data['y_test'])}"
    )

    seed_results = []
    if args.resume:
        seed_results, resume_path = load_resume_runs(config["lambda_grid"])
        if resume_path is not None:
            resumed_seeds = [run["seed"] for run in seed_results]
            print(f"Resumed completed seeds from {resume_path}: {resumed_seeds}")

    completed_seeds = {run["seed"] for run in seed_results}
    started = time.time()

    for seed in config["seeds"]:
        if seed in completed_seeds:
            print(f"\n=== Seed {seed} ===")
            print("Already complete; skipping because --resume was provided.")
            continue

        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        teacher, teacher_loaders, teacher_fit = train_teacher(data, config, seed)
        teacher_val = evaluate_teacher(teacher, teacher_loaders["val"])
        teacher_test = evaluate_teacher(teacher, teacher_loaders["test"])
        print(
            f"Teacher: best epoch {teacher_fit['best_epoch']}, "
            f"val {teacher_val['macro_f1']:.4f}, test {teacher_test['macro_f1']:.4f}"
        )

        set_seed(seed)
        baseline, baseline_loaders, baseline_fit = train_clinical_baseline(
            data, config, seed
        )
        baseline_val = evaluate_student(baseline, baseline_loaders["val"])
        baseline_test = evaluate_student(baseline, baseline_loaders["test"])
        print(
            f"Clinical-only: best epoch {baseline_fit['best_epoch']}, "
            f"val {baseline_val['macro_f1']:.4f}, test {baseline_test['macro_f1']:.4f}"
        )

        distilled_results = {}
        for lambda_value in config["lambda_grid"]:
            set_seed(seed)
            distilled, distilled_loaders, distilled_fit = train_distilled_student(
                data, config, seed, teacher, lambda_value
            )
            distilled_val = evaluate_student(distilled, distilled_loaders["val"])
            distilled_test = evaluate_student(distilled, distilled_loaders["test"])
            key = str(lambda_value)
            distilled_results[key] = {
                "lambda": lambda_value,
                "best_epoch": distilled_fit["best_epoch"],
                "best_val_macro_f1": distilled_fit["best_val_macro_f1"],
                "val": distilled_val,
                "test": distilled_test,
                "history": distilled_fit["history"],
            }
            print(
                f"Distilled lambda={lambda_value}: "
                f"epoch {distilled_fit['best_epoch']}, "
                f"val {distilled_val['macro_f1']:.4f}, "
                f"test {distilled_test['macro_f1']:.4f}"
            )

        run = {
            "seed": seed,
            "teacher": {
                "best_epoch": teacher_fit["best_epoch"],
                "best_val_macro_f1": teacher_fit["best_val_macro_f1"],
                "val": teacher_val,
                "test": teacher_test,
                "history": teacher_fit["history"],
            },
            "baseline": {
                "best_epoch": baseline_fit["best_epoch"],
                "best_val_macro_f1": baseline_fit["best_val_macro_f1"],
                "val": baseline_val,
                "test": baseline_test,
                "history": baseline_fit["history"],
            },
            "distilled": distilled_results,
        }
        seed_results.append(run)
        print_run_line(seed, teacher_test, baseline_test, distilled_results)

        partial_summary = build_summary(seed_results, config["lambda_grid"])
        with open(RESULTS_DIR / "results_partial.json", "w", encoding="utf-8") as f:
            json.dump(
                {"config": config, "runs": seed_results, "summary": partial_summary},
                f,
                indent=2,
            )

    summary = build_summary(seed_results, config["lambda_grid"])
    results = {
        "config": config,
        "runs": seed_results,
        "summary": summary,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
    }
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(
        "Teacher test macro F1: "
        f"{summary['teacher_test_macro_f1']['mean']:.4f} "
        f"+/- {summary['teacher_test_macro_f1']['std']:.4f}"
    )
    print(
        "Clinical-only test macro F1: "
        f"{summary['baseline_test_macro_f1']['mean']:.4f} "
        f"+/- {summary['baseline_test_macro_f1']['std']:.4f}"
    )
    print(f"Selected lambda by mean val F1: {summary['selected_lambda']}")
    print(
        "Distilled test macro F1: "
        f"{summary['selected_test_macro_f1']['mean']:.4f} "
        f"+/- {summary['selected_test_macro_f1']['std']:.4f}"
    )
    print(
        "Distilled - clinical-only test macro F1: "
        f"{summary['selected_minus_baseline_test_macro_f1']['mean']:+.4f} "
        f"+/- {summary['selected_minus_baseline_test_macro_f1']['std']:.4f}"
    )
    print(f"Saved: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
