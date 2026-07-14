"""
================================================================================
Phase 2 — EDA
Notebook 05: Survival Analysis
================================================================================

GOAL:
    Determine whether the three treatment groups differ significantly
    in Overall Survival (OS), and identify which clinical/genomic features
    are associated with better or worse survival.

METHODS:
    Kaplan-Meier   → estimate survival probability over time
    Log-Rank Test  → test whether survival curves differ significantly
    Median OS      → time at which 50% of patients have died

KEY CONCEPT — Censoring:
    Not all patients have died by the end of the study.
    OS_EVENT = 0 means the patient was still alive (or lost to follow-up).
    Kaplan-Meier handles these "censored" observations correctly —
    they contribute information up to their last known alive time,
    then are removed from the at-risk pool without being counted as deaths.

WHY THIS MATTERS FOR OUR PROJECT:
    If the three treatment groups show significantly different OS curves,
    it validates that treatment choice is clinically meaningful —
    and predicting the right treatment could genuinely improve outcomes.
================================================================================
"""

# ================================================================================
# SECTION 1: Imports and Setup
# ================================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# lifelines: the standard Python library for survival analysis
# Install with: pip install lifelines
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    print("✓ lifelines loaded")
except ImportError:
    print("✗ lifelines not found — run: pip install lifelines")
    exit()

FIG_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/figures/eda')
FIG_PATH.mkdir(parents=True, exist_ok=True)

TREATMENT_COLORS = {
    'Immunotherapy': '#2196F3',
    'Chemotherapy':  '#FF9800',
    'Targeted':      '#4CAF50',
}
TREATMENT_ORDER = ['Immunotherapy', 'Chemotherapy', 'Targeted']

print("✓ Setup complete")

# ================================================================================
# SECTION 2: Load Data
# ================================================================================
DATA_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
df = pd.read_csv(DATA_PATH)

for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(float)

# Survival columns
T = 'OS_MONTHS'   # time to event or censoring
E = 'OS_EVENT'    # 1 = death observed, 0 = censored

print(f"✓ Loaded: {df.shape[0]:,} patients")
print(f"\n  OS_EVENT=1 (death)   : {df[E].sum():,}  ({df[E].mean()*100:.1f}%)")
print(f"  OS_EVENT=0 (censored): {(df[E]==0).sum():,}  ({(df[E]==0).mean()*100:.1f}%)")
print(f"  OS_MONTHS range      : {df[T].min():.1f} – {df[T].max():.1f} months")

groups = {name: df[df['TREATMENT_FINAL'] == name] for name in TREATMENT_ORDER}

# ================================================================================
# SECTION 3: Overall Survival — All Patients
# ================================================================================
# First, plot the survival curve for the entire cohort (no stratification).
# This gives us the "baseline" — what is the overall median OS?

print(f"\n{'='*60}")
print("  SECTION 1: OVERALL SURVIVAL — ENTIRE COHORT")
print(f"{'='*60}")

kmf_all = KaplanMeierFitter()
kmf_all.fit(df[T], event_observed=df[E], label='All patients')

median_os = kmf_all.median_survival_time_
print(f"\n  Median OS (all patients): {median_os:.1f} months")
print(f"  = {median_os/12:.1f} years")

fig, ax = plt.subplots(figsize=(10, 6))

kmf_all.plot_survival_function(
    ax=ax,
    color='#546E7A',
    linewidth=2,
    ci_show=True,    # ci = confidence interval — shaded area around the curve
    ci_alpha=0.15,
)

# Add a horizontal line at 0.5 (50% survival) to visually show median OS
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.7,
           label='50% survival threshold')
ax.axvline(median_os, color='red', linestyle=':', linewidth=1, alpha=0.7)

ax.set_xlabel('Time (months)', fontsize=11)
ax.set_ylabel('Survival Probability', fontsize=11)
ax.set_title(f'Overall Survival — All NSCLC Patients (n={len(df):,})\n'
             f'Median OS = {median_os:.1f} months',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_PATH / 'survival_overall.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: survival_overall.png")

# ================================================================================
# SECTION 4: Survival by Treatment Group
# ================================================================================
# The main analysis: does treatment group predict survival?
#
# We fit one Kaplan-Meier curve per treatment group and plot them together.
# Then the log-rank test tells us if the differences are statistically significant.

print(f"\n{'='*60}")
print("  SECTION 2: SURVIVAL BY TREATMENT GROUP")
print(f"{'='*60}")

# Fit KM for each group and print median OS
print(f"\n  Median OS per treatment group:")
print(f"  {'Treatment':<20} {'n':>6}  {'Events':>8}  {'Median OS':>12}  {'95% CI'}")
print(f"  {'─'*65}")

kmf_groups = {}
for name in TREATMENT_ORDER:
    grp = groups[name]
    kmf = KaplanMeierFitter()
    kmf.fit(grp[T], event_observed=grp[E], label=name)
    kmf_groups[name] = kmf

    median    = kmf.median_survival_time_
    n_events  = int(grp[E].sum())

    # Confidence interval for median OS
    try:
        ci = kmf.confidence_interval_cumulative_density_
        # find the time range where survival crosses 0.5
        sf = kmf.survival_function_
        above = sf[sf['KM_estimate'] >= 0.5]
        if len(above) > 0:
            ci_low  = above.index[-1]
        else:
            ci_low = float('nan')
        ci_str = f">{ci_low:.0f}m" if not np.isnan(ci_low) else "N/A"
    except Exception:
        ci_str = "N/A"

    print(f"  {name:<20} {len(grp):>6}  {n_events:>8}  {median:>10.1f}m  {ci_str}")

# ── Log-rank test ──────────────────────────────────────────────────────────────
# Tests the null hypothesis: "all three survival curves are the same"
# A small p-value rejects this — the curves are significantly different

result = multivariate_logrank_test(
    df[T], df['TREATMENT_FINAL'], df[E]
)
print(f"\n  Log-Rank Test (3-way):")
print(f"    Test statistic : {result.test_statistic:.3f}")
print(f"    p-value        : {result.p_value:.6f}")
print(f"    Significant    : {'✓ Yes (p < 0.05)' if result.p_value < 0.05 else '✗ No'}")

# ── Plot KM curves by treatment group ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

for name in TREATMENT_ORDER:
    kmf = kmf_groups[name]
    kmf.plot_survival_function(
        ax=ax,
        color=TREATMENT_COLORS[name],
        linewidth=2.5,
        ci_show=True,
        ci_alpha=0.1,
    )

ax.axhline(0.5, color='gray', linestyle='--', linewidth=1,
           alpha=0.5, label='50% threshold')
ax.set_xlabel('Time (months)', fontsize=11)
ax.set_ylabel('Survival Probability', fontsize=11)
ax.set_title(f'Overall Survival by Treatment Group\n'
             f'Log-Rank p = {result.p_value:.2e}',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_PATH / 'survival_by_treatment.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: survival_by_treatment.png")

# ================================================================================
# SECTION 5: Pairwise Log-Rank Tests
# ================================================================================
# The 3-way test tells us "at least one group is different."
# Pairwise tests tell us WHICH pairs are different from each other.

print(f"\n{'='*60}")
print("  SECTION 3: PAIRWISE LOG-RANK TESTS")
print(f"{'='*60}")

pairs = [
    ('Immunotherapy', 'Chemotherapy'),
    ('Immunotherapy', 'Targeted'),
    ('Chemotherapy',  'Targeted'),
]

print(f"\n  {'Comparison':<35} {'p-value':>10}  Significant?")
print(f"  {'─'*55}")

for g1, g2 in pairs:
    grp1 = groups[g1]
    grp2 = groups[g2]
    result_pair = logrank_test(
        grp1[T], grp2[T],
        event_observed_A=grp1[E],
        event_observed_B=grp2[E]
    )
    comparison = f"{g1} vs {g2}"
    sig = "✓ Yes" if result_pair.p_value < 0.05 else "✗ No"
    print(f"  {comparison:<35} {result_pair.p_value:>10.6f}  {sig}")

# ================================================================================
# SECTION 6: Survival by Key Genomic Features
# ================================================================================
# From notebook 04, EGFR and KRAS were the strongest genomic predictors.
# Here we check: do EGFR-mutated patients have different survival than wild-type?

print(f"\n{'='*60}")
print("  SECTION 4: SURVIVAL BY KEY GENOMIC FEATURES")
print(f"{'='*60}")

key_genes = ['MUT_EGFR', 'MUT_KRAS', 'MUT_STK11', 'MUT_KEAP1']
key_genes  = [g for g in key_genes if g in df.columns]

fig, axes = plt.subplots(1, len(key_genes), figsize=(5 * len(key_genes), 6))

gene_results = []
for i, gene in enumerate(key_genes):
    ax = axes[i]
    gene_name = gene.replace('MUT_', '')

    mutated  = df[df[gene] == 1]
    wildtype = df[df[gene] == 0]

    kmf_mut = KaplanMeierFitter()
    kmf_wt  = KaplanMeierFitter()

    kmf_mut.fit(mutated[T],  event_observed=mutated[E],
                label=f'{gene_name} mutated (n={len(mutated):,})')
    kmf_wt.fit(wildtype[T],  event_observed=wildtype[E],
               label=f'{gene_name} wild-type (n={len(wildtype):,})')

    # Log-rank test between mutated vs wild-type
    lr = logrank_test(
        mutated[T], wildtype[T],
        event_observed_A=mutated[E],
        event_observed_B=wildtype[E]
    )

    kmf_mut.plot_survival_function(ax=ax, color='#E53935', linewidth=2,
                                    ci_show=True, ci_alpha=0.1)
    kmf_wt.plot_survival_function(ax=ax,  color='#1565C0', linewidth=2,
                                   ci_show=True, ci_alpha=0.1)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    ax.set_title(f'{gene_name}\np={lr.p_value:.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Months', fontsize=9)
    ax.set_ylabel('Survival Probability', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)

    median_mut = kmf_mut.median_survival_time_
    median_wt  = kmf_wt.median_survival_time_
    gene_results.append({
        'gene': gene_name, 'p_value': lr.p_value,
        'median_mutated': median_mut, 'median_wildtype': median_wt
    })

plt.suptitle('Survival by Mutation Status — Key Genes',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH / 'survival_by_gene.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: survival_by_gene.png")

print(f"\n  Median OS by mutation status:")
print(f"  {'Gene':<12} {'Mutated':>12}  {'Wild-Type':>12}  {'p-value':>10}")
print(f"  {'─'*50}")
for r in gene_results:
    print(f"  {r['gene']:<12} {r['median_mutated']:>10.1f}m  "
          f"{r['median_wildtype']:>10.1f}m  {r['p_value']:>10.6f}")

# ================================================================================
# SECTION 7: Survival by Stage
# ================================================================================
print(f"\n{'='*60}")
print("  SECTION 5: SURVIVAL BY STAGE")
print(f"{'='*60}")

fig, ax = plt.subplots(figsize=(10, 6))

stage_labels = {0: 'Stage 1-3', 1: 'Stage 4'}
stage_colors = {0: '#1565C0', 1: '#E53935'}

for stage_val, stage_label in stage_labels.items():
    grp = df[df['STAGE_ENC'] == stage_val]
    kmf = KaplanMeierFitter()
    kmf.fit(grp[T], event_observed=grp[E],
            label=f'{stage_label} (n={len(grp):,})')
    kmf.plot_survival_function(ax=ax, color=stage_colors[stage_val],
                                linewidth=2.5, ci_show=True, ci_alpha=0.1)
    print(f"\n  {stage_label}: n={len(grp):,}, "
          f"median OS={kmf.median_survival_time_:.1f}m")

lr_stage = logrank_test(
    df[df['STAGE_ENC']==0][T], df[df['STAGE_ENC']==1][T],
    event_observed_A=df[df['STAGE_ENC']==0][E],
    event_observed_B=df[df['STAGE_ENC']==1][E]
)
print(f"\n  Log-rank p-value (Stage 1-3 vs 4): {lr_stage.p_value:.6f}")

ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time (months)', fontsize=11)
ax.set_ylabel('Survival Probability', fontsize=11)
ax.set_title(f'Overall Survival by Stage\nLog-Rank p = {lr_stage.p_value:.2e}',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_PATH / 'survival_by_stage.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: survival_by_stage.png")

# ================================================================================
# SECTION 8: Summary
# ================================================================================
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

print(f"""
  KEY FINDINGS:

  1. Overall cohort:
     Median OS = {kmf_all.median_survival_time_:.1f} months

  2. Treatment group differences:
     Immunotherapy : {kmf_groups['Immunotherapy'].median_survival_time_:.1f} months
     Chemotherapy  : {kmf_groups['Chemotherapy'].median_survival_time_:.1f} months
     Targeted      : {kmf_groups['Targeted'].median_survival_time_:.1f} months

  3. Are treatment differences significant?
     Log-rank p = {result.p_value:.2e}
     {'✓ YES — survival differs significantly between groups' if result.p_value < 0.05
      else '✗ NO — no significant difference'}

  FIGURES SAVED:
    survival_overall.png
    survival_by_treatment.png
    survival_by_gene.png
    survival_by_stage.png
""")

print("✓ Notebook 05 complete — EDA phase finished!")
print("\n  Next: Phase 3 — Baseline Models")
