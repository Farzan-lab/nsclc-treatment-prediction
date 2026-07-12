# docs/

## What is this folder?

The storage location for **project documentation** — information that is necessary
for understanding and reproducing the project, but does not belong inside the code.

---

## Existing Files

### `data_dictionary.md` — Data Dictionary

A detailed description of every column in the dataset:
- Column name
- Data type (numeric / binary / ordinal)
- Clinical description
- Range of values

---

## Files to Be Added

### `research_decisions.md` — Research Decision Log

Every important project decision is recorded here along with its rationale.
This is extremely useful when writing the Methods section of the paper.

Example:

```markdown
## 2025-07-12 — Selection of Top 50 Genes
**Decision:** Use the 50 most frequently mutated NSCLC genes.
**Rationale:** Covers 93.1% of patients; rare genes lack sufficient signal.
**Rejected alternative:** All genes — leads to curse of dimensionality.

## 2025-07-15 — Removal of Patients with Unknown Treatment
**Decision:** 1,698 patients with unspecified treatment were excluded.
**Rationale:** The target variable must be clearly defined.
```

---

### `experiment_log.md` — Experiment Log

A plain-language summary of the results from each experiment:

```markdown
## Exp 03 — Cross-Modal lambda=0.5
**Date:** 2025-08-10
**Result:** val macro_f1 = 0.748 (better than baseline 0.721)
**Observation:** Attention focuses on TMB and KRAS
**Next step:** Try lambda=1.0
```

---

### `paper_outline.md` — Paper Outline

The structure of the paper, its main claims, and the required figures.
Updated continuously as the project progresses.

---

### `references.bib` — References

A BibTeX file containing all referenced papers:
- MSK-CHORD (Nature 2024)
- G-HANet (CVPR 2024)
- MIND (TMLR 2025)
- DeePaN
- And other related works

---

## Why Does This Documentation Matter?

Six months from now when you are writing the paper, you will not remember
why a particular decision was made. `research_decisions.md` has the answer.

If a reviewer wants to reproduce the project, `data_dictionary.md`
tells them what every column means.

If a co-author wants to join the project, `experiment_log.md`
shows them exactly where things stand.
