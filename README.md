# One Concept at a Time: Subspace-Constrained Causal Inference for High-Dimensional Treatments

Working paper and Monte Carlo replication code for **"One Concept at a Time: Subspace-Constrained Causal Inference for High-Dimensional Treatments"** by Gaurav Sood.

## Overview

When treatments are high-dimensional objects such as texts, images, or audio, empirical researchers rarely care about the causal effect of the raw surface object $T$. They want the causal effect of a human-interpretable concept $C$ (such as politeness, threat, or formality), holding all other background attributes $W$ fixed.

This repository implements a two-stage framework separating concept geometry from causal estimation:

1. **Stage 1 (Subspace Extraction and Pre-Outcome Falsification Gate):**
   - Decomposes the contrastive difference matrix $D = [\Delta h_1, \dots, \Delta h_m] \in \mathbb{R}^{d \times m}$ from minimal pairs $(T_j^-, T_j^+)$ into the normalized mean shift $\hat{u}_{\mathrm{mean}} = \bar{\Delta h} / \|\bar{\Delta h}\|_2$ (which averages out high-variance zero-mean nuisance edit noise for $k=1$) and centered covariance principal components (for $k > 1$ context heterogeneity).
   - Evaluates four pre-outcome diagnostics before touching outcome data $Y$: signal concentration ($R_k^2$ and mean-vs-PCA alignment), split-half principal-angle stability, out-of-domain transfer across context families, and off-target leakage.
2. **Stage 2 (Causal Estimation via Steering or Double/Debiased Machine Learning):**
   - **Design A (Experimental Steering):** Perturbs representations along $U_k$ with round-trip decoding diagnostics (on-target dose realization and off-subspace leakage ratio $\mathcal{L}_i(\Gamma_i)$).
   - **Design B (Observational DML):** Projects representations onto the concept subspace and its $(d-k)$-dimensional orthogonal complement basis $U_\perp$, using **oblique pseudoinverse projections** $Z_i = (U_k^\top U_k)^{-1}U_k^\top h_i$ and $R_i = (I_d - U_k (U_k^\top U_k)^{-1}U_k^\top)h_i$ when multiple concepts are non-orthogonal, and reports the Stage-2 overlap diagnostic $R^2(Z \mid R)$.

## Repository Structure

- `ms/steer.tex`, `ms/references.bib`, `ms/Makefile`, `ms/steer.pdf`: LaTeX manuscript (amended Kieran Healy XeLaTeX template) and compiled PDF.
- `scripts/steer_simulation.py`: Monte Carlo simulation replicating Table 1 (Stage 1 diagnostics and rejection gate) and Table 2 (Stage 2 DML estimation and multi-concept oblique projection).
- `scripts/empirical_politeness.py`: Empirical application and placebo suite on the Stanford Politeness Corpora (Wikipedia $N=4{,}353$ and Stack Exchange $N=6{,}603$) using frozen `all-MiniLM-L6-v2` ($d=384$), generating Table 3 and Table 4.
- `tabs/table1_stage1.tex`, `tabs/table2_stage2.tex`, `tabs/table3_empirical_stage1.tex`, `tabs/table4_empirical_stage2.tex`: Generated LaTeX tables included in `ms/steer.tex`.

## Reproducing the Simulation, Empirical Application, and Manuscript

```bash
python3 scripts/steer_simulation.py
uv run --with sentence-transformers --with scikit-learn --with numpy python3 scripts/empirical_politeness.py
cd ms && make
```
