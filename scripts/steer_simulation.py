"""Monte Carlo simulation for One Concept at a Time (steer).

Generates tabs/table1_stage1.tex, tabs/table2_stage2.tex,
tabs/macros_sim.tex, and figs/fig1_simulation.pdf from one computed source.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import ACCENT, AMBER, MUTED, TEAL, WARN, apply_style

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
TABS = ROOT / "tabs"
FIGS = ROOT / "figs"
TABS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


def orthonormal_columns(d, k, rng):
    mat = rng.randn(d, k)
    q, _ = np.linalg.qr(mat)
    return q[:, :k]


def principal_angle_cos(u_a, u_b):
    u_a = u_a / np.linalg.norm(u_a)
    u_b = u_b / np.linalg.norm(u_b)
    return float(np.abs(np.dot(u_a, u_b)))


def estimate_stage1(delta_h, rng):
    """Return uncentered PCA u1, mean-direction u_mean, and diagnostics."""
    _, m = delta_h.shape
    s_mat = (delta_h @ delta_h.T) / m
    evals, evecs = np.linalg.eigh(s_mat)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    u_pca = evecs[:, idx[0]]

    mean_vec = delta_h.mean(axis=1)
    u_mean = mean_vec / np.linalg.norm(mean_vec)
    if np.dot(u_pca, u_mean) < 0:
        u_pca = -u_pca

    r2_pca = float(evals[0] / np.sum(evals))
    cos_pca, cos_mean = [], []
    half = m // 2
    for _ in range(20):
        perm = rng.permutation(m)
        h1, h2 = delta_h[:, perm[:half]], delta_h[:, perm[half:]]
        _, v1 = np.linalg.eigh((h1 @ h1.T) / h1.shape[1])
        _, v2 = np.linalg.eigh((h2 @ h2.T) / h2.shape[1])
        cos_pca.append(principal_angle_cos(v1[:, -1], v2[:, -1]))
        cos_mean.append(principal_angle_cos(h1.mean(axis=1), h2.mean(axis=1)))

    split_mean_val = float(np.mean(cos_mean))
    return {
        "u_pca": u_pca,
        "u_mean": u_mean,
        "r2_1": r2_pca,
        "split_cos_pca": float(np.mean(cos_pca)),
        "split_cos_mean": split_mean_val,
        "pass_gate": float(split_mean_val >= 0.70),
    }


def dml_partially_linear(y, z, r_mat, alpha_ridge=1.0, n_folds=5):
    """K-fold cross-fitted DML with ridge regression for g(R) and m(R)."""
    n = len(y)
    if z.ndim == 1:
        z = z[:, None]
    r_centered = r_mat - r_mat.mean(axis=0)
    _, s_r, vt_r = np.linalg.svd(r_centered, full_matrices=False)
    keep = s_r > 1e-6
    r_low = r_centered @ vt_r[keep].T
    p_dim = r_low.shape[1]
    z_tilde = np.zeros_like(z)
    y_tilde = np.zeros_like(y)

    folds = np.array_split(np.arange(n), n_folds)
    for f_idx in range(n_folds):
        val_idx = folds[f_idx]
        other = [folds[j] for j in range(n_folds) if j != f_idx]
        tr_idx = np.concatenate(other)
        r_tr, r_val = r_low[tr_idx], r_low[val_idx]
        y_tr = y[tr_idx]
        z_tr = z[tr_idx]
        r_mean = r_tr.mean(axis=0)
        r_tr_c = r_tr - r_mean
        r_val_c = r_val - r_mean
        gram = r_tr_c.T @ r_tr_c + alpha_ridge * np.eye(p_dim)

        w_y = np.linalg.solve(gram, r_tr_c.T @ (y_tr - y_tr.mean()))
        y_hat = y_tr.mean() + r_val_c @ w_y
        y_tilde[val_idx] = y[val_idx] - y_hat

        w_z = np.linalg.solve(gram, r_tr_c.T @ (z_tr - z_tr.mean(axis=0)))
        z_hat = z_tr.mean(axis=0) + r_val_c @ w_z
        z_tilde[val_idx] = z[val_idx] - z_hat

    theta_hat = np.linalg.solve(z_tilde.T @ z_tilde, z_tilde.T @ y_tilde)
    resid = y_tilde - z_tilde @ theta_hat
    j_inv = np.linalg.inv((z_tilde.T @ z_tilde) / n)
    meat = ((z_tilde * resid[:, None]).T @ (z_tilde * resid[:, None])) / n
    dof_scale = n / max(n - p_dim - z.shape[1], 1)
    vcov = dof_scale * (j_inv @ meat @ j_inv) / n
    se = np.sqrt(np.diag(vcov))
    r2_z = 1.0 - np.var(z_tilde[:, 0]) / np.var(z[:, 0])
    return theta_hat, se, float(r2_z)


def plot_simulation(stage1_rows, stage2_rows):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.05, 1.15]}
    )

    # Panel A: Stage 1 gate acceptance vs rejection
    regimes = ["clean", "high_var_noise", "curved_no_subspace"]
    x_labels = [
        "Clean shift\n(Pass: 100%)",
        "Nuisance noise\n(Pass: 100%)",
        "Dispersed\n(Reject: 0%)",
    ]
    x = np.arange(len(regimes))
    width = 0.24
    s_mean = [np.mean(stage1_rows[r]["split_mean"]) for r in regimes]
    c_mean = [np.mean(stage1_rows[r]["cos_mean_true"]) for r in regimes]
    c_pca = [np.mean(stage1_rows[r]["cos_pca_true"]) for r in regimes]

    ax1.bar(x - width, s_mean, width, color=ACCENT, label="Split-half (mean)")
    ax1.bar(
        x,
        c_mean,
        width,
        color=TEAL,
        label=r"$\cos(\hat{u}_{\mathrm{mean}}, u_C^*)$",
    )
    ax1.bar(
        x + width,
        c_pca,
        width,
        color=AMBER,
        label=r"$\cos(\hat{u}_{\mathrm{PCA}}, u_C^*)$",
    )
    ax1.axhline(0.70, color=WARN, linestyle="--", linewidth=1.0)
    ax1.text(
        2.35,
        0.73,
        "Gate (0.70)",
        color=WARN,
        fontsize=7.8,
        ha="right",
        va="bottom",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("A. Stage 1 diagnostic gate", loc="left")
    ax1.legend(
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.22),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.8,
    )

    # Panel B: Stage 2 estimation error dot-and-95%-sampling-interval plot
    methods = [
        ("dml_multi_oblique", "Oblique multi-DML", ACCENT),
        ("dml_mean_dir", r"DML ($\hat{u}_{\mathrm{mean}}$)", TEAL),
        ("dml_uncentered_pca", r"DML ($\hat{u}_{\mathrm{PCA}}$)", AMBER),
        ("naive_ols", "Unadjusted OLS", WARN),
    ]
    y_pos = np.arange(len(methods))
    ax2.axvline(0.0, color=MUTED, linestyle="--", linewidth=0.9)
    for idx, (key, label, col) in enumerate(methods):
        errs = np.array(stage2_rows[key]["bias"])
        mean_e = float(np.mean(errs))
        lo, hi = float(np.percentile(errs, 2.5)), float(
            np.percentile(errs, 97.5)
        )
        ax2.plot([lo, hi], [idx, idx], color=col, linewidth=2.0)
        ax2.plot(mean_e, idx, "o", color=col, markersize=5.2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m[1] for m in methods], fontsize=8.2)
    ax2.set_xlabel(
        r"Estimation error ($\hat{\theta}_1 - \theta_1^*$, target $= 0$)"
    )
    ax2.set_title("B. Stage 2 estimation error", loc="left")
    ax2.set_xlim(-2.6, 1.15)
    ax2.tick_params(axis="y", length=0)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGS / "fig1_simulation.pdf")
    plt.close(fig)


def run_simulations(n_sims=400, d=64, m=80, n=600, theta_true=1.5):
    stage1_rows = {
        reg: {
            "r2": [],
            "cos_pca_true": [],
            "cos_mean_true": [],
            "split_pca": [],
            "split_mean": [],
            "pass_gate": [],
        }
        for reg in ("clean", "high_var_noise", "curved_no_subspace")
    }
    stage2_rows = {
        method: {"bias": [], "rmse": [], "cov": [], "r2_z": []}
        for method in (
            "naive_ols",
            "dml_uncentered_pca",
            "dml_mean_dir",
            "dml_multi_incorrect",
            "dml_multi_oblique",
        )
    }

    for sim in range(n_sims):
        rng = np.random.RandomState(sim + 100)
        basis = orthonormal_columns(d, 5, rng)
        u_c1 = basis[:, 0]
        u_nuis = basis[:, 1]
        u_conf = basis[:, 2]
        u_c2 = np.cos(np.pi / 4) * u_c1 + np.sin(np.pi / 4) * basis[:, 3]

        # Regime A: Clean rank-1 shift + isotropic small edit noise
        amp_a = 1.0 + 0.2 * rng.randn(m)
        dh_clean = np.outer(u_c1, amp_a) + 0.15 * rng.randn(d, m)
        s1_clean = estimate_stage1(dh_clean, rng)
        row_a = stage1_rows["clean"]
        row_a["r2"].append(s1_clean["r2_1"])
        row_a["cos_pca_true"].append(
            principal_angle_cos(s1_clean["u_pca"], u_c1)
        )
        row_a["cos_mean_true"].append(
            principal_angle_cos(s1_clean["u_mean"], u_c1)
        )
        row_a["split_pca"].append(s1_clean["split_cos_pca"])
        row_a["split_mean"].append(s1_clean["split_cos_mean"])
        row_a["pass_gate"].append(s1_clean["pass_gate"])

        # Regime B: Mean shift along u_c1, high zero-mean noise along u_nuis
        dh_noisy = (
            np.outer(u_c1, 1.0 + 0.15 * rng.randn(m))
            + np.outer(u_nuis, 1.8 * rng.randn(m))
            + 0.15 * rng.randn(d, m)
        )
        s1_noisy = estimate_stage1(dh_noisy, rng)
        row_b = stage1_rows["high_var_noise"]
        row_b["r2"].append(s1_noisy["r2_1"])
        row_b["cos_pca_true"].append(
            principal_angle_cos(s1_noisy["u_pca"], u_c1)
        )
        row_b["cos_mean_true"].append(
            principal_angle_cos(s1_noisy["u_mean"], u_c1)
        )
        row_b["split_pca"].append(s1_noisy["split_cos_pca"])
        row_b["split_mean"].append(s1_noisy["split_cos_mean"])
        row_b["pass_gate"].append(s1_noisy["pass_gate"])

        # Regime C: Dispersed edit across 15 random directions (no subspace)
        rand_dirs = orthonormal_columns(d, 15, rng)
        dh_curved = rand_dirs @ rng.randn(15, m) + 0.2 * rng.randn(d, m)
        s1_curved = estimate_stage1(dh_curved, rng)
        row_c = stage1_rows["curved_no_subspace"]
        row_c["r2"].append(s1_curved["r2_1"])
        row_c["cos_pca_true"].append(
            principal_angle_cos(s1_curved["u_pca"], u_c1)
        )
        row_c["cos_mean_true"].append(
            principal_angle_cos(s1_curved["u_mean"], u_c1)
        )
        row_c["split_pca"].append(s1_curved["split_cos_pca"])
        row_c["split_mean"].append(s1_curved["split_cos_mean"])
        row_c["pass_gate"].append(s1_curved["pass_gate"])

        # Stage 2 Observational DGP
        conf = rng.randn(n)
        nuis = rng.randn(n)
        c1_lat = 0.8 * conf + 0.6 * nuis + rng.randn(n)
        c2_lat = 0.5 * conf + rng.randn(n)

        h_obs = (
            np.outer(c1_lat, u_c1)
            + np.outer(nuis, u_nuis)
            + np.outer(conf, u_conf)
            + 0.3 * rng.randn(n, d)
        )
        beta_h = theta_true * u_c1 + 1.2 * u_conf + 1.0 * u_nuis
        y_obs = h_obs @ beta_h + rng.randn(n)

        # 1. Naive OLS without R
        u_hat_mean = s1_noisy["u_mean"]
        z_mean = h_obs @ u_hat_mean
        theta_mean_target = float(np.dot(u_hat_mean, beta_h))
        z_c = z_mean - z_mean.mean()
        y_c = y_obs - y_obs.mean()
        theta_naive = float(np.dot(z_c, y_c) / np.dot(z_c, z_c))
        resid_naive = y_c - theta_naive * z_c
        se_naive = float(
            np.sqrt(np.sum((z_c * resid_naive) ** 2) / (np.dot(z_c, z_c) ** 2))
        )
        err_naive = theta_naive - theta_true
        stage2_rows["naive_ols"]["bias"].append(err_naive)
        stage2_rows["naive_ols"]["rmse"].append(err_naive**2)
        stage2_rows["naive_ols"]["cov"].append(
            abs(err_naive) <= 1.96 * se_naive
        )
        stage2_rows["naive_ols"]["r2_z"].append(0.0)

        # 2. DML using uncentered PCA u_1 from Regime B
        u_hat_pca = s1_noisy["u_pca"]
        z_pca = h_obs @ u_hat_pca
        r_pca = h_obs - np.outer(z_pca, u_hat_pca)
        th_pca, se_pca, r2_pca = dml_partially_linear(
            y_obs, z_pca, r_pca, alpha_ridge=1e-4
        )
        err_pca = float(th_pca[0]) - theta_true
        stage2_rows["dml_uncentered_pca"]["bias"].append(err_pca)
        stage2_rows["dml_uncentered_pca"]["rmse"].append(err_pca**2)
        stage2_rows["dml_uncentered_pca"]["cov"].append(
            abs(err_pca) <= 1.96 * float(se_pca[0])
        )
        stage2_rows["dml_uncentered_pca"]["r2_z"].append(r2_pca)

        # 3. DML using mean direction u_mean from Regime B
        r_mean = h_obs - np.outer(z_mean, u_hat_mean)
        th_m, se_m, r2_m = dml_partially_linear(
            y_obs, z_mean, r_mean, alpha_ridge=1e-4
        )
        err_m = float(th_m[0]) - theta_mean_target
        stage2_rows["dml_mean_dir"]["bias"].append(err_m)
        stage2_rows["dml_mean_dir"]["rmse"].append(err_m**2)
        stage2_rows["dml_mean_dir"]["cov"].append(
            abs(err_m) <= 1.96 * float(se_m[0])
        )
        stage2_rows["dml_mean_dir"]["r2_z"].append(r2_m)

        # Multi-concept DGP (two non-orthogonal concepts, cos = 0.707)
        theta2_true = -1.0
        u_multi = np.column_stack([u_c1, u_c2])
        u_dual = u_multi @ np.linalg.inv(u_multi.T @ u_multi)
        c1_term = theta_true * u_dual[:, 0]
        c2_term = theta2_true * u_dual[:, 1]
        beta_h_multi = c1_term + c2_term + 1.2 * u_conf
        h_multi = (
            np.outer(c1_lat, u_c1)
            + np.outer(c2_lat, u_c2)
            + np.outer(conf, u_conf)
            + 0.3 * rng.randn(n, d)
        )
        y_multi = h_multi @ beta_h_multi + rng.randn(n)

        # 4. Naive orthogonal formula Z = U^T h, R = (I - U U^T) h
        z_inc = h_multi @ u_multi
        r_inc = h_multi - z_inc @ u_multi.T
        th_inc, se_inc, r2_inc = dml_partially_linear(
            y_multi, z_inc, r_inc, alpha_ridge=1e-4
        )
        err_inc = float(th_inc[0]) - theta_true
        stage2_rows["dml_multi_incorrect"]["bias"].append(err_inc)
        stage2_rows["dml_multi_incorrect"]["rmse"].append(err_inc**2)
        stage2_rows["dml_multi_incorrect"]["cov"].append(
            abs(err_inc) <= 1.96 * float(se_inc[0])
        )
        stage2_rows["dml_multi_incorrect"]["r2_z"].append(r2_inc)

        # 5. Oblique pseudoinverse projection Z = (U^T U)^(-1) U^T h
        z_obl = h_multi @ u_dual
        r_obl = h_multi - z_obl @ u_multi.T
        th_obl, se_obl, r2_obl = dml_partially_linear(
            y_multi, z_obl, r_obl, alpha_ridge=1e-4
        )
        err_obl = float(th_obl[0]) - theta_true
        stage2_rows["dml_multi_oblique"]["bias"].append(err_obl)
        stage2_rows["dml_multi_oblique"]["rmse"].append(err_obl**2)
        stage2_rows["dml_multi_oblique"]["cov"].append(
            abs(err_obl) <= 1.96 * float(se_obl[0])
        )
        stage2_rows["dml_multi_oblique"]["r2_z"].append(r2_obl)

    # Write Table 1
    t1_hdr = (
        r"Minimal-Pair Regime & $R_1^2$ & "
        r"$\cos(\hat{u}_{\mathrm{PCA}}, u_C^*)$ & "
        r"$\cos(\hat{u}_{\mathrm{mean}}, u_C^*)$ & "
        r"Split (PCA) & Split (Mean) & Pass \\"
    )
    t1_lines = [
        r"\begin{tabular}{@{}l rrrrrr@{}}",
        r"\toprule",
        t1_hdr,
        r"\midrule",
    ]
    labels_s1 = {
        "clean": "Clean low-rank shift",
        "high_var_noise": "High-variance nuisance edit noise",
        "curved_no_subspace": "Dispersed / no subspace (reject)",
    }
    for key, label in labels_s1.items():
        d_row = stage1_rows[key]
        pg = f"{100.0 * np.mean(d_row['pass_gate']):.0f}\\%"
        t1_lines.append(
            f"{label} & {np.mean(d_row['r2']):.2f} & "
            f"{np.mean(d_row['cos_pca_true']):.2f} & "
            f"{np.mean(d_row['cos_mean_true']):.2f} & "
            f"{np.mean(d_row['split_pca']):.2f} & "
            f"{np.mean(d_row['split_mean']):.2f} & {pg} \\\\"
        )
    t1_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABS / "table1_stage1.tex").write_text("\n".join(t1_lines) + "\n")

    # Write Table 2
    t2_hdr = (
        r"Estimator / Projection Specification & "
        r"Bias & RMSE & 95\% Cov. & $R^2(Z \mid R)$ \\"
    )
    t2_lines = [
        r"\begin{tabular}{@{}l rrrr@{}}",
        r"\toprule",
        t2_hdr,
        r"\midrule",
    ]
    s_pca = r"$\hat{u}_{\mathrm{PCA}}$"
    s_mean = r"$\hat{u}_{\mathrm{mean}}$"
    s_cos = r"($\cos\theta = 0.71$)"
    labels_s2 = {
        "naive_ols": "Unadjusted OLS (no residual control $R_i$)",
        "dml_uncentered_pca": f"DML with uncentered PCA {s_pca} (noise-corrupted)",
        "dml_mean_dir": f"DML with mean-shift {s_mean} (single)",
        "dml_multi_incorrect": f"Multi-concept DML, naive $U_k^\\top h_i$ {s_cos}",
        "dml_multi_oblique": (
            "Multi-concept DML, oblique $(U_k^\\top U_k)^{-1}U_k^\\top h_i$"
        ),
    }
    for key, label in labels_s2.items():
        d_row = stage2_rows[key]
        bias = float(np.mean(d_row["bias"]))
        rmse = float(np.sqrt(np.mean(d_row["rmse"])))
        cov = float(np.mean(d_row["cov"]))
        r2z = float(np.mean(d_row["r2_z"]))
        t2_lines.append(
            f"{label} & {bias:+.3f} & {rmse:.3f} & {cov:.3f} & {r2z:.2f} \\\\"
        )
    t2_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABS / "table2_stage2.tex").write_text("\n".join(t2_lines) + "\n")

    # Write generated LaTeX macros so prose never hand-types simulation numbers
    s1c = stage1_rows["clean"]
    s1n = stage1_rows["high_var_noise"]
    s1d = stage1_rows["curved_no_subspace"]

    def s2_stats(k):
        b = float(np.mean(stage2_rows[k]["bias"]))
        r = float(np.sqrt(np.mean(stage2_rows[k]["rmse"])))
        c = 100.0 * float(np.mean(stage2_rows[k]["cov"]))
        z = float(np.mean(stage2_rows[k]["r2_z"]))
        return b, r, c, z

    b_ols, r_ols, c_ols, _ = s2_stats("naive_ols")
    b_pca, _, _, _ = s2_stats("dml_uncentered_pca")
    b_m, r_m, c_m, z_m = s2_stats("dml_mean_dir")
    b_inc, r_inc, _, z_inc = s2_stats("dml_multi_incorrect")
    b_obl, r_obl, c_obl, z_obl = s2_stats("dml_multi_oblique")

    macros = [
        f"\\newcommand{{\\simNReps}}{{{n_sims}}}",
        f"\\newcommand{{\\simDim}}{{{d}}}",
        f"\\newcommand{{\\simPairs}}{{{m}}}",
        f"\\newcommand{{\\simObs}}{{{n}}}",
        f"\\newcommand{{\\simTrueTheta}}{{{theta_true:.2f}}}",
        f"\\newcommand{{\\simCleanCosMean}}{{{np.mean(s1c['cos_mean_true']):.2f}}}",
        f"\\newcommand{{\\simCleanSplitMean}}{{{np.mean(s1c['split_mean']):.2f}}}",
        f"\\newcommand{{\\simNoisyROne}}{{{np.mean(s1n['r2']):.2f}}}",
        f"\\newcommand{{\\simNoisySplitPca}}{{{np.mean(s1n['split_pca']):.2f}}}",
        f"\\newcommand{{\\simNoisyCosPca}}{{{np.mean(s1n['cos_pca_true']):.2f}}}",
        f"\\newcommand{{\\simNoisyCosMean}}{{{np.mean(s1n['cos_mean_true']):.2f}}}",
        f"\\newcommand{{\\simNoisySplitMean}}{{{np.mean(s1n['split_mean']):.2f}}}",
        f"\\newcommand{{\\simDispROne}}{{{np.mean(s1d['r2']):.2f}}}",
        f"\\newcommand{{\\simDispSplitMean}}{{{np.mean(s1d['split_mean']):.2f}}}",
        f"\\newcommand{{\\simDispSplitPca}}{{{np.mean(s1d['split_pca']):.2f}}}",
        f"\\newcommand{{\\simOlsBias}}{{{b_ols:+.3f}}}",
        f"\\newcommand{{\\simOlsRmse}}{{{r_ols:.3f}}}",
        f"\\newcommand{{\\simPcaBias}}{{{b_pca:+.3f}}}",
        f"\\newcommand{{\\simMeanBias}}{{{b_m:+.3f}}}",
        f"\\newcommand{{\\simMeanRmse}}{{{r_m:.3f}}}",
        f"\\newcommand{{\\simMeanCov}}{{{c_m:.1f}}}",
        f"\\newcommand{{\\simMeanRTwo}}{{{z_m:.2f}}}",
        f"\\newcommand{{\\simNaiveBias}}{{{b_inc:+.1f}}}",
        f"\\newcommand{{\\simNaiveRmse}}{{{r_inc:.1f}}}",
        f"\\newcommand{{\\simNaiveRTwo}}{{{z_inc:.2f}}}",
        f"\\newcommand{{\\simOblBias}}{{{b_obl:+.3f}}}",
        f"\\newcommand{{\\simOblRmse}}{{{r_obl:.3f}}}",
        f"\\newcommand{{\\simOblCov}}{{{c_obl:.1f}}}",
        f"\\newcommand{{\\simOblRTwo}}{{{z_obl:.2f}}}",
    ]
    (TABS / "macros_sim.tex").write_text("\n".join(macros) + "\n")

    plot_simulation(stage1_rows, stage2_rows)


if __name__ == "__main__":
    run_simulations()
