"""Monte Carlo simulation for One Concept at a Time (steer)."""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
TABS = ROOT / "tabs"
TABS.mkdir(exist_ok=True)


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
    d, m = delta_h.shape
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
    c_mat = np.cov(delta_h)
    c_evals = np.linalg.eigvalsh(c_mat)[::-1]

    # Split-half stability of mean direction vs uncentered PCA
    cos_pca, cos_mean = [], []
    half = m // 2
    for _ in range(20):
        perm = rng.permutation(m)
        h1, h2 = delta_h[:, perm[:half]], delta_h[:, perm[half:]]
        _, v1 = np.linalg.eigh((h1 @ h1.T) / h1.shape[1])
        _, v2 = np.linalg.eigh((h2 @ h2.T) / h2.shape[1])
        cos_pca.append(principal_angle_cos(v1[:, -1], v2[:, -1]))
        cos_mean.append(principal_angle_cos(h1.mean(axis=1), h2.mean(axis=1)))

    return {
        "u_pca": u_pca,
        "u_mean": u_mean,
        "r2_1": r2_pca,
        "centered_top_ratio": float(c_evals[0] / np.sum(c_evals)),
        "split_cos_pca": float(np.mean(cos_pca)),
        "split_cos_mean": float(np.mean(cos_mean)),
    }


def dml_partially_linear(y, z, r_mat, alpha_ridge=1.0, n_folds=5):
    """K-fold cross-fitted DML with ridge regression for g(R) and m(R)."""
    n = len(y)
    if z.ndim == 1:
        z = z[:, None]
    # Orthonormal basis for active column space of r_mat (rank <= d - k)
    r_centered = r_mat - r_mat.mean(axis=0)
    _, s_r, vt_r = np.linalg.svd(r_centered, full_matrices=False)
    keep = s_r > 1e-6
    r_low = r_centered @ vt_r[keep].T
    p_dim = r_low.shape[1]
    z_tilde = np.zeros_like(z)
    y_tilde = np.zeros_like(y)

    folds = np.array_split(np.arange(n), n_folds)
    for f_idx in range(n_folds):
        test_idx = folds[f_idx]
        other = [folds[j] for j in range(n_folds) if j != f_idx]
        train_idx = np.concatenate(other)
        r_tr, r_te = r_low[train_idx], r_low[test_idx]
        y_tr = y[train_idx]
        z_tr = z[train_idx]
        r_mean = r_tr.mean(axis=0)
        r_tr_c = r_tr - r_mean
        r_te_c = r_te - r_mean
        gram = r_tr_c.T @ r_tr_c + alpha_ridge * np.eye(p_dim)

        w_y = np.linalg.solve(gram, r_tr_c.T @ (y_tr - y_tr.mean()))
        y_hat = y_tr.mean() + r_te_c @ w_y
        y_tilde[test_idx] = y[test_idx] - y_hat

        w_z = np.linalg.solve(gram, r_tr_c.T @ (z_tr - z_tr.mean(axis=0)))
        z_hat = z_tr.mean(axis=0) + r_te_c @ w_z
        z_tilde[test_idx] = z[test_idx] - z_hat

    theta_hat = np.linalg.solve(z_tilde.T @ z_tilde, z_tilde.T @ y_tilde)
    resid = y_tilde - z_tilde @ theta_hat
    j_inv = np.linalg.inv((z_tilde.T @ z_tilde) / n)
    meat = ((z_tilde * resid[:, None]).T @ (z_tilde * resid[:, None])) / n
    dof_scale = n / max(n - p_dim - z.shape[1], 1)
    vcov = dof_scale * (j_inv @ meat @ j_inv) / n
    se = np.sqrt(np.diag(vcov))
    r2_z = 1.0 - np.var(z_tilde[:, 0]) / np.var(z[:, 0])
    return theta_hat, se, float(r2_z)


def run_simulations(n_sims=400, d=64, m=80, n=600, theta_true=1.5):
    stage1_rows = {
        reg: {
            "r2": [],
            "cos_pca_true": [],
            "cos_mean_true": [],
            "split_pca": [],
            "split_mean": [],
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
        # Second concept C2 has inner product cos(pi/4) = 0.707 with C1
        u_c2 = np.cos(np.pi / 4) * u_c1 + np.sin(np.pi / 4) * basis[:, 3]

        # --- Stage 1 Regimes ---
        # Regime A: Clean rank-1 shift + isotropic small edit noise
        amp_a = 1.0 + 0.2 * rng.randn(m)
        dh_clean = np.outer(u_c1, amp_a) + 0.15 * rng.randn(d, m)
        s1_clean = estimate_stage1(dh_clean, rng)
        stage1_rows["clean"]["r2"].append(s1_clean["r2_1"])
        stage1_rows["clean"]["cos_pca_true"].append(
            principal_angle_cos(s1_clean["u_pca"], u_c1)
        )
        stage1_rows["clean"]["cos_mean_true"].append(
            principal_angle_cos(s1_clean["u_mean"], u_c1)
        )
        stage1_rows["clean"]["split_pca"].append(s1_clean["split_cos_pca"])
        stage1_rows["clean"]["split_mean"].append(s1_clean["split_cos_mean"])

        # Regime B: Mean shift along u_c1, high zero-mean noise along u_nuis
        dh_noisy = (
            np.outer(u_c1, 1.0 + 0.15 * rng.randn(m))
            + np.outer(u_nuis, 1.8 * rng.randn(m))
            + 0.15 * rng.randn(d, m)
        )
        s1_noisy = estimate_stage1(dh_noisy, rng)
        row_b = stage1_rows["high_var_noise"]
        row_b["r2"].append(s1_noisy["r2_1"])
        cp_b = principal_angle_cos(s1_noisy["u_pca"], u_c1)
        cm_b = principal_angle_cos(s1_noisy["u_mean"], u_c1)
        row_b["cos_pca_true"].append(cp_b)
        row_b["cos_mean_true"].append(cm_b)
        row_b["split_pca"].append(s1_noisy["split_cos_pca"])
        row_b["split_mean"].append(s1_noisy["split_cos_mean"])

        # Regime C: Dispersed edit across 15 random directions (no subspace)
        rand_dirs = orthonormal_columns(d, 15, rng)
        dh_curved = rand_dirs @ rng.randn(15, m) + 0.2 * rng.randn(d, m)
        s1_curved = estimate_stage1(dh_curved, rng)
        stage1_rows["curved_no_subspace"]["r2"].append(s1_curved["r2_1"])
        stage1_rows["curved_no_subspace"]["cos_pca_true"].append(
            principal_angle_cos(s1_curved["u_pca"], u_c1)
        )
        stage1_rows["curved_no_subspace"]["cos_mean_true"].append(
            principal_angle_cos(s1_curved["u_mean"], u_c1)
        )
        stage1_rows["curved_no_subspace"]["split_pca"].append(
            s1_curved["split_cos_pca"]
        )
        stage1_rows["curved_no_subspace"]["split_mean"].append(
            s1_curved["split_cos_mean"]
        )

        # --- Stage 2 Observational DGP (representation-level model Eq. 5) ---
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
        # True structural linear functional on h_i
        beta_h = theta_true * u_c1 + 1.2 * u_conf + 1.0 * u_nuis
        y_obs = h_obs @ beta_h + rng.randn(n)

        # 1. Naive OLS of Y on Z_mean without controlling for R
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
        cov_naive = abs(err_naive) <= 1.96 * se_naive
        stage2_rows["naive_ols"]["bias"].append(err_naive)
        stage2_rows["naive_ols"]["rmse"].append(err_naive**2)
        stage2_rows["naive_ols"]["cov"].append(cov_naive)
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
            y_obs,
            z_mean,
            r_mean,
            alpha_ridge=1e-4,
        )
        err_m = float(th_m[0]) - theta_mean_target
        cov_m = abs(err_m) <= 1.96 * float(se_m[0])
        stage2_rows["dml_mean_dir"]["bias"].append(err_m)
        stage2_rows["dml_mean_dir"]["rmse"].append(err_m**2)
        stage2_rows["dml_mean_dir"]["cov"].append(cov_m)
        stage2_rows["dml_mean_dir"]["r2_z"].append(r2_m)

        # --- Multi-concept DGP (two non-orthogonal concepts, cos = 0.707) ---
        theta2_true = -1.0
        u_multi = np.column_stack([u_c1, u_c2])
        utu_inv = np.linalg.inv(u_multi.T @ u_multi)
        # Dual basis U_dual = U (U^T U)^(-1) so U_dual^T U = I_2
        u_dual = u_multi @ utu_inv
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

    # Write Table 1 (Stage 1 diagnostics)
    t1_header = (
        r"Minimal-Pair Regime & $R_1^2$ & "
        r"$\cos(\hat{u}_{\rm PCA}, u_C^*)$ & "
        r"$\cos(\hat{u}_{\rm mean}, u_C^*)$ & "
        r"Split (PCA) & Split (Mean) \\"
    )
    t1_lines = [
        r"\begin{tabular}{@{}l rrrrr@{}}",
        r"\toprule",
        t1_header,
        r"\midrule",
    ]
    labels_s1 = {
        "clean": "Clean low-rank shift",
        "high_var_noise": "High-variance nuisance edit noise",
        "curved_no_subspace": "Dispersed / no subspace (reject)",
    }
    for key, label in labels_s1.items():
        d_row = stage1_rows[key]
        r2_val = np.mean(d_row["r2"])
        cp_val = np.mean(d_row["cos_pca_true"])
        cm_val = np.mean(d_row["cos_mean_true"])
        sp_val = np.mean(d_row["split_pca"])
        sm_val = np.mean(d_row["split_mean"])
        t1_lines.append(
            f"{label} & {r2_val:.2f} & {cp_val:.2f} & "
            f"{cm_val:.2f} & {sp_val:.2f} & {sm_val:.2f} \\\\"
        )
    t1_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABS / "table1_stage1.tex").write_text("\n".join(t1_lines) + "\n")

    # Write Table 2 (Stage 2 DML estimation)
    t2_header = (
        r"Estimator / Projection Specification & "
        r"Bias & RMSE & 95\% Cov. & $R^2(Z \mid R)$ \\"
    )
    t2_lines = [
        r"\begin{tabular}{@{}l rrrr@{}}",
        r"\toprule",
        t2_header,
        r"\midrule",
    ]
    lbl_pca = "DML with uncentered PCA $\\hat{u}_{\\rm PCA}$ (noise-corrupted)"
    lbl_mean = "DML with mean-shift $\\hat{u}_{\\rm mean}$ (single concept)"
    s_cos = "($\\cos\\theta = 0.71$)"
    lbl_inc = f"Multi-concept DML, naive $U_k^\\top h_i$ {s_cos}"
    lbl_obl = "Multi-concept DML, oblique $(U_k^\\top U_k)^{-1}U_k^\\top h_i$"
    labels_s2 = {
        "naive_ols": "Unadjusted OLS (no residual control $R_i$)",
        "dml_uncentered_pca": lbl_pca,
        "dml_mean_dir": lbl_mean,
        "dml_multi_incorrect": lbl_inc,
        "dml_multi_oblique": lbl_obl,
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

    print("\n".join(t1_lines))
    print()
    print("\n".join(t2_lines))


if __name__ == "__main__":
    run_simulations()
