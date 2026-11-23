"""Empirical application and placebo suite on Stanford Politeness Corpora.

Generates tabs/table3_empirical_stage1.tex, tabs/table4_empirical_stage2.tex,
tabs/macros_emp.tex, and figs/fig2_empirical.pdf from one computed source.
"""

import json
import re
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import ACCENT, AMBER, MUTED, TEAL, WARN, apply_style
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
TABS = ROOT / "tabs"
FIGS = ROOT / "figs"
TABS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
DATA_DIR = Path.home() / "AgentWork/scratch/finite-sample/steer/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_URLS = {
    "wikipedia-politeness-corpus": (
        "https://zissou.infosci.cornell.edu/convokit/datasets/"
        "wikipedia-politeness-corpus/wikipedia-politeness-corpus.zip"
    ),
    "stack-exchange-politeness-corpus": (
        "https://zissou.infosci.cornell.edu/convokit/datasets/"
        "stack-exchange-politeness-corpus/"
        "stack-exchange-politeness-corpus.zip"
    ),
}

POLITE_RE = re.compile(
    r"\b(please|thanks|thank you|kindly|appreciate|could you|would you)\b",
    re.IGNORECASE,
)

WORKPLACE_BASE = [
    (
        "Send me the quarterly budget spreadsheet by Friday.",
        "Could you please send me the quarterly budget spreadsheet by Friday?",
    ),
    (
        "Fix the formatting error on slide four.",
        "Would you mind fixing the formatting error on slide four?",
    ),
    (
        "Update the client contact list before the meeting.",
        "Could you please update the client contact list before the meeting?",
    ),
    (
        "Share the notes from yesterday's planning call.",
        "Would you be willing to share the notes from yesterday's call?",
    ),
    (
        "Confirm the room reservation for Tuesday morning.",
        "Could you please confirm the room reservation for Tuesday morning?",
    ),
    (
        "Review the draft contract and send back comments.",
        "I would appreciate it if you could review the draft contract.",
    ),
    (
        "Forward the invoice to the accounting department.",
        "Could you kindly forward the invoice to the accounting department?",
    ),
    (
        "Add my name to the project distribution list.",
        "Would you mind adding my name to the project distribution list?",
    ),
    (
        "Check whether the vendor shipped the equipment.",
        "Could you please check whether the vendor shipped the equipment?",
    ),
    (
        "Reschedule our one-on-one to Thursday afternoon.",
        "Would it be possible to reschedule our meeting to Thursday?",
    ),
    (
        "Upload the signed agreement to the shared folder.",
        "Could you please upload the signed agreement to the shared folder?",
    ),
    (
        "Clarify the timeline for the product launch.",
        "Could you kindly clarify the timeline for the product launch?",
    ),
    (
        "Provide the final headcount for the workshop.",
        "Would you please provide the final headcount for the workshop?",
    ),
    (
        "Verify the figures in section two of the memo.",
        "Could you please verify the figures in section two of the memo?",
    ),
    (
        "Send the login credentials for the staging server.",
        "Would you mind sending the login credentials for the server?",
    ),
    (
        "Approve the travel reimbursement form today.",
        "Could you please approve the travel reimbursement form today?",
    ),
    (
        "Explain why the delivery was delayed.",
        "Could you kindly explain why the delivery was delayed?",
    ),
    (
        "Replace the outdated chart in the appendix.",
        "Would you mind replacing the outdated chart in the appendix?",
    ),
    (
        "Give me access to the analytics dashboard.",
        "Could you please give me access to the analytics dashboard?",
    ),
    (
        "Summarize the feedback from the focus group.",
        "Would you be able to summarize the feedback from the focus group?",
    ),
]

TECH_BASE = [
    (
        "Show me how to parse a JSON file in Python.",
        "Could you please show me how to parse a JSON file in Python?",
    ),
    (
        "Explain why this SQL query returns duplicate rows.",
        "Would you mind explaining why this SQL query returns duplicates?",
    ),
    (
        "Tell me how to reset a local Git branch.",
        "Could you please tell me how to reset a local Git branch?",
    ),
    (
        "Point out the memory leak in this C++ function.",
        "I would appreciate it if you could point out this memory leak.",
    ),
    (
        "Fix the regular expression so it matches dates.",
        "Could you kindly help fix the regex so it matches dates?",
    ),
    (
        "Give an example of an async HTTP request.",
        "Would you mind giving an example of an async HTTP request?",
    ),
    (
        "Explain the difference between inner and left joins.",
        "Could you please explain the difference between these SQL joins?",
    ),
    (
        "Tell me why Docker fails to mount this volume.",
        "Would you be able to tell me why Docker fails to mount this?",
    ),
    (
        "Show how to sort a dictionary by its values.",
        "Could you please show how to sort a dictionary by its values?",
    ),
    (
        "Check whether this regex handles Unicode characters.",
        "Would you mind checking whether this regex handles Unicode?",
    ),
    (
        "Help me debug this null pointer exception.",
        "Could you please help me debug this null pointer exception?",
    ),
    (
        "Explain how to configure SSL on Nginx.",
        "Could you kindly explain how to configure SSL on Nginx?",
    ),
    (
        "Give the command to list open TCP ports.",
        "Could you please give the command to list open TCP ports?",
    ),
    (
        "Show how to merge two pandas DataFrames cleanly.",
        "Would you mind showing how to merge two pandas DataFrames?",
    ),
    (
        "Tell me what this compiler warning means.",
        "Could you please tell me what this compiler warning means?",
    ),
    (
        "Provide a minimal test case for this bug.",
        "Would you be willing to provide a minimal test case for this bug?",
    ),
    (
        "Explain why this recursive function overflows.",
        "Could you kindly explain why this recursive function overflows?",
    ),
    (
        "Show how to set an environment variable in zsh.",
        "Could you please show how to set an environment variable in zsh?",
    ),
    (
        "Clarify how PyTorch handles gradient accumulation.",
        "Would you mind clarifying how PyTorch handles gradients?",
    ),
    (
        "Tell me how to export this table to CSV.",
        "Could you please tell me how to export this table to CSV?",
    ),
]

EVERYDAY_BASE = [
    (
        "Pass me the salt and pepper.",
        "Could you please pass me the salt and pepper?",
    ),
    (
        "Turn down the music in the living room.",
        "Would you mind turning down the music in the living room?",
    ),
    (
        "Close the window before it starts raining.",
        "Could you please close the window before it starts raining?",
    ),
    (
        "Hold the elevator door for a second.",
        "Would you mind holding the elevator door for a second?",
    ),
    (
        "Pick up milk on your way home.",
        "Could you please pick up milk on your way home?",
    ),
    (
        "Tell me what time the library closes tonight.",
        "Could you kindly tell me what time the library closes tonight?",
    ),
    (
        "Move your bag off the empty seat.",
        "Would you mind moving your bag off the empty seat?",
    ),
    (
        "Water the plants while I am away.",
        "I would appreciate it if you could water the plants while I am away.",
    ),
    (
        "Lend me a pen to sign this receipt.",
        "Could you please lend me a pen to sign this receipt?",
    ),
    (
        "Let me know when the package arrives.",
        "Would you please let me know when the package arrives?",
    ),
    (
        "Wait five minutes until I finish packing.",
        "Would you mind waiting five minutes until I finish packing?",
    ),
    (
        "Bring the receipt back from the hardware store.",
        "Could you please bring the receipt back from the hardware store?",
    ),
    (
        "Check if the back door is locked.",
        "Could you kindly check if the back door is locked?",
    ),
    (
        "Text me the address of the restaurant.",
        "Would you please text me the address of the restaurant?",
    ),
    (
        "Save me a seat near the aisle.",
        "Could you please save me a seat near the aisle?",
    ),
    (
        "Keep the noise down after midnight.",
        "Would you mind keeping the noise down after midnight?",
    ),
    (
        "Help carry these groceries up the stairs.",
        "Would you be able to help carry these groceries up the stairs?",
    ),
    (
        "Tell me which bus goes to the train station.",
        "Could you please tell me which bus goes to the train station?",
    ),
    (
        "Leave the spare key under the mat.",
        "Could you kindly leave the spare key under the mat?",
    ),
    (
        "Call me once you reach the airport.",
        "Would you please call me once you reach the airport?",
    ),
]

POLITE_PAIRS = WORKPLACE_BASE + TECH_BASE + EVERYDAY_BASE
POLITE_DOMAINS = (
    ["workplace"] * len(WORKPLACE_BASE)
    + ["technical"] * len(TECH_BASE)
    + ["everyday"] * len(EVERYDAY_BASE)
)

URGENT_SUFFIX = " immediately, this is urgent."
URGENT_PAIRS = []
for neg, _ in POLITE_PAIRS:
    URGENT_PAIRS.append((neg, f"{neg.rstrip('.?')}{URGENT_SUFFIX}"))

PLURAL_PAIRS = [
    ("I checked the report on the desk.", "I checked the reports on desks."),
    ("She read the article this morning.", "She read the articles today."),
    ("He fixed the bug in the module.", "He fixed the bugs in modules."),
    ("The student opened the notebook.", "The students opened notebooks."),
    ("We reviewed the proposal yesterday.", "We reviewed proposals recently."),
    ("The driver parked the truck outside.", "The drivers parked trucks."),
    ("I bought a book at the shop.", "I bought books at the shops."),
    ("The engineer tested the sensor.", "The engineers tested sensors."),
    ("She mailed the letter from work.", "She mailed letters from work."),
    ("The chef prepared the dish quickly.", "The chefs prepared dishes."),
    ("He edited the paragraph in draft.", "He edited paragraphs in drafts."),
    ("The analyst checked the table.", "The analysts checked the tables."),
    ("We planted a tree along the path.", "We planted trees along paths."),
    ("The doctor examined the chart.", "The doctors examined the charts."),
    ("I downloaded the file from server.", "I downloaded files from servers."),
    ("She answered the question clearly.", "She answered questions clearly."),
    ("The builder inspected the bridge.", "The builders inspected bridges."),
    ("He replaced the battery in clock.", "He replaced batteries in clocks."),
    ("The teacher graded the essay.", "The teachers graded the essays."),
    ("We visited the museum last week.", "We visited museums last week."),
    ("The clerk stamped the form.", "The clerks stamped the forms."),
    ("I updated the slide in the deck.", "I updated slides in the decks."),
    ("She organized the folder on drive.", "She organized folders on drives."),
    ("He painted the wall in hallway.", "He painted walls in hallways."),
    ("The baker sliced the loaf.", "The bakers sliced the loaves."),
    ("We booked the ticket online.", "We booked the tickets online."),
    ("The guard locked the gate.", "The guards locked the gates."),
    ("I cleaned the lens on camera.", "I cleaned lenses on cameras."),
    ("She solved the equation on board.", "She solved equations on boards."),
    ("He wrapped the gift in paper.", "He wrapped the gifts in paper."),
]


def ensure_corpus(name):
    utt_path = DATA_DIR / name / "utterances.jsonl"
    if not utt_path.exists():
        zip_path = DATA_DIR / f"{name}.zip"
        urllib.request.urlretrieve(CORPUS_URLS[name], zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
    texts, scores = [], []
    for line in utt_path.read_text().splitlines():
        row = json.loads(line)
        txt = row["text"].strip()
        sc = row["meta"].get("Normalized Score")
        if txt and sc is not None:
            texts.append(txt)
            scores.append(float(sc))
    y = np.array(scores, dtype=float)
    y = (y - y.mean()) / y.std()
    return texts, y


def get_embeddings(model, cache_name, texts):
    cache = DATA_DIR / f"emb_{cache_name}.npy"
    if cache.exists():
        return np.load(cache)
    emb = model.encode(texts, batch_size=128, show_progress_bar=False)
    np.save(cache, emb)
    return emb


def principal_angle_cos(u_a, u_b):
    denom = np.linalg.norm(u_a) * np.linalg.norm(u_b)
    return float(np.abs(np.dot(u_a, u_b)) / denom)


def stage1_diagnostics(delta_h, domains=None, seed=42):
    rng = np.random.RandomState(seed)
    m, _ = delta_h.shape
    s_mat = (delta_h.T @ delta_h) / m
    evals, evecs = np.linalg.eigh(s_mat)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    u_pca = evecs[:, idx[0]]

    mean_vec = delta_h.mean(axis=0)
    u_mean = mean_vec / np.linalg.norm(mean_vec)
    if np.dot(u_pca, u_mean) < 0:
        u_pca = -u_pca

    r2_1 = float(evals[0] / np.sum(evals))
    align = principal_angle_cos(u_pca, u_mean)

    half = m // 2
    split_mean, split_pca = [], []
    for _ in range(50):
        perm = rng.permutation(m)
        h1, h2 = delta_h[perm[:half]], delta_h[perm[half:]]
        c_m = principal_angle_cos(h1.mean(axis=0), h2.mean(axis=0))
        split_mean.append(c_m)
        _, v1 = np.linalg.eigh((h1.T @ h1) / len(h1))
        _, v2 = np.linalg.eigh((h2.T @ h2) / len(h2))
        split_pca.append(principal_angle_cos(v1[:, -1], v2[:, -1]))

    ood_cos = None
    if domains is not None:
        uniq = sorted(set(domains))
        dom_vecs = {}
        for g in uniq:
            mask = np.array([dg == g for dg in domains])
            vec = delta_h[mask].mean(axis=0)
            dom_vecs[g] = vec / np.linalg.norm(vec)
        pairs_cos = []
        for i, g1 in enumerate(uniq):
            nxt = i + 1
            for g2 in uniq[nxt:]:
                c_g = principal_angle_cos(dom_vecs[g1], dom_vecs[g2])
                pairs_cos.append(c_g)
        ood_cos = float(np.mean(pairs_cos))

    return {
        "u_mean": u_mean,
        "u_pca": u_pca,
        "r2_1": r2_1,
        "align_pca_mean": align,
        "split_mean": float(np.mean(split_mean)),
        "split_pca": float(np.mean(split_pca)),
        "ood_cos": ood_cos,
    }


def dml_stage2(y, z, r_mat, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    if z.ndim == 1:
        z = z[:, None]
    k = z.shape[1]

    r_c = r_mat - r_mat.mean(axis=0)
    _, s_r, vt_r = np.linalg.svd(r_c, full_matrices=False)
    keep = s_r > 1e-6
    r_low = r_c @ vt_r[keep].T

    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    y_tilde = np.zeros(n)
    z_tilde = np.zeros((n, k))
    alphas = np.logspace(-2, 4, 13)

    for f_idx in range(n_folds):
        val_idx = folds[f_idx]
        other = [folds[j] for j in range(n_folds) if j != f_idx]
        tr_idx = np.concatenate(other)
        ry = RidgeCV(alphas=alphas).fit(r_low[tr_idx], y[tr_idx])
        y_tilde[val_idx] = y[val_idx] - ry.predict(r_low[val_idx])
        for j in range(k):
            rz = RidgeCV(alphas=alphas).fit(r_low[tr_idx], z[tr_idx, j])
            z_tilde[val_idx, j] = z[val_idx, j] - rz.predict(r_low[val_idx])

    theta = np.linalg.solve(z_tilde.T @ z_tilde, z_tilde.T @ y_tilde)
    resid = y_tilde - z_tilde @ theta
    j_inv = np.linalg.inv((z_tilde.T @ z_tilde) / n)
    meat = ((z_tilde * resid[:, None]).T @ (z_tilde * resid[:, None])) / n
    vcov = (j_inv @ meat @ j_inv) / n
    se = np.sqrt(np.diag(vcov))
    r2_z = []
    for j in range(k):
        r2_z.append(float(1.0 - np.var(z_tilde[:, j]) / np.var(z[:, j])))
    return theta, se, r2_z


def ols_unadjusted(y, z):
    z_c = z - z.mean()
    y_c = y - y.mean()
    beta = float(np.dot(z_c, y_c) / np.dot(z_c, z_c))
    resid = y_c - beta * z_c
    se = float(np.sqrt(np.sum((z_c * resid) ** 2) / (np.dot(z_c, z_c) ** 2)))
    return beta, se


def plot_empirical(rw, rs):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

    rows = [
        ("Placebo: Plurality (DML)", "dml_plu", MUTED, True),
        ("Placebo: Plurality (OLS)", "ols_plu", WARN, False),
        ("Implicit-only (0 keywords, DML)", "dml_implicit", TEAL, True),
        ("Cross-corpus probe (DML)", "dml_probe", AMBER, True),
        ("Oblique multi-DML (Pol | Urg)", "dml_obl", ACCENT, True),
        ("Minimal-pair DML (ours)", "dml_pol", ACCENT, True),
        ("Unadjusted OLS (no R)", "ols_pol", WARN, False),
    ]
    y_pos = np.arange(len(rows))

    for ax, res, title in [
        (ax1, rw, "A. Wikipedia requests (N = 4,353)"),
        (ax2, rs, "B. Stack Exchange requests (N = 6,603)"),
    ]:
        ax.axvline(0.0, color=MUTED, linestyle="--", linewidth=0.9)
        ax.axhline(1.5, color="#DDDDDD", linestyle="-", linewidth=0.7)
        for idx, (_, key, col, is_dml) in enumerate(rows):
            if is_dml:
                est = float(res[key][0][0])
                se = float(res[key][1][0])
            else:
                est, se = res[key]
            ax.plot(
                [est - 1.96 * se, est + 1.96 * se],
                [idx, idx],
                color=col,
                linewidth=2.0,
            )
            ax.plot(est, idx, "o", color=col, markersize=4.8)
        ax.set_xlim(-0.08, 0.50)
        ax.set_xlabel("Estimated effect (SD per 1 SD in Z)")
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGS / "fig2_empirical.pdf")
    plt.close(fig)


def run_empirical():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode_diffs(pairs):
        neg = model.encode([p[0] for p in pairs], show_progress_bar=False)
        pos = model.encode([p[1] for p in pairs], show_progress_bar=False)
        return pos - neg

    dh_pol = encode_diffs(POLITE_PAIRS)
    dh_urg = encode_diffs(URGENT_PAIRS)
    dh_plu = encode_diffs(PLURAL_PAIRS)
    rng = np.random.RandomState(7)
    signs = rng.choice([-1.0, 1.0], size=(len(POLITE_PAIRS), 1))
    dh_rand = dh_pol * signs

    s1_pol = stage1_diagnostics(dh_pol, POLITE_DOMAINS)
    s1_urg = stage1_diagnostics(dh_urg, POLITE_DOMAINS)
    s1_plu = stage1_diagnostics(dh_plu)
    s1_rnd = stage1_diagnostics(dh_rand, POLITE_DOMAINS)

    # Write Table 3
    t3_hdr = (
        r"Concept / Placebo & $R_1^2$ & "
        r"$\cos(\hat{u}_{\mathrm{PCA}}, \hat{u}_{\mathrm{mean}})$ & "
        r"Split (Mean) & OOD Transfer & Gate \\"
    )
    t3_lines = [
        r"\begin{tabular}{@{}l rrrrc@{}}",
        r"\toprule",
        t3_hdr,
        r"\midrule",
    ]
    for lbl, res in [
        ("Politeness (target, $m=60$)", s1_pol),
        ("Urgency (2nd concept, $m=60$)", s1_urg),
        ("Plurality (negative control, $m=30$)", s1_plu),
        ("Sign-flipped placebo ($m=60$)", s1_rnd),
    ]:
        ood_s = f"{res['ood_cos']:.2f}" if res["ood_cos"] is not None else "--"
        gate = "Pass" if res["split_mean"] >= 0.70 else "Reject"
        t3_lines.append(
            f"{lbl} & {res['r2_1']:.2f} & {res['align_pca_mean']:.2f} & "
            f"{res['split_mean']:.2f} & {ood_s} & {gate} \\\\"
        )
    t3_lines.extend([r"\bottomrule", r"\end{tabular}"])
    t3_path = TABS / "table3_empirical_stage1.tex"
    t3_path.write_text("\n".join(t3_lines) + "\n")

    # Stage 2 on both real corpora
    t_wiki, y_wiki = ensure_corpus("wikipedia-politeness-corpus")
    t_se, y_se = ensure_corpus("stack-exchange-politeness-corpus")
    h_wiki = get_embeddings(model, "wiki", t_wiki)
    h_se = get_embeddings(model, "se", t_se)

    u_pol = s1_pol["u_mean"]
    u_urg = s1_urg["u_mean"]
    u_plu = s1_plu["u_mean"]
    cos_pol_urg = principal_angle_cos(u_pol, u_urg)

    alphas = np.logspace(-2, 4, 13)
    w_wiki = RidgeCV(alphas=alphas).fit(h_wiki, y_wiki).coef_
    u_probe_wiki = w_wiki / np.linalg.norm(w_wiki)
    w_se = RidgeCV(alphas=alphas).fit(h_se, y_se).coef_
    u_probe_se = w_se / np.linalg.norm(w_se)

    # Pooled confounder-leakage metrics (text length & domain)
    h_all = np.vstack([h_wiki, h_se])
    dom_all = np.concatenate([np.zeros(len(t_wiki)), np.ones(len(t_se))])
    len_all = np.log1p(
        np.array([len(t.split()) for t in t_wiki + t_se], dtype=float)
    )
    r_len_ours = abs(float(np.corrcoef(h_all @ u_pol, len_all)[0, 1]))
    r_len_pw = abs(float(np.corrcoef(h_all @ u_probe_wiki, len_all)[0, 1]))
    r_len_ps = abs(float(np.corrcoef(h_all @ u_probe_se, len_all)[0, 1]))

    hc_w = h_wiki - h_wiki.mean(axis=0)
    _, _, vt_w = np.linalg.svd(hc_w, full_matrices=False)
    u_pc_wiki = vt_w[0]
    pc_pair_acc = 100.0 * float(np.mean((dh_pol @ u_pc_wiki) > 0))
    pc_dom_r = abs(float(np.corrcoef(h_all @ u_pc_wiki, dom_all)[0, 1]))

    def eval_corpus(texts, h_mat, y_vec, u_probe_ood):
        _, d = h_mat.shape
        out = {}

        zp_raw = h_mat @ u_pol
        zp = zp_raw / np.std(zp_raw)
        rp = h_mat - np.outer(zp_raw, u_pol)
        out["ols_pol"] = ols_unadjusted(y_vec, zp)
        out["dml_pol"] = dml_stage2(y_vec, zp, rp)

        u_m = np.column_stack([u_pol, u_urg])
        u_dual = u_m @ np.linalg.inv(u_m.T @ u_m)
        zo_raw = h_mat @ u_dual
        ro = h_mat - zo_raw @ u_m.T
        zo = zo_raw / np.std(zo_raw, axis=0)
        out["dml_obl"] = dml_stage2(y_vec, zo, ro)

        zn_raw = h_mat @ u_m
        rn = h_mat - zn_raw @ u_m.T
        zn = zn_raw / np.std(zn_raw, axis=0)
        out["dml_nav"] = dml_stage2(y_vec, zn, rn)

        zpr_raw = h_mat @ u_probe_ood
        zpr = zpr_raw / np.std(zpr_raw)
        rpr = h_mat - np.outer(zpr_raw, u_probe_ood)
        out["dml_probe"] = dml_stage2(y_vec, zpr, rpr)

        counts = [len(POLITE_RE.findall(t)) for t in texts]
        lex_cnt = np.array(counts, dtype=float)
        no_kw = lex_cnt == 0
        hs, ys = h_mat[no_kw], y_vec[no_kw]
        zs_raw = hs @ u_pol
        zs = zs_raw / np.std(zs_raw)
        rs = hs - np.outer(zs_raw, u_pol)
        out["dml_implicit"] = dml_stage2(ys, zs, rs)
        out["n_implicit"] = int(no_kw.sum())
        out["pct_implicit"] = 100.0 * float(np.mean(no_kw))

        zpl_raw = h_mat @ u_plu
        zpl = zpl_raw / np.std(zpl_raw)
        rpl = h_mat - np.outer(zpl_raw, u_plu)
        out["ols_plu"] = ols_unadjusted(y_vec, zpl)
        out["dml_plu"] = dml_stage2(y_vec, zpl, rpl)

        rng_p = np.random.RandomState(123)
        p_rej = []
        for p_i in range(20):
            y_perm = rng_p.permutation(y_vec)
            th_p, se_p, _ = dml_stage2(y_perm, zp, rp, seed=p_i)
            p_rej.append(abs(float(th_p[0]) / float(se_p[0])) > 1.96)
        out["perm_rej"] = 100.0 * float(np.mean(p_rej))

        ols_rej, dml_rej = [], []
        for r_i in range(20):
            v = rng_p.randn(d)
            v = v - np.dot(v, u_pol) * u_pol
            v = v / np.linalg.norm(v)
            zr_raw = h_mat @ v
            zr = zr_raw / np.std(zr_raw)
            rr = h_mat - np.outer(zr_raw, v)
            bo, seo = ols_unadjusted(y_vec, zr)
            ols_rej.append(abs(bo / seo) > 1.96)
            td, sed, _ = dml_stage2(y_vec, zr, rr, seed=r_i)
            dml_rej.append(abs(float(td[0]) / float(sed[0])) > 1.96)
        out["rand_ols_rej"] = 100.0 * float(np.mean(ols_rej))
        out["rand_dml_rej"] = 100.0 * float(np.mean(dml_rej))
        return out

    rw = eval_corpus(t_wiki, h_wiki, y_wiki, u_probe_se)
    rs = eval_corpus(t_se, h_se, y_se, u_probe_wiki)

    def fmt(pair):
        b, se = pair
        return f"{b:+.3f} ({se:.3f})"

    def fmt_dml(tup):
        th, se, r2 = tup
        return f"{float(th[0]):+.3f} ({float(se[0]):.3f})", f"{r2[0]:.2f}"

    def dml_row(label, key):
        w_est, w_r2 = fmt_dml(rw[key])
        s_est, s_r2 = fmt_dml(rs[key])
        return f"{label} & {w_est} & {w_r2} & {s_est} & {s_r2} \\\\"

    def ols_row(label, key):
        return f"{label} & {fmt(rw[key])} & -- & {fmt(rs[key])} & -- \\\\"

    def pct_row(label, key):
        return f"{label} & {rw[key]:.1f}\\% & -- & {rs[key]:.1f}\\% & -- \\\\"

    mc5 = r"\multicolumn{5}{@{}l}"
    pan_a = mc5 + r"{\emph{Panel A: Target Concept (Politeness, 1 SD)}} \\"
    pan_b = mc5 + r"{\emph{Panel B: Comparisons Against Baselines}} \\"
    pan_c = mc5 + r"{\emph{Panel C: Placebo Checks (Expected Null)}} \\"
    t4_lines = [
        r"\begin{tabular}{@{}l r r r r@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Wikipedia ($N=4{,}353$)} & "
        r"\multicolumn{2}{c}{Stack Exchange ($N=6{,}603$)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Specification / Placebo Check & $\hat{\theta}$ (SE) & "
        r"$R^2(Z{\mid}R)$ & $\hat{\theta}$ (SE) & $R^2(Z{\mid}R)$ \\",
        r"\midrule",
        pan_a,
        ols_row("Unadjusted OLS (no $R_i$ control)", "ols_pol"),
        dml_row(
            "5-fold DML (minimal-pair $\\hat{u}_{\\mathrm{mean}}$)",
            "dml_pol",
        ),
        dml_row("Oblique Multi-DML (Politeness $|$ Urgency)", "dml_obl"),
        dml_row("Naive Multi-DML (broken $U_k^\\top h_i$)", "dml_nav"),
        r"\midrule",
        pan_b,
        dml_row("Cross-corpus supervised probe (DML)", "dml_probe"),
        dml_row("Implicit-only subset (0 keywords; ours)", "dml_implicit"),
        r"\midrule",
        pan_c,
        ols_row("Negative-control concept: Plurality (OLS)", "ols_plu"),
        dml_row("Negative-control concept: Plurality (DML)", "dml_plu"),
        pct_row("Outcome permutations (20$\\times$, reject@5\\%)", "perm_rej"),
        pct_row(
            "Random $\\perp$ directions (20$\\times$, OLS reject@5\\%)",
            "rand_ols_rej",
        ),
        pct_row(
            "Random $\\perp$ directions (20$\\times$, DML reject@5\\%)",
            "rand_dml_rej",
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    t4_path = TABS / "table4_empirical_stage2.tex"
    t4_path.write_text("\n".join(t4_lines) + "\n")

    # Emit macros_emp.tex so prose never hand-types empirical numbers
    macros = [
        f"\\newcommand{{\\empNWiki}}{{{len(t_wiki):,}}}",
        f"\\newcommand{{\\empNSe}}{{{len(t_se):,}}}",
        f"\\newcommand{{\\empPolROne}}{{{s1_pol['r2_1']:.2f}}}",
        f"\\newcommand{{\\empPolSplit}}{{{s1_pol['split_mean']:.2f}}}",
        f"\\newcommand{{\\empPolOod}}{{{s1_pol['ood_cos']:.2f}}}",
        f"\\newcommand{{\\empUrgSplit}}{{{s1_urg['split_mean']:.2f}}}",
        f"\\newcommand{{\\empUrgOod}}{{{s1_urg['ood_cos']:.2f}}}",
        f"\\newcommand{{\\empPluSplit}}{{{s1_plu['split_mean']:.2f}}}",
        f"\\newcommand{{\\empRndROne}}{{{s1_rnd['r2_1']:.2f}}}",
        f"\\newcommand{{\\empRndSplit}}{{{s1_rnd['split_mean']:.2f}}}",
        f"\\newcommand{{\\empRndOod}}{{{s1_rnd['ood_cos']:.2f}}}",
        f"\\newcommand{{\\empCosPolUrg}}{{{cos_pol_urg:.2f}}}",
        f"\\newcommand{{\\empWikiOls}}{{{rw['ols_pol'][0]:+.3f}}}",
        f"\\newcommand{{\\empWikiDml}}{{{float(rw['dml_pol'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empWikiDmlSe}}{{{float(rw['dml_pol'][1][0]):.3f}}}",
        f"\\newcommand{{\\empWikiRTwo}}{{{rw['dml_pol'][2][0]:.2f}}}",
        f"\\newcommand{{\\empSeOls}}{{{rs['ols_pol'][0]:+.3f}}}",
        f"\\newcommand{{\\empSeOlsSe}}{{{rs['ols_pol'][1]:.3f}}}",
        f"\\newcommand{{\\empSeDml}}{{{float(rs['dml_pol'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empSeDmlSe}}{{{float(rs['dml_pol'][1][0]):.3f}}}",
        f"\\newcommand{{\\empSeRTwo}}{{{rs['dml_pol'][2][0]:.2f}}}",
        f"\\newcommand{{\\empWikiObl}}{{{float(rw['dml_obl'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empSeObl}}{{{float(rs['dml_obl'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empWikiNav}}{{{float(rw['dml_nav'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empSeNav}}{{{float(rs['dml_nav'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empWikiProbe}}{{{float(rw['dml_probe'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empWikiProbeSe}}{{{float(rw['dml_probe'][1][0]):.3f}}}",
        f"\\newcommand{{\\empWikiProbeRTwo}}{{{rw['dml_probe'][2][0]:.2f}}}",
        f"\\newcommand{{\\empSeProbe}}{{{float(rs['dml_probe'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empSeProbeRTwo}}{{{rs['dml_probe'][2][0]:.2f}}}",
        f"\\newcommand{{\\empWikiNoKwPct}}{{{rw['pct_implicit']:.0f}}}",
        f"\\newcommand{{\\empWikiNoKwN}}{{{rw['n_implicit']:,}}}",
        f"\\newcommand{{\\empSeNoKwPct}}{{{rs['pct_implicit']:.0f}}}",
        f"\\newcommand{{\\empSeNoKwN}}{{{rs['n_implicit']:,}}}",
        f"\\newcommand{{\\empWikiImp}}{{{float(rw['dml_implicit'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empWikiImpSe}}{{{float(rw['dml_implicit'][1][0]):.3f}}}",
        f"\\newcommand{{\\empSeImp}}{{{float(rs['dml_implicit'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empSeImpSe}}{{{float(rs['dml_implicit'][1][0]):.3f}}}",
        f"\\newcommand{{\\empPcPairAcc}}{{{pc_pair_acc:.1f}}}",
        f"\\newcommand{{\\empPcDomR}}{{{pc_dom_r:.2f}}}",
        f"\\newcommand{{\\empRLenOurs}}{{{r_len_ours:.2f}}}",
        f"\\newcommand{{\\empRLenProbeLo}}{{{min(r_len_pw, r_len_ps):.2f}}}",
        f"\\newcommand{{\\empRLenProbeHi}}{{{max(r_len_pw, r_len_ps):.2f}}}",
        f"\\newcommand{{\\empWikiPluOls}}{{{rw['ols_plu'][0]:+.3f}}}",
        f"\\newcommand{{\\empWikiPluOlsSe}}{{{rw['ols_plu'][1]:.3f}}}",
        f"\\newcommand{{\\empWikiPluDml}}{{{float(rw['dml_plu'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empSePluDml}}{{{float(rs['dml_plu'][0][0]):+.3f}}}",
        f"\\newcommand{{\\empWikiPluDmlSe}}{{{float(rw['dml_plu'][1][0]):.3f}}}",
        f"\\newcommand{{\\empWikiRandOls}}{{{rw['rand_ols_rej']:.0f}}}",
        f"\\newcommand{{\\empSeRandOls}}{{{rs['rand_ols_rej']:.0f}}}",
        f"\\newcommand{{\\empWikiRandDml}}{{{rw['rand_dml_rej']:.0f}}}",
        f"\\newcommand{{\\empSeRandDml}}{{{rs['rand_dml_rej']:.0f}}}",
    ]
    (TABS / "macros_emp.tex").write_text("\n".join(macros) + "\n")

    # Write Appendix Table A1 (representative external minimal pairs)
    ex_rows = [
        ("Politeness (Workplace)", WORKPLACE_BASE[0][0], WORKPLACE_BASE[0][1]),
        ("Politeness (Technical)", TECH_BASE[1][0], TECH_BASE[1][1]),
        ("Politeness (Everyday)", EVERYDAY_BASE[3][0], EVERYDAY_BASE[3][1]),
        ("Urgency (Workplace)", URGENT_PAIRS[0][0], URGENT_PAIRS[0][1]),
        ("Plurality (Control)", PLURAL_PAIRS[0][0], PLURAL_PAIRS[0][1]),
    ]
    ta1 = [
        r"\begin{tabular}{@{}l p{5.5cm} p{6.2cm}@{}}",
        r"\toprule",
        r"Concept (Domain) & Baseline ($T_j^-$) & Contrastive Edit ($T_j^+$) \\",
        r"\midrule",
    ]
    for dom, neg_s, pos_s in ex_rows:
        ta1.append(f"{dom} & {neg_s} & {pos_s} \\\\")
    ta1.extend([r"\bottomrule", r"\end{tabular}"])
    (TABS / "tableA1_pairs.tex").write_text("\n".join(ta1) + "\n")

    plot_empirical(rw, rs)


if __name__ == "__main__":
    run_empirical()
