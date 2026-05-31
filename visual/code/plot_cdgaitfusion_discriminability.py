#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, pairwise_distances, roc_curve
from sklearn.preprocessing import normalize, StandardScaler


def load_embeddings(npz_path, keep_numeric_labels=True):
    data = np.load(npz_path, allow_pickle=False)
    feats = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)
    types = data["types"].astype(str) if "types" in data.files else np.array([""] * len(labels))
    views = data["views"].astype(str) if "views" in data.files else np.array([""] * len(labels))
    feats = feats.reshape(feats.shape[0], -1)
    if keep_numeric_labels:
        keep = np.array([label.isdigit() for label in labels])
        feats = feats[keep]
        labels = labels[keep]
        types = types[keep]
        views = views[keep]
    return feats, labels, types, views


def group_by_label(labels):
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[str(label)].append(idx)
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


def choose_id_subset(labels, min_samples=2, max_ids=80, seed=2026):
    rng = np.random.default_rng(seed)
    groups = group_by_label(labels)
    valid = [lab for lab, idxs in groups.items() if len(idxs) >= min_samples]
    valid = np.array(sorted(valid))
    if len(valid) > max_ids:
        valid = rng.choice(valid, size=max_ids, replace=False)
        valid = np.array(sorted(valid))
    return valid, groups


def sample_balanced_indices(labels, max_ids=80, samples_per_id=2, seed=2026):
    rng = np.random.default_rng(seed)
    chosen_ids, groups = choose_id_subset(labels, samples_per_id, max_ids, seed)
    indices = []
    for lab in chosen_ids:
        idxs = groups[lab]
        if len(idxs) > samples_per_id:
            idxs = rng.choice(idxs, size=samples_per_id, replace=False)
        indices.extend(idxs.tolist())
    indices = np.array(indices, dtype=np.int64)
    order = np.argsort(labels[indices])
    return indices[order]


def sample_pairs(labels, max_pairs=120000, seed=2026):
    rng = np.random.default_rng(seed)
    groups = group_by_label(labels)
    valid_labels = [lab for lab, idxs in groups.items() if len(idxs) >= 2]
    if not valid_labels:
        raise RuntimeError("No identity has at least two samples; cannot build positive pairs.")

    pos_a, pos_b = [], []
    for _ in range(max_pairs // 2):
        lab = valid_labels[rng.integers(0, len(valid_labels))]
        a, b = rng.choice(groups[lab], size=2, replace=False)
        pos_a.append(a)
        pos_b.append(b)

    all_labels = np.array(sorted(groups.keys()))
    neg_a, neg_b = [], []
    for _ in range(max_pairs // 2):
        lab_a, lab_b = rng.choice(all_labels, size=2, replace=False)
        neg_a.append(rng.choice(groups[lab_a]))
        neg_b.append(rng.choice(groups[lab_b]))

    pair_a = np.array(pos_a + neg_a, dtype=np.int64)
    pair_b = np.array(pos_b + neg_b, dtype=np.int64)
    pair_y = np.array([1] * len(pos_a) + [0] * len(neg_a), dtype=np.int64)
    return pair_a, pair_b, pair_y


def cosine_distance_pairs(feats, pair_a, pair_b):
    feats_n = normalize(feats, norm="l2", axis=1)
    sim = np.sum(feats_n[pair_a] * feats_n[pair_b], axis=1)
    dist = 1.0 - sim
    return dist, sim


def compute_eer(fpr, tpr, thresholds):
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linewidth": 0.7,
        "savefig.bbox": "tight",
    })


def plot_distance_distribution(intra, inter, out_path):
    set_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bins = np.linspace(min(intra.min(), inter.min()), max(intra.max(), inter.max()), 80)
    ax.hist(inter, bins=bins, density=True, alpha=0.62, color="#d95f02", label="Inter-class pairs")
    ax.hist(intra, bins=bins, density=True, alpha=0.68, color="#1b9e77", label="Intra-class pairs")
    ax.axvline(intra.mean(), color="#087f5b", linestyle="--", linewidth=2, label=f"Intra mean = {intra.mean():.3f}")
    ax.axvline(inter.mean(), color="#b45309", linestyle="--", linewidth=2, label=f"Inter mean = {inter.mean():.3f}")
    ax.set_title("CDGaitFusion Intra-/Inter-class Distance Distribution")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_roc(scores, pair_y, out_path):
    set_style()
    fpr, tpr, thresholds = roc_curve(pair_y, scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_thr = compute_eer(fpr, tpr, thresholds)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.4, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.1)
    ax.scatter([eer], [1 - eer], s=62, color="#dc2626", zorder=3, label=f"EER = {eer:.4f}")
    ax.set_title("CDGaitFusion Verification ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return roc_auc, eer, eer_thr


def plot_distance_matrix(feats, labels, out_path, max_ids=30, samples_per_id=2, seed=2026):
    set_style()
    idx = sample_balanced_indices(labels, max_ids=max_ids, samples_per_id=samples_per_id, seed=seed)
    sub_feats = normalize(feats[idx], norm="l2", axis=1)
    sub_labels = labels[idx]
    dist = pairwise_distances(sub_feats, metric="cosine")

    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    im = ax.imshow(dist, cmap="magma_r", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine distance")
    ax.set_title("CDGaitFusion Pairwise Distance Matrix")
    ax.set_xlabel("Samples grouped by identity")
    ax.set_ylabel("Samples grouped by identity")
    tick_positions = []
    tick_labels = []
    last = None
    starts = []
    for pos, lab in enumerate(sub_labels):
        if lab != last:
            starts.append(pos)
            tick_positions.append(pos + (samples_per_id - 1) / 2)
            tick_labels.append(lab)
            last = lab
    for s in starts:
        ax.axhline(s - 0.5, color="white", linewidth=0.45, alpha=0.7)
        ax.axvline(s - 0.5, color="white", linewidth=0.45, alpha=0.7)
    if len(tick_labels) <= 35:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_tsne_umap(feats, labels, out_path, max_ids=60, samples_per_id=2, seed=2026):
    set_style()
    idx = sample_balanced_indices(labels, max_ids=max_ids, samples_per_id=samples_per_id, seed=seed)
    sub_feats = feats[idx]
    sub_labels = labels[idx]
    lab_names, lab_ids = np.unique(sub_labels, return_inverse=True)

    scaled = StandardScaler().fit_transform(sub_feats)
    n_pca = min(50, scaled.shape[0] - 1, scaled.shape[1])
    if n_pca >= 2:
        scaled = PCA(n_components=n_pca, random_state=seed).fit_transform(scaled)

    perplexity = max(5, min(30, (scaled.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(scaled)

    umap_or_pca = None
    method_name = "UMAP"
    try:
        import umap
        umap_or_pca = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15, random_state=seed).fit_transform(scaled)
    except Exception:
        method_name = "PCA fallback"
        umap_or_pca = PCA(n_components=2, random_state=seed).fit_transform(scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))
    cmap = plt.get_cmap("tab20")
    for ax, emb, title in zip(axes, [tsne, umap_or_pca], ["t-SNE", method_name]):
        for lab_idx, lab in enumerate(lab_names):
            mask = lab_ids == lab_idx
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=22,
                alpha=0.84,
                color=cmap(lab_idx % 20),
                linewidths=0,
            )
            center = emb[mask].mean(axis=0)
            ax.text(center[0], center[1], str(lab), fontsize=6.5, color="#111111", ha="center", va="center")
        ax.set_title(f"CDGaitFusion {title} Feature Space")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CDGaitFusion discriminability figures from saved GREW embeddings.")
    parser.add_argument("--npz", default="/root/sq/output/GREW/figures/grew_identity_embeddings_180000.npz")
    parser.add_argument("--out_dir", default="/root/sq/visual/picture")
    parser.add_argument("--max_pairs", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include_non_identity_labels", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feats, labels, _, _ = load_embeddings(args.npz, keep_numeric_labels=not args.include_non_identity_labels)
    pair_a, pair_b, pair_y = sample_pairs(labels, max_pairs=args.max_pairs, seed=args.seed)
    distances, similarities = cosine_distance_pairs(feats, pair_a, pair_b)
    intra = distances[pair_y == 1]
    inter = distances[pair_y == 0]

    plot_distance_distribution(intra, inter, os.path.join(args.out_dir, "01_intra_inter_distance.png"))
    roc_auc, eer, eer_thr = plot_roc(similarities, pair_y, os.path.join(args.out_dir, "02_roc_auc_eer.png"))
    plot_distance_matrix(feats, labels, os.path.join(args.out_dir, "03_distance_matrix.png"), seed=args.seed)
    plot_tsne_umap(feats, labels, os.path.join(args.out_dir, "04_tsne_umap.png"), seed=args.seed)

    ratio = float(inter.mean() / max(intra.mean(), 1e-12))
    summary = [
        "CDGaitFusion discriminability summary",
        f"Embedding file: {args.npz}",
        f"Samples: {feats.shape[0]}",
        f"Feature dim: {feats.shape[1]}",
        f"Identities: {len(np.unique(labels))}",
        f"Pair samples: {len(pair_y)}",
        f"Mean intra-class cosine distance: {intra.mean():.6f}",
        f"Mean inter-class cosine distance: {inter.mean():.6f}",
        f"Inter/Intra distance ratio: {ratio:.6f}",
        f"ROC AUC: {roc_auc:.6f}",
        f"EER: {eer:.6f}",
        f"EER threshold(similarity): {eer_thr:.6f}",
    ]
    summary_path = os.path.join(args.out_dir, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
