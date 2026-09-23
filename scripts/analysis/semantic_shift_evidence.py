# semantic_shift_evidence.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

PLOT_GROUPS = {
    "zirconia": "zirconia",
    "everx": "short_fiber_composite",
    "omnichroma": "structural_color_composite",
}

# -----------------------------
# Loading helpers
# -----------------------------

def load_wv(path):
    """
    Loads either a full Gensim Word2Vec model or KeyedVectors.
    """
    path = str(path)
    try:
        model = Word2Vec.load(path)
        return model.wv
    except Exception:
        return KeyedVectors.load(path)


def read_terms_table(path):
    """
    Accepts:
      1) txt file:
           term
           term
           ...
         or:
           term<TAB>group

      2) csv/tsv file with at least a term column.
         Optional group column names:
           group, material_system, category
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t")
    else:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "\t" in line:
                    parts = line.split("\t")
                elif "," in line:
                    parts = line.split(",")
                else:
                    parts = [line]

                term = parts[0].strip()
                group = parts[1].strip() if len(parts) > 1 else "target_terms"

                rows.append({
                    "term": term,
                    "material_system": group,
                })

        df = pd.DataFrame(rows)

    if "term" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "term"})

    group_col = None
    for c in ["material_system", "group", "category"]:
        if c in df.columns:
            group_col = c
            break

    if group_col is None:
        df["material_system"] = "target_terms"
    elif group_col != "material_system":
        df = df.rename(columns={group_col: "material_system"})

    df["term"] = df["term"].astype(str).str.strip()
    df["material_system"] = df["material_system"].astype(str).str.strip()

    df = df[df["term"] != ""].copy()
    df = df.drop_duplicates(subset=["term", "material_system"])

    return df


def read_visualization_sets(path, terms_df):
    """Read ordered visualization terms and verify canonical targets are included."""
    df = pd.read_csv(path, sep="\t")
    required = {"term", "material_system"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns: term, material_system"
        )

    df["term"] = df["term"].astype(str).str.strip()
    df["material_system"] = df["material_system"].astype(str).str.strip()

    plot_sets = {}
    for plot_name, material_system in PLOT_GROUPS.items():
        selected_terms = (
            df.loc[df["material_system"] == material_system, "term"]
            .drop_duplicates()
            .tolist()
        )
        canonical_terms = set(
            terms_df.loc[
                terms_df["material_system"] == material_system,
                "term",
            ]
        )
        missing = canonical_terms - set(selected_terms)
        if missing:
            raise ValueError(
                f"{path}: visualization set {material_system} is missing canonical "
                f"target term(s): {', '.join(sorted(missing))}"
            )
        plot_sets[plot_name] = selected_terms

    return plot_sets


def get_count(wv, word):
    try:
        return int(wv.get_vecattr(word, "count"))
    except Exception:
        return 0


# -----------------------------
# Nearest-neighbor utilities
# -----------------------------

def build_candidate_matrix(wv, candidate_terms):
    """
    Builds normalized matrix for restricted nearest-neighbor search.
    """
    terms = [w for w in candidate_terms if w in wv]

    matrix = np.vstack([
        wv[w] for w in terms
    ]).astype(np.float32)

    matrix = normalize(matrix)

    return terms, matrix


def nearest_neighbors_from_matrix(
    wv,
    query,
    candidate_terms,
    candidate_matrix,
    topn=20,
):
    """
    Finds nearest neighbors of query among candidate_terms.
    Uses cosine similarity.
    """
    if query not in wv:
        return []

    q = wv[query].reshape(1, -1).astype(np.float32)
    q = normalize(q)

    sims = cosine_similarity(q, candidate_matrix)[0]

    # Exclude query itself if present among candidates
    try:
        self_idx = candidate_terms.index(query)
        sims[self_idx] = -np.inf
    except ValueError:
        pass

    top_idx = np.argsort(sims)[::-1][:topn]

    return [
        {
            "neighbor": candidate_terms[i],
            "similarity": float(sims[i]),
        }
        for i in top_idx
        if np.isfinite(sims[i])
    ]


def neighbors_to_string(neighbors):
    return "; ".join([
        f"{x['neighbor']} ({x['similarity']:.3f})"
        for x in neighbors
    ])


def compute_jaccard_and_neighbors(
    pre_wv,
    post_wv,
    terms_df,
    out_dir,
    k_values=(5, 10, 20),
    nn_topn=20,
    mode="common_vocab",
):
    """
    mode:
      common_vocab: nearest neighbors searched only among vocab_pre n vocab_post
      full_vocab: nearest neighbors searched in each model's own full vocabulary
    """
    target_terms = sorted(set(terms_df["term"]))

    if mode == "common_vocab":
        candidate_pre = sorted(set(pre_wv.index_to_key) & set(post_wv.index_to_key))
        candidate_post = candidate_pre
    elif mode == "full_vocab":
        candidate_pre = pre_wv.index_to_key
        candidate_post = post_wv.index_to_key
    else:
        raise ValueError("mode must be 'common_vocab' or 'full_vocab'")

    print(f"\nBuilding candidate matrices for mode={mode}...")
    print(f"Pre candidates: {len(candidate_pre)}")
    print(f"Post candidates: {len(candidate_post)}")

    pre_candidate_terms, pre_matrix = build_candidate_matrix(pre_wv, candidate_pre)
    post_candidate_terms, post_matrix = build_candidate_matrix(post_wv, candidate_post)

    nn_rows = []
    jaccard_rows = []

    max_k = max(max(k_values), nn_topn)

    for term in target_terms:
        in_pre = term in pre_wv
        in_post = term in post_wv

        group_values = (
            terms_df.loc[terms_df["term"] == term, "material_system"]
            .drop_duplicates()
            .tolist()
        )
        group_str = ";".join(group_values)

        if not in_pre or not in_post:
            nn_rows.append({
                "term": term,
                "material_system": group_str,
                "status": "missing",
                "in_pre": in_pre,
                "in_post": in_post,
                "pre_count": get_count(pre_wv, term) if in_pre else 0,
                "post_count": get_count(post_wv, term) if in_post else 0,
                "mode": mode,
                "pre_neighbors": "",
                "post_neighbors": "",
            })
            continue

        pre_nn = nearest_neighbors_from_matrix(
            pre_wv,
            term,
            pre_candidate_terms,
            pre_matrix,
            topn=max_k,
        )

        post_nn = nearest_neighbors_from_matrix(
            post_wv,
            term,
            post_candidate_terms,
            post_matrix,
            topn=max_k,
        )

        nn_rows.append({
            "term": term,
            "material_system": group_str,
            "status": "compared",
            "in_pre": True,
            "in_post": True,
            "pre_count": get_count(pre_wv, term),
            "post_count": get_count(post_wv, term),
            "mode": mode,
            "pre_neighbors": neighbors_to_string(pre_nn[:nn_topn]),
            "post_neighbors": neighbors_to_string(post_nn[:nn_topn]),
        })

        for k in k_values:
            pre_set = set([x["neighbor"] for x in pre_nn[:k]])
            post_set = set([x["neighbor"] for x in post_nn[:k]])

            intersection = pre_set & post_set
            union = pre_set | post_set

            jaccard = len(intersection) / len(union) if union else np.nan
            overlap_count = len(intersection)

            jaccard_rows.append({
                "term": term,
                "material_system": group_str,
                "mode": mode,
                "k": k,
                "status": "compared",
                "pre_count": get_count(pre_wv, term),
                "post_count": get_count(post_wv, term),
                "overlap_count": overlap_count,
                "jaccard": jaccard,
                "pre_neighbors": "; ".join([x["neighbor"] for x in pre_nn[:k]]),
                "post_neighbors": "; ".join([x["neighbor"] for x in post_nn[:k]]),
                "shared_neighbors": "; ".join(sorted(intersection)),
            })

    nn_df = pd.DataFrame(nn_rows)
    jaccard_df = pd.DataFrame(jaccard_rows)

    nn_path = out_dir / f"nearest_neighbor_table_{mode}.csv"
    jaccard_path = out_dir / f"jaccard_overlap_{mode}.csv"

    nn_df.to_csv(nn_path, index=False)
    jaccard_df.to_csv(jaccard_path, index=False)

    # Summary
    if len(jaccard_df) > 0:
        summary = (
            jaccard_df
            .groupby(["mode", "k"])
            .agg(
                n_terms=("term", "count"),
                mean_jaccard=("jaccard", "mean"),
                median_jaccard=("jaccard", "median"),
                mean_overlap=("overlap_count", "mean"),
                median_overlap=("overlap_count", "median"),
            )
            .reset_index()
        )

        group_summary = (
            jaccard_df
            .groupby(["mode", "material_system", "k"])
            .agg(
                n_terms=("term", "count"),
                mean_jaccard=("jaccard", "mean"),
                median_jaccard=("jaccard", "median"),
                mean_overlap=("overlap_count", "mean"),
                median_overlap=("overlap_count", "median"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
        group_summary = pd.DataFrame()

    summary.to_csv(out_dir / f"jaccard_summary_{mode}.csv", index=False)
    group_summary.to_csv(out_dir / f"jaccard_group_summary_{mode}.csv", index=False)

    print(f"Saved: {nn_path}")
    print(f"Saved: {jaccard_path}")

    return nn_df, jaccard_df, summary, group_summary


# -----------------------------
# Vocabulary emergence
# -----------------------------

def compute_vocabulary_emergence(pre_wv, post_wv, terms_df, out_dir):
    rows = []

    for _, row in terms_df.iterrows():
        term = row["term"]
        group = row["material_system"]

        in_pre = term in pre_wv
        in_post = term in post_wv

        if (not in_pre) and in_post:
            status = "post_only"
        elif in_pre and (not in_post):
            status = "pre_only"
        elif in_pre and in_post:
            status = "present_in_both"
        else:
            status = "absent_in_both"

        rows.append({
            "term": term,
            "material_system": group,
            "status": status,
            "in_pre": in_pre,
            "in_post": in_post,
            "pre_count": get_count(pre_wv, term) if in_pre else 0,
            "post_count": get_count(post_wv, term) if in_post else 0,
        })

    emergence_df = pd.DataFrame(rows)

    emergence_df.to_csv(
        out_dir / "target_term_vocabulary_coverage_and_emergence.csv",
        index=False,
    )

    post_only = emergence_df[emergence_df["status"] == "post_only"].copy()
    post_only.to_csv(
        out_dir / "target_terms_post_only_vocabulary_emergence.csv",
        index=False,
    )

    summary = (
        emergence_df
        .groupby(["material_system", "status"])
        .size()
        .reset_index(name="n_terms")
    )

    summary.to_csv(
        out_dir / "target_term_vocabulary_emergence_summary.csv",
        index=False,
    )

    print("Saved vocabulary emergence tables.")

    return emergence_df, post_only, summary


# -----------------------------
# Procrustes-aligned vectors
# -----------------------------

def load_alignment_npz(path):
    data = np.load(path)
    R = data["R"]
    mu_pre = data["mu_pre"]
    mu_post = data["mu_post"]
    return R, mu_pre, mu_post


def aligned_pre_vector(pre_wv, word, R, mu_pre, mu_post):
    v = pre_wv[word].reshape(1, -1).astype(np.float64)
    v = normalize(v)
    return ((v - mu_pre) @ R + mu_post)[0]


def post_vector(post_wv, word):
    v = post_wv[word].reshape(1, -1).astype(np.float64)
    return normalize(v)[0]

def build_aligned_plot_vectors(pre_wv, post_wv, terms_df, alignment_npz, selected_terms=None):
    R, mu_pre, mu_post = load_alignment_npz(alignment_npz)

    rows = []
    vectors = []

    if selected_terms is not None:
        selected_terms = list(dict.fromkeys(selected_terms))  # preserve order, remove duplicates

        tmp_rows = []
        for term in selected_terms:
            matched = terms_df[terms_df["term"] == term]

            if len(matched) > 0:
                groups = matched["material_system"].drop_duplicates().tolist()
                group = ";".join(groups)
            else:
                group = "custom_plot_set"

            tmp_rows.append({
                "term": term,
                "material_system": group,
            })

        plot_terms_df = pd.DataFrame(tmp_rows)

    else:
        plot_terms_df = terms_df.drop_duplicates(subset=["term", "material_system"]).copy()

    for _, row in plot_terms_df.iterrows():
        term = row["term"]
        group = row["material_system"]

        in_pre = term in pre_wv
        in_post = term in post_wv

        if in_pre and in_post:
            v_pre = aligned_pre_vector(pre_wv, term, R, mu_pre, mu_post)
            v_post = post_vector(post_wv, term)

            rows.append({
                "term": term,
                "material_system": group,
                "period": "pre_2018_aligned",
                "paired": True,
                "pre_count": get_count(pre_wv, term),
                "post_count": get_count(post_wv, term),
            })
            vectors.append(v_pre)

            rows.append({
                "term": term,
                "material_system": group,
                "period": "post_2018",
                "paired": True,
                "pre_count": get_count(pre_wv, term),
                "post_count": get_count(post_wv, term),
            })
            vectors.append(v_post)

        elif (not in_pre) and in_post:
            # post-only emerging term
            v_post = post_vector(post_wv, term)

            rows.append({
                "term": term,
                "material_system": group,
                "period": "post_2018_only",
                "paired": False,
                "pre_count": 0,
                "post_count": get_count(post_wv, term),
            })
            vectors.append(v_post)

        elif in_pre and (not in_post):
            # pre-only term: usually not expected here, but keep metadata
            v_pre = aligned_pre_vector(pre_wv, term, R, mu_pre, mu_post)

            rows.append({
                "term": term,
                "material_system": group,
                "period": "pre_2018_only_aligned",
                "paired": False,
                "pre_count": get_count(pre_wv, term),
                "post_count": 0,
            })
            vectors.append(v_pre)

    meta_df = pd.DataFrame(rows)

    if len(vectors) == 0:
        return meta_df, None

    X = np.vstack(vectors).astype(np.float64)

    return meta_df, X

# -----------------------------
# Dimensionality reduction
# -----------------------------
def run_dimensionality_reduction(meta_df, X, out_dir, prefix="all"):
    methods = {}

    if X is None or X.shape[0] < 2:
        print(f"Skipping dimensionality reduction for {prefix}: not enough points.")
        return methods

    # PCA
    n_components = 2 if X.shape[0] >= 2 else 1
    pca = PCA(n_components=n_components, random_state=42)
    coords_pca = pca.fit_transform(X)

    if coords_pca.shape[1] == 1:
        coords_pca = np.column_stack([coords_pca[:, 0], np.zeros(coords_pca.shape[0])])

    methods["pca"] = {
        "coords": coords_pca,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    # t-SNE: needs enough points
    n = X.shape[0]
    if n >= 6:
        perplexity = min(30, max(2, (n - 1) // 3))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )

        coords_tsne = tsne.fit_transform(X)

        methods["tsne"] = {
            "coords": coords_tsne,
            "perplexity": perplexity,
        }
    else:
        print(f"Skipping t-SNE for {prefix}: only {n} points.")

    # UMAP
    try:
        import umap

        if n >= 4:
            n_neighbors = min(15, max(2, n - 1))

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )

            coords_umap = reducer.fit_transform(X)

            methods["umap"] = {
                "coords": coords_umap,
                "n_neighbors": n_neighbors,
            }
        else:
            print(f"Skipping UMAP for {prefix}: only {n} points.")

    except Exception as e:
        print(f"UMAP skipped for {prefix}: {e}")

    # Save coordinates
    for method, obj in methods.items():
        coords = obj["coords"]
        df_coords = meta_df.copy()
        df_coords[f"{method}_1"] = coords[:, 0]
        df_coords[f"{method}_2"] = coords[:, 1]

        df_coords.to_csv(
            out_dir / f"{prefix}_{method}_coordinates.csv",
            index=False,
        )

    # Save metadata
    reduction_meta = {
        method: {
            k: v for k, v in obj.items()
            if k != "coords"
        }
        for method, obj in methods.items()
    }

    with open(out_dir / f"{prefix}_dimensionality_reduction_metadata.json", "w", encoding="utf-8") as f:
        json.dump(reduction_meta, f, indent=2)

    return methods

# -----------------------------
# Plotting
# -----------------------------
def plot_reduction(meta_df, coords, method, out_path, title_prefix="all" ):
    df = meta_df.copy()
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    plt.figure(figsize=(12, 9))

    pre_period = "pre_2018_aligned"
    post_period =  "post_2018"
    period_markers = {
        pre_period: "o",
        post_period: "s",
        "post_2018_only": "X",
        "pre_2018_only_aligned": "D",
    }

    period_labels = {
        pre_period: "90% post-2018 model aligned to full model",
        post_period: "Full post-2018 model",
        "post_2018_only": "Post-2018 model only",
        "pre_2018_only_aligned": "Pre-2018 model only",
    }


    # Draw paired lines between pre and post
    paired_terms = df[df["paired"] == True]["term"].unique()

    for term in paired_terms:
        sub = df[df["term"] == term]

        if {pre_period, post_period}.issubset(set(sub["period"])):
            p0 = sub[sub["period"] == pre_period].iloc[0]
            p1 = sub[sub["period"] == post_period].iloc[0]

            plt.plot(
                [p0["x"], p1["x"]],
                [p0["y"], p1["y"]],
                linewidth=0.8,
                alpha=0.4,
            )

    # Draw points
    for period, marker in period_markers.items():
        sub = df[df["period"] == period]

        if len(sub) == 0:
            continue

        plt.scatter(
            sub["x"],
            sub["y"],
            marker=marker,
            s=50,
            alpha=0.85,
            label=period_labels.get(period, period)
        )

    # Label post and post-only points
    label_df = df[df["period"].isin([post_period, "post_2018_only"])].copy()

    for _, row in label_df.iterrows():
        plt.text(
            row["x"],
            row["y"],
            row["term"],
            fontsize=8,
            alpha=0.85,
        )

    if title_prefix:
        title = f"{title_prefix}: {method.upper()} projection"
    else:
        title = f"{method.upper()} projection"

    plt.title(title)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_all_plots(meta_df, methods, out_dir, prefix="all" ):
    for method, obj in methods.items():
        out_path = out_dir / f"{prefix}_{method}_projection.png"

        plot_reduction(
            meta_df=meta_df,
            coords=obj["coords"],
            method=method,
            out_path=out_path,
            title_prefix=prefix
        )

        print(f"Saved: {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pre-model", required=True)
    parser.add_argument("--post-model", required=True)
    parser.add_argument("--target-terms", required=True)
    parser.add_argument("--visualization-terms", default="visualization_terms.tsv")
    parser.add_argument("--alignment-npz", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--nn-topn", type=int, default=20)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    pre_wv = load_wv(args.pre_model)
    post_wv = load_wv(args.post_model)

    print("Reading target terms...")
    terms_df = read_terms_table(args.target_terms)

    terms_df.to_csv(out_dir / "target_terms_used.csv", index=False)
    plot_sets = read_visualization_sets(args.visualization_terms, terms_df)

    # 1. Vocabulary emergence
    print("\nComputing vocabulary emergence...")
    compute_vocabulary_emergence(
        pre_wv=pre_wv,
        post_wv=post_wv,
        terms_df=terms_df,
        out_dir=out_dir,
    )

    # 2. Jaccard + nearest-neighbor tables
    # Main quantitative version: common vocabulary
    print("\nComputing Jaccard and nearest-neighbor tables, common vocabulary...")
    compute_jaccard_and_neighbors(
        pre_wv=pre_wv,
        post_wv=post_wv,
        terms_df=terms_df,
        out_dir=out_dir,
        k_values=args.k,
        nn_topn=args.nn_topn,
        mode="common_vocab",
    )

    # Descriptive/sensitivity version: full vocabulary
    print("\nComputing Jaccard and nearest-neighbor tables, full vocabulary...")
    compute_jaccard_and_neighbors(
        pre_wv=pre_wv,
        post_wv=post_wv,
        terms_df=terms_df,
        out_dir=out_dir,
        k_values=args.k,
        nn_topn=args.nn_topn,
        mode="full_vocab",
    )

    # 3. PCA / UMAP / t-SNE on Procrustes-aligned vectors
    print("\nRunning separate PCA / t-SNE / UMAP plots for zirconia, everx, and omnichroma...")
    
    for plot_name, selected_terms in plot_sets.items():
        print(f"\nPlot set: {plot_name}")
    
        meta_df, X = build_aligned_plot_vectors(
            pre_wv=pre_wv,
            post_wv=post_wv,
            terms_df=terms_df,
            alignment_npz=args.alignment_npz,
            selected_terms=selected_terms,
        )
    
        if X is None or len(meta_df) == 0:
            print(f"Skipping {plot_name}: no available vectors.")
            continue
    
        meta_df.to_csv(
            out_dir / f"{plot_name}_aligned_projection_points_metadata.csv",
            index=False,
        )
    
        methods = run_dimensionality_reduction(
            meta_df=meta_df,
            X=X,
            out_dir=out_dir,
            prefix=plot_name,
        )
    
        save_all_plots(
            meta_df=meta_df,
            methods=methods,
            out_dir=out_dir,
            prefix=plot_name
        )
    print("\nDone.")


if __name__ == "__main__":
    main()