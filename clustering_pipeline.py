"""
Job 1 (classify from data) + Job 2 (compare to genetic types).

Encodes the measured-geology feature set into a multi-hot binary matrix,
reduces dimensionality with UMAP (jaccard metric, fit on binary data),
clusters with HDBSCAN (density-based -> allows outliers, matching the
guidance note's expectation that some reports won't fit cleanly anywhere).

Saves all artifacts needed by the Streamlit app so the app never has to
refit anything at runtime.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
import umap
import hdbscan

from data_prep import build_report_table, CLUSTER_FEATURE_COLS, LABEL_COLS, CONTEXT_COLS, load_dictionary

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

MIN_TAG_FREQ = 3  # drop tag-columns that appear in fewer than N reports (pure noise / typos)


def encode_features(df, feature_cols):
    """Multi-hot encode each list-valued column, prefixing columns so
    the same code in two different fields doesn't collide (e.g. a code
    could theoretically repeat across host_unit vs alteration_type)."""
    blocks = []
    mlbs = {}
    for col in feature_cols:
        mlb = MultiLabelBinarizer()
        mat = mlb.fit_transform(df[col])
        cols = [f"{col}::{c}" for c in mlb.classes_]
        block = pd.DataFrame(mat, columns=cols, index=df.index)
        # drop ultra-rare tags (likely noise, and they add dimensionality for no signal)
        keep = block.columns[block.sum(axis=0) >= MIN_TAG_FREQ]
        blocks.append(block[keep])
        mlbs[col] = mlb
    X = pd.concat(blocks, axis=1)
    return X, mlbs


def run_pipeline():
    df = build_report_table()
    X, mlbs = encode_features(df, CLUSTER_FEATURE_COLS)
    print("Feature matrix:", X.shape)

    # --- dimensionality reduction (jaccard suits sparse binary tag data) ---
    reducer = umap.UMAP(
        n_neighbors=20, min_dist=0.05, n_components=10,
        metric="jaccard", random_state=42,
    )
    embedding = reducer.fit_transform(X.values)

    # a 2D embedding purely for visualization in the app
    reducer_2d = umap.UMAP(
        n_neighbors=20, min_dist=0.1, n_components=2,
        metric="jaccard", random_state=42,
    )
    embedding_2d = reducer_2d.fit_transform(X.values)

    # --- clustering: HDBSCAN, density based, allows outliers (-1) ---
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=40, min_samples=10,
        prediction_data=True, cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(embedding)
    print("Clusters found:", len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
          "| outliers:", (cluster_labels == -1).sum())

    df["cluster"] = cluster_labels
    df["emb_x"] = embedding_2d[:, 0]
    df["emb_y"] = embedding_2d[:, 1]

    # --- cluster signature: lift score per tag ---
    overall_freq = X.mean(axis=0)
    signatures = {}
    for c in sorted(set(cluster_labels)):
        mask = (cluster_labels == c)
        if mask.sum() == 0:
            continue
        cluster_freq = X[mask].mean(axis=0)
        lift = (cluster_freq / overall_freq.replace(0, np.nan)).fillna(0)
        # only consider tags that are reasonably present in this cluster
        present = cluster_freq[cluster_freq > 0.15]
        ranked = lift[present.index].sort_values(ascending=False)
        signatures[c] = ranked.head(12)

    # --- Job 2: compare clusters to conventional deposit_type ---
    df["deposit_type_primary"] = df["deposit_type"].apply(lambda l: l[0] if l else None)
    has_label = df["deposit_type_primary"].notna()
    ari = adjusted_rand_score(df.loc[has_label, "deposit_type_primary"], df.loc[has_label, "cluster"])
    hom, comp, vmeas = homogeneity_completeness_v_measure(
        df.loc[has_label, "deposit_type_primary"], df.loc[has_label, "cluster"]
    )
    print(f"ARI={ari:.3f}  homogeneity={hom:.3f}  completeness={comp:.3f}  v={vmeas:.3f}")

    crosstab = pd.crosstab(df["cluster"], df["deposit_type_primary"])

    # majority label per cluster (for QA mismatch flagging)
    majority_label = crosstab.idxmax(axis=1)
    df["cluster_majority_label"] = df["cluster"].map(majority_label)
    df["label_mismatch"] = (
        df["deposit_type_primary"].notna()
        & (df["deposit_type_primary"] != df["cluster_majority_label"])
    )
    df["is_outlier"] = df["cluster"] == -1

    # --- save everything the app needs (pickle, since several columns are list-valued) ---
    df.to_pickle(ARTIFACT_DIR / "report_clusters.pkl")
    with open(ARTIFACT_DIR / "reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)
    with open(ARTIFACT_DIR / "clusterer.pkl", "wb") as f:
        pickle.dump(clusterer, f)
    with open(ARTIFACT_DIR / "mlbs.pkl", "wb") as f:
        pickle.dump(mlbs, f)
    with open(ARTIFACT_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    with open(ARTIFACT_DIR / "signatures.pkl", "wb") as f:
        pickle.dump(signatures, f)
    crosstab.to_pickle(ARTIFACT_DIR / "crosstab.pkl")
    with open(ARTIFACT_DIR / "metrics.pkl", "wb") as f:
        pickle.dump({"ari": ari, "homogeneity": hom, "completeness": comp, "v_measure": vmeas,
                     "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                     "n_outliers": int((cluster_labels == -1).sum()), "n_reports": len(df)}, f)
    X.astype(np.uint8).to_pickle(ARTIFACT_DIR / "feature_matrix.pkl")

    print("Artifacts saved to", ARTIFACT_DIR)
    return df


if __name__ == "__main__":
    run_pipeline()
