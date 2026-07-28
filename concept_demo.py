"""
Standalone, non-Streamlit walkthrough of the concept app.py is built on:

  1. Cluster the Global Deposit Database (10,089 NI 43-101 reports) using
     only measured geological attributes -- deposit type, commodity and
     geography are held OUT of clustering so the data forms its own
     groups instead of repeating each report author's label.
  2. Compare those data-driven groups back to the conventional genetic
     types afterward. Agreement builds confidence; disagreement is a
     finding worth investigating, not an error.
  3. Given a new exploration project's observed attributes, find its
     closest analogues in the library by raw Jaccard similarity, and
     read off which data-driven group (and which conventional deposit
     types) that project most resembles.

Run: python concept_demo.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from data_prep import CLUSTER_FEATURE_COLS
from decode import decode_code, decode_feature_column, pretty_field_name

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def load_artifacts():
    df = pd.read_pickle(ARTIFACT_DIR / "report_clusters.pkl")
    with open(ARTIFACT_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(ARTIFACT_DIR / "signatures.pkl", "rb") as f:
        signatures = pickle.load(f)
    crosstab = pd.read_pickle(ARTIFACT_DIR / "crosstab.pkl")
    with open(ARTIFACT_DIR / "metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    X = pd.read_pickle(ARTIFACT_DIR / "feature_matrix.pkl")
    return df, feature_columns, signatures, crosstab, metrics, X


def cluster_label(df, crosstab, c):
    if c == -1:
        return "Outliers / no clear group"
    top = crosstab.loc[c].sort_values(ascending=False)
    top_name = decode_code("deposit_type", top.index[0]) if top.index[0] else "Unlabeled"
    return f'Cluster {c} — closest to "{top_name}" (n={int(top.sum())})'


def print_framework_overview(metrics):
    print("=" * 78)
    print("FRAMEWORK: two separate databases, kept apart on purpose")
    print("=" * 78)
    print(
        "  Global Deposit Database (the LIBRARY)      -> what deposits look like\n"
        "  Exploration Project Database (the SEARCH)   -> what a real project holds\n"
    )
    print(f"  Reports in library:              {metrics['n_reports']}")
    print(f"  Data-driven groups:              {metrics['n_clusters']}")
    print(f"  Outliers (no clear group):       {metrics['n_outliers']}")
    print(f"  Agreement w/ genetic types (ARI): {metrics['ari']:.2f}")
    print()


def print_cluster_signatures(df, crosstab, signatures):
    print("=" * 78)
    print("DATA-DRIVEN GROUPS vs. CONVENTIONAL DEPOSIT TYPES")
    print("=" * 78)
    cluster_ids = sorted(c for c in df["cluster"].unique() if c != -1)
    for c in cluster_ids:
        print(f"\n{cluster_label(df, crosstab, c)}")
        if c not in signatures:
            print("  (no signature -- outlier group)")
            continue
        sig = signatures[c]
        for feat_col, lift in sig.items():
            field, label = decode_feature_column(feat_col)
            print(f"  - {pretty_field_name(field)}: {label} (lift={lift:.2f})")
    print()


def find_analogues(user_selections, feature_columns, X, df, crosstab, signatures, k=15):
    """Same Jaccard-similarity analogue search as app.py's Project Analogue
    Search page: build a boolean vector for the user's observed attributes,
    rank library reports by Jaccard similarity, and majority-vote the K
    nearest neighbours' clusters to pick a best-matching group."""
    chosen_cols = [c for cols in user_selections.values() for c in cols]
    if not chosen_cols:
        print("No attributes selected -- nothing to search.")
        return

    vec = pd.Series(0, index=feature_columns, dtype=np.uint8)
    vec[chosen_cols] = 1

    Xv = X.values.astype(bool)
    uv = vec.values.astype(bool)
    intersection = (Xv & uv).sum(axis=1)
    union = (Xv | uv).sum(axis=1)
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

    top_idx = np.argsort(-jaccard)[:k]
    neighbour_clusters = df.iloc[top_idx]["cluster"].values
    neighbour_sims = jaccard[top_idx]

    print("=" * 78)
    print("PROJECT ANALOGUE SEARCH (example)")
    print("=" * 78)
    print("Observed attributes:")
    for field, cols in user_selections.items():
        if cols:
            labels = [decode_feature_column(c)[1] for c in cols]
            print(f"  - {pretty_field_name(field)}: {', '.join(labels)}")

    if neighbour_sims.max() <= 0:
        print("\nNo overlap with any library report -- select more attributes.")
        return

    vals, counts = np.unique(neighbour_clusters, return_counts=True)
    pred_cluster = int(vals[np.argmax(counts)])
    agreement = counts.max() / k

    print(f"\nNearest {k} analogue reports -> cluster votes: "
          + ", ".join(f"{v}:{c}" for v, c in zip(vals, counts)))

    if pred_cluster == -1 or agreement < 0.34:
        print(
            "\nNo clear group match -- nearest reports are spread across several "
            "groups. Per the guidance note, that's a valid result (possible "
            "hybrid system or distinct signature), not an error."
        )
    else:
        print(f"\nBest match: {cluster_label(df, crosstab, pred_cluster)}")
        print(f"  ({int(agreement * k)} of {k} closest analogues fall in this group)")

    print("\nClosest individual analogue reports:")
    nearest = df.iloc[top_idx][["pdf_hash", "deposit_type_primary", "cluster"]].copy()
    nearest["similarity"] = neighbour_sims.round(3)
    nearest["deposit_type_primary"] = nearest["deposit_type_primary"].apply(
        lambda c: decode_code("deposit_type", c) if c else "—"
    )
    print(nearest.to_string(index=False))


def main():
    df, feature_columns, signatures, crosstab, metrics, X = load_artifacts()

    print_framework_overview(metrics)
    print_cluster_signatures(df, crosstab, signatures)

    # Example: a hypothetical junior project's observed attributes -- pick
    # the first available option for a couple of fields just to demonstrate
    # the search mechanics end-to-end.
    example_selections = {}
    for field in CLUSTER_FEATURE_COLS[:3]:
        options = [c for c in feature_columns if c.startswith(f"{field}::")]
        example_selections[field] = options[:1]

    find_analogues(example_selections, feature_columns, X, df, crosstab, signatures)


if __name__ == "__main__":
    main()
