import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from pathlib import Path

from data_prep import CLUSTER_FEATURE_COLS, load_dictionary
from decode import decode_code, decode_feature_column, pretty_field_name

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Mineral Prospectivity Modelling", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    df = pd.read_pickle(ARTIFACT_DIR / "report_clusters.pkl")
    with open(ARTIFACT_DIR / "mlbs.pkl", "rb") as f:
        mlbs = pickle.load(f)
    with open(ARTIFACT_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(ARTIFACT_DIR / "signatures.pkl", "rb") as f:
        signatures = pickle.load(f)
    crosstab = pd.read_pickle(ARTIFACT_DIR / "crosstab.pkl")
    with open(ARTIFACT_DIR / "metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    X = pd.read_pickle(ARTIFACT_DIR / "feature_matrix.pkl")
    return df, mlbs, feature_columns, signatures, crosstab, metrics, X


df, mlbs, feature_columns, signatures, crosstab, metrics, X = load_artifacts()

CLUSTER_IDS = sorted([c for c in df["cluster"].unique() if c != -1])


def cluster_label(c):
    if c == -1:
        return "Outliers / no clear group"
    top = crosstab.loc[c].sort_values(ascending=False)
    top_name = decode_code("deposit_type", top.index[0]) if top.index[0] else "Unlabeled"
    return f"Cluster {c} — closest to \u201c{top_name}\u201d (n={int(top.sum())})"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Mineral Prospectivity Modelling")
page = st.sidebar.radio(
    "Section",
    ["Framework Overview", "Library Explorer (Global Deposit DB)",
     "Project Analogue Search", "Resource-Size Insights", "QA Review List"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Library built from 10,089 NI 43-101 reports \u00b7 "
    f"{metrics['n_clusters']} data-driven groups \u00b7 "
    f"{metrics['n_outliers']} outliers"
)


# ---------------------------------------------------------------------------
# PAGE 1 — Framework overview
# ---------------------------------------------------------------------------
if page == "Framework Overview":
    st.title("Mineral Prospectivity Modelling — Framework")
    st.markdown(
        "This tool is built around a distinction set out in the internal guidance note: "
        "**two separate databases**, kept apart on purpose."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Deposit Database")
        st.markdown(
            "- The **library**\n"
            "- Built from 10,089 NI 43-101 technical reports\n"
            "- Tells us what deposits look like\n"
            "- This is what you're browsing in *Library Explorer*"
        )
    with col2:
        st.subheader("Exploration Project Database")
        st.markdown(
            "- The **search data**\n"
            "- Mapping, structure, geochem, geophysics, sparse drilling for *your* licence area\n"
            "- This is what a real project actually holds\n"
            "- Entered in *Project Analogue Search*"
        )

    st.markdown("---")
    st.subheader("What the library is used for")
    st.markdown(
        "**1. Classify** deposits by their measured attributes only — deposit type, "
        "commodity and geography are hidden during grouping, so the data forms its own "
        "groups instead of repeating each report author's label.\n\n"
        "**2. Compare** those groups back to the standard genetic types (epithermal, "
        "orogenic, porphyry, etc.). Agreement builds confidence. Disagreement is treated "
        "as a finding worth investigating — not an error.\n\n"
        "**3. Link to resource size** — within a coherent group, test whether any "
        "parameter tracks eventual deposit scale. *(Not yet possible — see the "
        "Resource-Size Insights page for the data gap.)*"
    )

    st.markdown("---")
    st.subheader("Current library status")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reports in library", metrics["n_reports"])
    m2.metric("Data-driven groups", metrics["n_clusters"])
    m3.metric("Outliers (no clear group)", metrics["n_outliers"])
    m4.metric("Agreement with genetic types (ARI)", f"{metrics['ari']:.2f}")
    st.caption(
        "ARI (Adjusted Rand Index) measures how well the data-driven clusters line up with "
        "the reported deposit types. A low score here is not a failure of the clustering — "
        "per the guidance note, it usually means several conventional types are geologically "
        "similar enough to merge, which is exactly the kind of finding this exercise is for."
    )


# ---------------------------------------------------------------------------
# PAGE 2 — Library Explorer
# ---------------------------------------------------------------------------
elif page == "Library Explorer (Global Deposit DB)":
    st.title("Library Explorer")
    st.caption("Job 1 + Job 2: data-driven groups, compared against conventional deposit types.")

    fig = px.scatter(
        df, x="emb_x", y="emb_y",
        color=df["cluster"].astype(str),
        hover_data={"pdf_hash": True, "deposit_type_primary": True, "emb_x": False, "emb_y": False},
        title="2D map of all reports, coloured by data-driven group",
        labels={"color": "Cluster"},
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    chosen = st.selectbox(
        "Inspect a group", options=[-1] + CLUSTER_IDS,
        format_func=cluster_label, index=1,
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Geological signature")
        if chosen in signatures:
            sig = signatures[chosen]
            rows = []
            for feat_col, lift in sig.items():
                field, label = decode_feature_column(feat_col)
                rows.append({"Field": pretty_field_name(field), "Attribute": label, "Lift": round(lift, 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        else:
            st.info("No signature available (outlier group).")

    with right:
        st.subheader("Conventional deposit types in this group")
        if chosen in crosstab.index:
            counts = crosstab.loc[chosen].sort_values(ascending=False)
            counts = counts[counts > 0].head(10)
            counts.index = [decode_code("deposit_type", c) for c in counts.index]
            st.bar_chart(counts)
        else:
            st.info("No labelled reports in this group.")

    st.markdown("---")
    st.subheader("Reports in this group")
    sub = df[df["cluster"] == chosen][
        ["pdf_hash", "deposit_type_primary", "cluster_majority_label", "label_mismatch"]
    ].copy()
    sub["deposit_type_primary"] = sub["deposit_type_primary"].apply(
        lambda c: decode_code("deposit_type", c) if c else "—"
    )
    sub["cluster_majority_label"] = sub["cluster_majority_label"].apply(
        lambda c: decode_code("deposit_type", c) if c else "—"
    )
    st.dataframe(sub.head(200), hide_index=True, width='stretch')
    st.caption(f"Showing up to 200 of {len(sub)} reports in this group.")


# ---------------------------------------------------------------------------
# PAGE 3 — Project Analogue Search
# ---------------------------------------------------------------------------
elif page == "Project Analogue Search":
    st.title("Project Analogue Search")
    st.caption(
        "Enter what your project's exploration data actually shows — mapping, structure, "
        "alteration, mineralisation, veining — and find the closest-matching groups in the library."
    )
    st.info(
        "Only fields a typical junior exploration project can realistically observe are used here "
        "(per the guidance note). Geochronology, isotope and fluid-inclusion data are intentionally "
        "excluded — they are research-grade analyses most projects will never have.",
        icon="ℹ️",
    )

    user_selections = {}
    cols = st.columns(2)
    for i, field in enumerate(CLUSTER_FEATURE_COLS):
        options_cols = [c for c in feature_columns if c.startswith(f"{field}::")]
        option_labels = {}
        for oc in options_cols:
            _, label = decode_feature_column(oc)
            option_labels[oc] = label
        with cols[i % 2]:
            chosen = st.multiselect(
                pretty_field_name(field),
                options=sorted(options_cols, key=lambda c: option_labels[c]),
                format_func=lambda c: option_labels[c],
                key=f"sel_{field}",
            )
            user_selections[field] = chosen

    st.markdown("---")
    run = st.button("Find analogues", type="primary")

    if run:
        chosen_cols = [c for v in user_selections.values() for c in v]
        if not chosen_cols:
            st.warning("Select at least a few observed attributes above.")
        else:
            # build user vector aligned to the saved feature-column order
            vec = pd.Series(0, index=feature_columns, dtype=np.uint8)
            vec[chosen_cols] = 1

            # --- nearest neighbours by raw Jaccard similarity in the original
            # feature space. This is used both to show individual analogue
            # reports AND to decide the best-matching group, by a majority
            # vote among the nearest neighbours. This is more robust than an
            # out-of-sample UMAP transform on a small/sparse user vector, and
            # arguably easier to defend to a geologist: "your project looks
            # most like these N reports" is a transparent, checkable claim.
            Xv = X.values.astype(bool)
            uv = vec.values.astype(bool)
            intersection = (Xv & uv).sum(axis=1)
            union = (Xv | uv).sum(axis=1)
            jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

            K = 15
            top_idx = np.argsort(-jaccard)[:K]
            neighbour_clusters = df.iloc[top_idx]["cluster"].values
            neighbour_sims = jaccard[top_idx]

            st.subheader("Result")
            if neighbour_sims.max() <= 0:
                st.warning(
                    "None of the selected attributes overlap with any report in the library. "
                    "Try selecting a few more observed attributes."
                )
            else:
                vals, counts_arr = np.unique(neighbour_clusters, return_counts=True)
                pred_cluster = int(vals[np.argmax(counts_arr)])
                agreement = counts_arr.max() / K

                if pred_cluster == -1 or agreement < 0.34:
                    st.warning(
                        "This combination of attributes does not match any existing group "
                        "clearly — the nearest historical reports are spread across several "
                        "different groups. Per the guidance note, that is a valid result: it "
                        "may indicate a hybrid system or a genuinely distinct signature worth "
                        "a closer look, rather than an error."
                    )
                else:
                    st.success(
                        f"Best match: {cluster_label(pred_cluster)}  \u2014  "
                        f"{int(agreement * K)} of {K} closest analogue reports fall in this group"
                    )
                    sig = signatures.get(pred_cluster)
                    if sig is not None:
                        st.markdown("**Typical signature of this group:**")
                        rows = []
                        for feat_col, lift in sig.items():
                            field, label = decode_feature_column(feat_col)
                            rows.append({"Field": pretty_field_name(field), "Attribute": label})
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

                    counts = crosstab.loc[pred_cluster].sort_values(ascending=False)
                    counts = counts[counts > 0].head(8)
                    if len(counts):
                        st.markdown("**Conventional deposit types seen in this group historically:**")
                        counts.index = [decode_code("deposit_type", c) for c in counts.index]
                        st.bar_chart(counts)

            # nearest individual analogue reports, by raw Jaccard similarity
            st.markdown("---")
            st.subheader("Closest individual analogue reports")
            nearest = df.iloc[top_idx][["pdf_hash", "deposit_type_primary", "cluster"]].copy()
            nearest["similarity"] = neighbour_sims.round(3)
            nearest["deposit_type_primary"] = nearest["deposit_type_primary"].apply(
                lambda c: decode_code("deposit_type", c) if c else "—"
            )
            st.dataframe(nearest, hide_index=True, width='stretch')


# ---------------------------------------------------------------------------
# PAGE 4 — Resource-Size Insights (placeholder — data gap)
# ---------------------------------------------------------------------------
elif page == "Resource-Size Insights":
    st.title("Resource-Size Insights")
    st.error(
        "**Not yet available — data gap.**\n\n"
        "The guidance note's third job for the library is to look for parameters, within a "
        "coherent group, that correlate with eventual resource size. This requires actual "
        "tonnage / grade / contained-metal figures per report.\n\n"
        "The current delivered dataset only has a boolean flag "
        "(`mineral_resource_estimate_exists`) — no magnitude. This page will be built as soon "
        "as resource-size figures are added to the extraction."
    )
    st.markdown(
        "**What this page will do once the data lands:**\n"
        "- Within each cluster from the Library Explorer, test which measured attributes "
        "correlate with resource size\n"
        "- Rank targets not just by deposit-type match, but by likely scale\n"
        "- Surface this ranking directly in Project Analogue Search results"
    )


# ---------------------------------------------------------------------------
# PAGE 5 — QA Review List
# ---------------------------------------------------------------------------
elif page == "QA Review List":
    st.title("QA Review List")
    st.caption(
        "Reports flagged because they fall outside any group, or because their group's "
        "dominant signature disagrees with the reported deposit type. Per the guidance note, "
        "these are findings worth investigating — not necessarily errors."
    )

    flagged = df[(df["is_outlier"]) | (df["label_mismatch"])][
        ["pdf_hash", "cluster", "deposit_type_primary", "cluster_majority_label", "is_outlier", "label_mismatch"]
    ].copy()
    flagged["deposit_type_primary"] = flagged["deposit_type_primary"].apply(
        lambda c: decode_code("deposit_type", c) if c else "—"
    )
    flagged["cluster_majority_label"] = flagged["cluster_majority_label"].apply(
        lambda c: decode_code("deposit_type", c) if c else "—"
    )
    flagged = flagged.rename(columns={
        "deposit_type_primary": "Stated deposit type",
        "cluster_majority_label": "Group's dominant type",
        "is_outlier": "No clear group",
        "label_mismatch": "Type mismatch",
    })

    st.metric("Reports flagged for review", len(flagged))
    st.dataframe(flagged, hide_index=True, width='stretch')

    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("Download full list as CSV", csv, "qa_review_list.csv", "text/csv")
