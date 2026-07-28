"""
Data preparation for the Global Deposit Database (the "library").

Loads the 9 geology extraction tables + metadata, aggregates them to one
row per pdf_hash, and separates:
  - CLUSTERING FEATURES: measured geological attributes a junior project
    could plausibly also observe (host rock, alteration, mineralisation,
    vein, fault, ore shape, pathfinder elements).
  - COMPARISON LABELS (held out from clustering): deposit_type,
    tectonic_setting -- interpretive / author-assigned, used only to
    check the data-driven groups against afterward (per guidance note).
  - CONTEXT-ONLY fields: geophysics_type (describes what surveys were
    flown, not the deposit itself -- kept for display, not clustering).
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _split_tags(series):
    """Split pipe-separated tag strings into lists, treating NaN/N/A as empty."""
    return series.fillna("").replace("N/A", "").apply(
        lambda s: [t for t in s.split("|") if t and t != "N/A"]
    )


def _agg_union(df, pdf_col, value_col):
    """For a table with possibly multiple rows per pdf_hash, union all tag
    values (including pipe-separated ones within a single cell) into one
    set of tags per pdf_hash."""
    tmp = df[[pdf_col, value_col]].copy()
    tmp["_tags"] = _split_tags(tmp[value_col].astype(str))
    grouped = tmp.groupby(pdf_col)["_tags"].apply(
        lambda lists: sorted(set(t for lst in lists for t in lst))
    )
    return grouped


def load_raw_tables():
    files = {
        "deposit": "1784804777264_geology_deposit.csv",
        "host_rock": "1784804777259_geology_host_rock.csv",
        "mineralization": "1784804777260_geology_mineralization.csv",
        "orebody_geometry": "1784804777261_geology_orebody_geometry.csv",
        "vein": "1784804777262_geology_vein.csv",
        "alteration": "1784804777263_geology_alteration.csv",
        "fault": "1784804777264_geology_fault.csv",
        "geophysics": "1784804777257_geology_geophysics.csv",
        "geochem_ratio": "1784804777256_geology_geochem_ratio.csv",
    }
    return {name: pd.read_csv(DATA_DIR / f, sep=";") for name, f in files.items()}


def build_report_table():
    """Return a per-pdf_hash DataFrame with one column per attribute,
    each cell holding a *list* of tag codes present for that report."""
    t = load_raw_tables()
    out = pd.DataFrame(index=sorted(set(t["deposit"]["pdf_hash"])))

    # --- clustering features (measured geology, junior-observable) ---
    out["host_unit"] = _agg_union(t["host_rock"], "pdf_hash", "host_unit")
    out["host_age"] = _agg_union(t["host_rock"], "pdf_hash", "host_age")
    out["mineralization_style"] = _agg_union(t["mineralization"], "pdf_hash", "mineralization_style")
    out["mineralization_texture"] = _agg_union(t["mineralization"], "pdf_hash", "mineralization_texture")
    out["continuity"] = _agg_union(t["mineralization"], "pdf_hash", "continuity")
    out["control_type"] = _agg_union(t["mineralization"], "pdf_hash", "control_type")
    out["ore_minerals"] = _agg_union(t["mineralization"], "pdf_hash", "ore_minerals")
    out["gangue_minerals"] = _agg_union(t["mineralization"], "pdf_hash", "gangue_minerals")
    out["ore_shape"] = _agg_union(t["orebody_geometry"], "pdf_hash", "ore_shape")
    out["vein_type"] = _agg_union(t["vein"], "pdf_hash", "vein_type")
    out["alteration_type"] = _agg_union(t["alteration"], "pdf_hash", "alteration_type")
    out["alteration_timing"] = _agg_union(t["alteration"], "pdf_hash", "alteration_timing")
    out["alteration_minerals"] = _agg_union(t["alteration"], "pdf_hash", "alteration_minerals")
    out["alteration_intensity"] = _agg_union(t["alteration"], "pdf_hash", "alteration_intensity")
    out["fault_setting"] = _agg_union(t["fault"], "pdf_hash", "fault_setting")
    out["structural_order"] = _agg_union(t["fault"], "pdf_hash", "structural_order")
    out["pathfinder_elements"] = _agg_union(t["deposit"], "pdf_hash", "pathfinder_elements")

    # --- comparison labels (held OUT of clustering, used afterward) ---
    out["deposit_type"] = _agg_union(t["deposit"], "pdf_hash", "deposit_type")
    out["tectonic_setting"] = _agg_union(t["deposit"], "pdf_hash", "tectonic_setting")

    # --- context-only (not geology of the deposit; describes data collected) ---
    out["geophysics_type"] = _agg_union(t["geophysics"], "pdf_hash", "geophysics_type")

    # fill any pdf_hash missing from a given table with empty lists
    for col in out.columns:
        out[col] = out[col].apply(lambda v: v if isinstance(v, list) else [])

    out.index.name = "pdf_hash"
    return out.reset_index()


CLUSTER_FEATURE_COLS = [
    "host_unit", "host_age", "mineralization_style", "mineralization_texture",
    "continuity", "control_type", "ore_minerals", "gangue_minerals", "ore_shape",
    "vein_type", "alteration_type", "alteration_timing", "alteration_minerals",
    "alteration_intensity", "fault_setting", "structural_order", "pathfinder_elements",
]

LABEL_COLS = ["deposit_type", "tectonic_setting"]
CONTEXT_COLS = ["geophysics_type"]


def load_dictionary():
    dic = pd.read_excel(DATA_DIR / "geology_code_dictionary.xlsx", sheet_name="Dictionary")
    minerals = pd.read_excel(DATA_DIR / "geology_code_dictionary.xlsx", sheet_name="Minerals")
    # code -> meaning lookup per (table, column)
    lookup = {}
    for _, row in dic.iterrows():
        lookup[(row["table"], row["column"], row["code"])] = row["meaning"]
    mineral_lookup = dict(zip(minerals["code"], minerals["mineral"]))
    return lookup, mineral_lookup


if __name__ == "__main__":
    df = build_report_table()
    print(df.shape)
    print(df.head(3).to_string())
    non_empty = {c: (df[c].apply(len) > 0).sum() for c in CLUSTER_FEATURE_COLS + LABEL_COLS + CONTEXT_COLS}
    print(non_empty)
