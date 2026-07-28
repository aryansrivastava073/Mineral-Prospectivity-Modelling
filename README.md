# Mineral Prospectivity Modelling — Streamlit App

Built around the two-database architecture from the internal guidance note
(Global Deposit Database = library, Exploration Project Database = search data).

## Setup
```
pip install -r requirements.txt
```

## Rebuild the library (clustering) artifacts
Only needed if the underlying geology extraction CSVs change.
```
python data_prep.py            # sanity check the merged report table
python clustering_pipeline.py  # re-fits UMAP + HDBSCAN, saves to artifacts/
```

## Run the app
```
streamlit run app.py
```

## Pages
- **Framework Overview** — the two-database architecture, current library stats
- **Library Explorer** — Job 1 (data-driven clusters) + Job 2 (compared to conventional deposit types), 2D map, per-cluster signatures
- **Project Analogue Search** — enter a project's observed attributes, get nearest-neighbour cluster match + closest historical analogue reports
- **Resource-Size Insights** — placeholder; blocked on a real data gap (no tonnage/grade data delivered yet, only a boolean resource-exists flag)
- **QA Review List** — outliers + label-mismatch reports, exportable as CSV

## Known data gap
Job 3 from the guidance note (resource-size correlation) cannot be built until
actual tonnage/grade/contained-metal figures are added to the extraction.
