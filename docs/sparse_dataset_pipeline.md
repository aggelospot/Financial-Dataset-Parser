# Sparse dataset generation pipeline (historical SEC companyfacts approach)

This note summarizes the original approach used in this project before the move to the local PostgreSQL SEC dataset.

## 1) Start from ECL records
- The pipeline starts from `ECL_AA_subset.json`.
- Each row is keyed by company-level metadata such as CIK / filing context.

## 2) Join ECL rows with local SEC companyfacts JSON files
- For each ECL row, clean/pad the CIK to 10 digits and read `data/companyfacts/CIK##########.json`.
- SEC data is read once per CIK and reused while iterating rows.
- Only `facts.us-gaap` and `facts.dei` concepts are considered.

## 3) Keep only annual-report facts and match by fiscal year
- For each concept, flatten all unit arrays (`USD`, shares, etc.) into a single list of data points.
- Keep only points where `form` contains `10-K`.
- Match the data point to the row’s filing year using the SEC `fy` field.
- If matched, write the concept value into that ECL row as a new column.

## 4) Persist the raw merged output
- The merged rows (ECL + matched companyfacts concepts) are written to
  `outputs/ecl_companyfacts_raw.json`.

## 5) Post-process to remove rows with zero matched financial concepts
- The project compares each row’s keys against the original ECL schema.
- If no additional financial keys are present, the row is dropped.
- Output is saved to `outputs/ecl_companyfacts.json`.

## 6) Quantify sparsity per concept
- Compute per-column null counts / null percentages on the merged dataset,
  excluding original ECL columns.
- Save statistics to `outputs/column_statistics.csv`.

## 7) Reduce sparsity by thresholding concepts
- Keep only columns with null percentage under a configured threshold
  (example in code: `max_null_percentage=10`).
- Keep only numeric selected columns.
- Reattach original ECL columns.
- Drop rows containing nulls in selected numeric SEC columns.
- Save final processed dataset to `outputs/ecl_companyfacts_processed.csv`.

## 8) Why this produced a sparse matrix
- The initial feature space included a very large number of SEC tags.
- Matching was restricted to annual 10-K data for a specific fiscal year.
- Company reporting taxonomies differ heavily across issuers (missing / custom tags).
- Result: many concept columns were null for many firms, yielding a sparse dataset.

## 9) Later transition noted in codebase
- The utility used to call SEC `companyfacts` API is marked as no longer used.
- The newer path in `main.py` calls `retrieve_sec_tags_and_values(...)`, which pulls values from a local PostgreSQL SEC dataset and mapped canonical concepts.
