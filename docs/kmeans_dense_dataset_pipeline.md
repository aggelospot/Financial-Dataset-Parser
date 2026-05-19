# K-means dense companyfacts pipeline

This note documents the alternative dense companyfacts generator in
`helpers/generate_kmeans_companyfacts_dataset.py`.

The pipeline reuses the same metadata iteration, XBRL checks, primary registrant
filter, financial statement sections, and
`retrieve_statement_taxonomies_by_accession_number3(...)` database lookup used by
the fuzzy dense generator. It does not replace `companyfacts_dense.csv`; it
creates a separate modelling-oriented dataset with empirically clustered SEC
tags.

## Method

The generator first writes a long intermediate table:

```text
cik_adsh,label,year,section,tag,tlabel,value
```

This table preserves the observed SEC tag, SEC tag label, statement section, and
numeric value for each filing fact. Keeping the table on disk makes the
clustering input auditable and avoids changing the existing database retrieval
logic.

The second pass fits one k-means model per financial statement section over
unique observed `(tag, tlabel)` pairs. Text features are built with:

```python
TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
```

Each section uses `KMeans(random_state=42, n_init=20)`. Requested cluster counts
default to:

```text
IncomeStatement=60
BalanceSheet=80
CashFlow=50
StatementOfStockholdersEquity=40
```

If a section has fewer distinct observed tag/label texts than requested
clusters, the fitted cluster count is reduced for that section.

Cluster IDs are relabeled deterministically after fitting by ordering clusters
by their nearest observed representative tag/label text. This makes repeated
runs stable when the input data and requested cluster counts are unchanged.

## Output

The dense clustered CSV starts with:

```text
cik_adsh,label,year
```

Cluster columns are appended by section:

```text
is_cluster_001, bs_cluster_001, cf_cluster_001, eq_cluster_001, ...
```

When multiple observed tags from the same filing land in one cluster, the
default `nearest` aggregation selects the value from the observed tag closest to
that cluster centroid. The experimental `mean` aggregation averages numeric
values within the filing/cluster instead.

## Audit Files

The generator writes these audit files by default:

- `outputs/kmeans_companyfacts_long.csv`: long extracted source facts.
- `outputs/kmeans_tag_clusters.csv`: one row per unique section/tag/label with
  its cluster, centroid distance, and cluster representative.
- `outputs/kmeans_tags.csv`: one row per filing showing which SEC tag populated
  each dense cluster column.

These clusters are empirical tag families, not hand-validated accounting
concepts. A populated cluster column means the filing had an observed SEC tag
assigned to that cluster by text similarity. It should not be interpreted as a
manually reviewed canonical accounting concept.

## CLI

Example smoke run:

```powershell
python helpers/generate_kmeans_companyfacts_dataset.py --max-rows 100
```

Example with explicit cluster overrides:

```powershell
python helpers/generate_kmeans_companyfacts_dataset.py `
  --clusters IncomeStatement=60 BalanceSheet=80 CashFlow=50 StatementOfStockholdersEquity=40
```

Useful options:

- `--input`: metadata JSONL/CSV input, defaulting to
  `config.COMPANYFACTS_METADATA_PATH`.
- `--output`: dense clustered CSV path, defaulting to
  `outputs/companyfacts_kmeans_dense.csv`.
- `--clusters`: optional per-section overrides.
- `--max-rows`: metadata row limit for smoke tests.
- `--aggregation`: `nearest` by default, or `mean` for experimental modelling.
- `--intermediate-output`, `--tag-clusters-output`, `--tags-output`: optional
  paths for the long table and audit files.
