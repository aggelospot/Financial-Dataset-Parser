# Sparse companyfacts matching process

This note documents how the sparse companyfacts dataset matches SEC fact values
to ECL metadata rows.

## Input rows

The sparse generator reads the metadata JSONL configured by
`config.COMPANYFACTS_METADATA_PATH`. Each row must contain:

- `cik`, used to locate the local SEC companyfacts file.
- `accession_number`, used to match SEC fact points from the same filing.
- `cik_adsh`, the filing-level identifier written to the output dataset.
- `label` and `year`, which are retained as output columns.

## SEC companyfacts lookup

For each metadata row, the generator normalizes the row's CIK to a 10-digit
string and opens `data/companyfacts/CIK##########.json`. Files are cached by CIK
so each companyfacts JSON is read once and reused for later rows from the same
company.
[sparse_dataset_pipeline.md](sparse_dataset_pipeline.md)
Only concepts under `facts.us-gaap` and `facts.dei` are considered. For each
concept, all unit arrays are flattened into individual fact points.

## Accession-based fact matching

A fact point is accepted when:

- Its `form` contains `10-K`.
- Its SEC `accn` value equals the metadata row's `accession_number`.

The previous fiscal-year match is no longer used. The accession match ties the
value to the exact filing represented by the metadata row instead of relying on
the `fy` field.

When a concept has a matching fact point, the generator converts `val` to a
numeric value. Non-numeric values are ignored. The first matching numeric value
for a concept is written into that row.

## Output shape

The output CSV starts with:

```text
cik_adsh,label,year
```

All discovered numeric companyfacts concepts are appended after those columns in
sorted order. Rows without a matching value for a concept leave that concept
blank.
