"""Experimental plotting helpers for generated thesis datasets.

Usage examples:
    python helpers/experimental_data_analysis.py
    python helpers/experimental_data_analysis.py --dataset metadata
    python helpers/experimental_data_analysis.py --dataset sparse --show
    python helpers/experimental_data_analysis.py --dataset ecl
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config
from tools.data_loader import DataLoader


DATASET_PATHS: Dict[str, str] = {
    "ecl": config.ECL_FILE_PATH,
    "sparse": config.COMPANYFACTS_SPARSE_PATH,
    "dense": config.COMPANYFACTS_DENSE_PATH,
    "text": config.MDA_AUDITOR_DATASET_PATH,
    "metadata": config.COMPANYFACTS_METADATA_PATH,
}

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "experimental_data_analysis")
DEFAULT_DENSE_TAG_AUDIT_PATH = os.path.join(config.OUTPUT_DIR, "tags.csv")
TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}
DEFAULT_TAG_LABEL_MAX_CHARS = 30
SPARSE_PLOT_COLOR = "#1f77b4"
CIK_ADSH_COLUMN = "cik_adsh"
SECTION_DISPLAY_NAMES = {
    "IncomeStatement": "Income Statement",
    "BalanceSheet": "Balance Sheet",
    "CashFlow": "Cash Flow Statement",
    "StatementOfStockholdersEquity": "Stockholders' Equity",
}
SPARSE_METADATA_COLUMNS = {
    "accession_number",
    "bankruptcy_date_1",
    "bankruptcy_date_2",
    "bankruptcy_date_3",
    "bankruptcy_prediction_split",
    "can_label",
    "cik",
    "cik_adsh",
    "cik_year",
    "company",
    "datadate",
    "filename",
    "filing_date",
    "form",
    "gc_list",
    "gvkey",
    "isXBRL",
    "label",
    "period_of_report",
    "primaryDocument",
    "qualified",
    "reportDateIndex",
    "year",
}


def _format_number(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional dependencies."""
    if df.empty:
        return "_No rows._"

    display_df = df.astype(object).where(pd.notna(df), "")
    headers = [str(column) for column in display_df.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in display_df.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def resolve_dataset_path(dataset_name: str) -> str:
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {sorted(DATASET_PATHS)}")

    return DATASET_PATHS[dataset_name]


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load one configured dataset using the same config paths as the other helpers."""
    dataset_path = resolve_dataset_path(dataset_name)
    _, extension = os.path.splitext(dataset_path.lower())

    kwargs = {"lines": True} if extension == ".json" else {"low_memory": False}
    return DataLoader().load_dataset(dataset_path, alias=dataset_name, **kwargs)


def load_csv_dataset(path: str, alias: str) -> pd.DataFrame:
    """Load a CSV analysis input with the same loader logging used elsewhere."""
    return DataLoader().load_dataset(path, alias=alias, low_memory=False)


def load_selected_and_metadata_datasets(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the selected dataset plus metadata, reusing metadata for the metadata route."""
    metadata_df = load_dataset("metadata")
    selected_df = metadata_df if dataset_name == "metadata" else load_dataset(dataset_name)
    return selected_df, metadata_df


def _normalize_bool_value(value: object) -> Optional[bool]:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    value_str = str(value).strip().lower()
    if value_str in TRUE_VALUES:
        return True
    if value_str in FALSE_VALUES:
        return False

    return None


def _extract_year_series(series: pd.Series) -> pd.Series:
    """Return numeric year values from year, date, or cik__year-style values."""
    numeric_years = pd.to_numeric(series, errors="coerce")
    if numeric_years.notna().any():
        return numeric_years

    date_years = pd.to_datetime(series, errors="coerce").dt.year
    if date_years.notna().any():
        return date_years

    return pd.to_numeric(series.astype(str).str.extract(r"(\d{4})$", expand=False), errors="coerce")


def plot_isxbrl_distribution_by_year(
    df: pd.DataFrame,
    year_column: str = "year",
    target_column: str = "isXBRL",
    output_path: Optional[str] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Plot grouped bars for isXBRL=True and isXBRL=False counts per year."""
    plt = _get_pyplot()

    missing_columns = [column for column in (year_column, target_column) if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for metadata plot: {missing_columns}")

    plot_df = pd.DataFrame(
        {
            "year": _extract_year_series(df[year_column]),
            "isXBRL": df[target_column].map(_normalize_bool_value),
        }
    ).dropna(subset=["year", "isXBRL"])
    plot_df["year"] = plot_df["year"].astype(int)

    counts = (
        plot_df.groupby(["year", "isXBRL"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[True, False], fill_value=0)
        .sort_index()
    )
    counts.columns = ["isXBRL=True", "isXBRL=False"]

    ax = counts.plot(kind="bar", stacked=False, figsize=(12, 6), width=0.82)
    ax.set_xlabel("Year")
    ax.set_ylabel("Rows")
    ax.set_title("isXBRL Distribution by Year")
    ax.legend(title="XBRL status")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close()

    return counts.reset_index()


def plot_bankruptcy_distribution_by_year(
    df: pd.DataFrame,
    year_column: str = "filing_date",
    target_column: str = "label",
    output_path: Optional[str] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Plot the count of bankruptcy=True labels per year."""
    plt = _get_pyplot()

    missing_columns = [column for column in (year_column, target_column) if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for ECL plot: {missing_columns}")

    plot_df = pd.DataFrame(
        {
            "year": _extract_year_series(df[year_column]),
            "bankruptcy": df[target_column].map(_normalize_bool_value),
        }
    ).dropna(subset=["year", "bankruptcy"])
    plot_df["year"] = plot_df["year"].astype(int)

    pre_2007_bankruptcy_count = int(((plot_df["year"] == 2008) & plot_df["bankruptcy"]).sum())
    print(f"Bankruptcy labels before 2007: {pre_2007_bankruptcy_count}")

    counts = (
        plot_df.loc[plot_df["bankruptcy"]]
        .groupby("year")
        .size()
        .rename("bankruptcy=True")
        .sort_index()
        .to_frame()
    )

    ax = counts.plot(kind="bar", stacked=False, figsize=(12, 6), width=0.82, legend=False)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bankruptcies")
    ax.set_title("Bankruptcy Label Distribution by Year")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close()

    return counts.reset_index()


def print_duplicate_filing_keys(
    df: pd.DataFrame,
    key_column: str = CIK_ADSH_COLUMN,
    top_n: int = 50,
) -> pd.DataFrame:
    """Print duplicated filing keys once, with duplicate counts."""
    if key_column not in df.columns:
        raise ValueError(f"Column '{key_column}' was not found in the metadata dataset.")

    duplicate_counts = (
        df.loc[df.duplicated(subset=[key_column], keep=False), key_column]
        .value_counts(sort=False)
        .rename_axis(key_column)
        .reset_index(name="row_count")
    )

    if duplicate_counts.empty:
        print(f"\nNo duplicate {key_column} values found in metadata.")
        return duplicate_counts

    print(f"\nDuplicate {key_column} values in metadata:")
    print(f"  Unique duplicated values: {len(duplicate_counts):,}")
    print(f"  Duplicate rows beyond first occurrence: {int((duplicate_counts['row_count'] - 1).sum()):,}")
    print(f"  Showing first {min(top_n, len(duplicate_counts)):,}:")
    print(duplicate_counts.head(top_n).to_string(index=False))

    return duplicate_counts


def infer_sparse_tag_columns(df: pd.DataFrame, metadata_columns: Sequence[str] = tuple(SPARSE_METADATA_COLUMNS)) -> List[str]:
    """Return SEC tag columns from a sparse companyfacts dataset."""
    metadata_column_set = set(metadata_columns)
    return [column for column in df.columns if column not in metadata_column_set]


def add_metadata_columns_by_filing_key(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    columns: Sequence[str],
    key_column: str = CIK_ADSH_COLUMN,
) -> pd.DataFrame:
    """Attach selected metadata columns to a dataset using the filing-level key."""
    if key_column not in df.columns:
        raise ValueError(f"Column '{key_column}' was not found in the selected dataset.")
    if key_column not in metadata_df.columns:
        raise ValueError(f"Column '{key_column}' was not found in the metadata dataset.")

    missing_metadata_columns = [column for column in columns if column not in metadata_df.columns]
    if missing_metadata_columns:
        raise ValueError(f"Metadata dataset is missing required columns: {missing_metadata_columns}")

    columns_to_add = [column for column in columns if column not in df.columns]
    if not columns_to_add:
        return df

    duplicate_key_count = int(metadata_df.duplicated(subset=[key_column]).sum())
    if duplicate_key_count:
        print(
            f"Metadata dataset contains {duplicate_key_count:,} duplicate {key_column} rows; "
            f"using the first row for each {key_column}."
        )

    metadata_subset = (
        metadata_df.loc[:, [key_column, *columns_to_add]]
        .drop_duplicates(subset=[key_column], keep="first")
    )
    enriched_df = df.merge(
        metadata_subset,
        on=key_column,
        how="left",
        sort=False,
        validate="many_to_one",
    )

    for column in columns_to_add:
        missing_count = int(enriched_df[column].isna().sum())
        print(
            f"Added metadata column '{column}' by {key_column}; "
            f"{missing_count:,} rows have no matched metadata value."
        )

    return enriched_df


def filter_xbrl_filings(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows explicitly marked as XBRL filings."""
    if "isXBRL" not in df.columns:
        print("Column 'isXBRL' was not found; using all rows for the top tag frequency plot.")
        return df

    xbrl_mask = df["isXBRL"].map(_normalize_bool_value) == True
    filtered_df = df.loc[xbrl_mask]
    excluded_rows = len(df) - len(filtered_df)
    print(
        f"Top tag frequency plot uses {len(filtered_df):,} isXBRL=True filings "
        f"and excludes {excluded_rows:,} non-XBRL filings."
    )

    if len(filtered_df) == 0:
        raise ValueError("No isXBRL=True rows were found for the top tag frequency plot.")

    return filtered_df


def calculate_sparse_presence_mask(df: pd.DataFrame, tag_columns: Sequence[str]) -> pd.DataFrame:
    """Return True where a sparse SEC tag has a reported value."""
    tag_df = df.loc[:, list(tag_columns)]
    present_mask = tag_df.notna()

    object_columns = tag_df.select_dtypes(include=["object", "string"]).columns
    if len(object_columns) > 0:
        present_mask.loc[:, object_columns] &= tag_df.loc[:, object_columns].apply(
            lambda series: series.astype(str).str.strip().ne("")
        )

    return present_mask


def calculate_value_presence_mask(df: pd.DataFrame, value_columns: Sequence[str]) -> pd.DataFrame:
    """Return True where a selected value column has a non-empty value."""
    value_df = df.loc[:, list(value_columns)]
    present_mask = value_df.notna()

    object_columns = value_df.select_dtypes(include=["object", "string"]).columns
    if len(object_columns) > 0:
        present_mask.loc[:, object_columns] &= value_df.loc[:, object_columns].apply(
            lambda series: series.astype(str).str.strip().ne("")
        )

    return present_mask


def calculate_sparse_tag_frequency(
    df: pd.DataFrame,
    tag_columns: Optional[Sequence[str]] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate how often each sparse SEC tag appears across filings."""
    if len(df) == 0:
        raise ValueError("Sparse dataset is empty; cannot calculate tag frequencies.")

    selected_tag_columns = list(tag_columns) if tag_columns is not None else infer_sparse_tag_columns(df)
    if not selected_tag_columns:
        raise ValueError("No SEC tag columns were found in the sparse dataset.")

    missing_columns = [column for column in selected_tag_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing sparse tag columns: {missing_columns}")

    present_mask = calculate_sparse_presence_mask(df=df, tag_columns=selected_tag_columns)
    present_counts = present_mask.sum(axis=0).sort_values(ascending=False)
    frequency_df = pd.DataFrame(
        {
            "tag": present_counts.index,
            "filing_count": present_counts.values,
            "filing_frequency": (present_counts.values / len(df) * 100).round(4),
        }
    )

    if top_n is not None:
        return frequency_df.head(top_n)
    return frequency_df


def print_sparse_dataset_summary(df: pd.DataFrame, tag_columns: Sequence[str]) -> None:
    """Print sparse dataset shape and filing-level completeness diagnostics."""
    if len(df) == 0:
        print("Sparse dataset is empty.")
        return

    present_mask = calculate_sparse_presence_mask(df=df, tag_columns=tag_columns)
    tags_per_filing = present_mask.sum(axis=1)
    observed_tag_values = int(tags_per_filing.sum())
    total_possible_tag_values = len(df) * len(tag_columns)
    density = (observed_tag_values / total_possible_tag_values * 100) if total_possible_tag_values else 0

    print("\nSparse dataset shape:")
    print(f"  Filings: {len(df):,}")
    print(f"  SEC tag columns: {len(tag_columns):,}")
    print(f"  Total possible tag values: {total_possible_tag_values:,}")
    print(f"  Observed tag values: {observed_tag_values:,}")
    print(f"  Overall density: {density:.2f}%")
    print(f"  Overall sparsity: {100 - density:.2f}%")

    print("\nTags per filing:")
    print(f"  Mean: {tags_per_filing.mean():,.1f}")
    print(f"  Median: {tags_per_filing.median():,.1f}")
    print(f"  Min: {int(tags_per_filing.min()):,}")
    print(f"  Max: {int(tags_per_filing.max()):,}")
    print(f"  P25 / P75: {tags_per_filing.quantile(0.25):,.1f} / {tags_per_filing.quantile(0.75):,.1f}")

    if "label" not in df.columns:
        print("\nLabel-wise completeness: skipped; 'label' column was not found.")
        return

    print("\nLabel-wise completeness:")
    label_summary = (
        pd.DataFrame({"label": df["label"], "tags_per_filing": tags_per_filing})
        .groupby("label", dropna=False)["tags_per_filing"]
        .agg(["count", "mean", "median"])
    )
    for label, row in label_summary.iterrows():
        label_density = (row["mean"] / len(tag_columns) * 100) if tag_columns else 0
        print(
            f"  label={label!r}: filings={int(row['count']):,}, "
            f"mean_tags={row['mean']:,.1f}, median_tags={row['median']:,.1f}, density={label_density:.2f}%"
        )


def truncate_label(value: object, max_chars: int = DEFAULT_TAG_LABEL_MAX_CHARS) -> str:
    """Return a compact display label while preserving full values in source data."""
    value_str = str(value)
    if len(value_str) <= max_chars:
        return value_str

    return f"{value_str[: max_chars - 3]}..."


def plot_sparse_top_tag_frequency(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    show: bool = False,
    top_n: int = 20,
    max_label_chars: int = DEFAULT_TAG_LABEL_MAX_CHARS,
) -> pd.DataFrame:
    """Plot the top sparse SEC tags by filing coverage."""
    plt = _get_pyplot()

    tag_columns = infer_sparse_tag_columns(df)
    plot_df_source = filter_xbrl_filings(df)

    frequency_df = calculate_sparse_tag_frequency(df=plot_df_source, tag_columns=tag_columns, top_n=top_n)
    plot_df = frequency_df.sort_values("filing_frequency", ascending=True).copy()
    plot_df["display_tag"] = plot_df["tag"].map(lambda value: truncate_label(value, max_chars=max_label_chars))

    fig_height = max(5, top_n * 0.5)
    ax = plot_df.plot(
        kind="barh",
        x="display_tag",
        y="filing_frequency",
        figsize=(12, fig_height),
        legend=False,
        color=SPARSE_PLOT_COLOR,
    )
    ax.set_xlabel("Frequency across XBRL filings (%)")
    ax.set_ylabel("SEC tag")
    ax.set_title(f"Top {top_n} SEC Tag Frequencies in Sparse Dataset (XBRL Filings Only)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(right=max(plot_df["filing_frequency"].max() * 1.08, 1))

    for container in ax.containers:
        ax.bar_label(container, labels=[f"{value:.1f}%" for value in plot_df["filing_frequency"]], padding=3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close()

    return frequency_df


def calculate_sparse_coverage_threshold_summary(
    frequency_df: pd.DataFrame,
    thresholds: Sequence[float] = (70, 50, 25, 10, 1),
) -> pd.DataFrame:
    """Count how many SEC tags appear in at least each filing-coverage threshold."""
    return pd.DataFrame(
        {
            "coverage_threshold": [f">= {threshold:g}%" for threshold in thresholds],
            "threshold_value": list(thresholds),
            "tag_count": [
                int((frequency_df["filing_frequency"] >= threshold).sum())
                for threshold in thresholds
            ],
        }
    )


def plot_sparse_coverage_threshold_summary(
    frequency_df: pd.DataFrame,
    output_path: Optional[str] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Plot how many sparse SEC tags appear in at least each coverage threshold."""
    plt = _get_pyplot()

    threshold_df = calculate_sparse_coverage_threshold_summary(frequency_df=frequency_df)

    ax = threshold_df.plot(
        kind="bar",
        x="coverage_threshold",
        y="tag_count",
        figsize=(10, 6),
        legend=False,
        color=SPARSE_PLOT_COLOR,
    )
    ax.set_xlabel("Appears in at least this share of filings")
    ax.set_ylabel("Number of tags")
    ax.set_title("Common SEC tags Across Filings")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=0)

    for container in ax.containers:
        ax.bar_label(container, labels=[f"{value:,}" for value in threshold_df["tag_count"]], padding=3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close()

    return threshold_df


def load_xbrl_mapping(mapping_path: str = config.XBRL_MAPPING_PATH) -> dict:
    with open(mapping_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_dense_concepts_by_section(xbrl_mapping: dict) -> Dict[str, List[str]]:
    """Return standardized dense concepts grouped by financial statement section."""
    return {
        section_name: list(xbrl_mapping.get(section_name, {}).keys())
        for section_name in SECTION_DISPLAY_NAMES
    }


def flatten_concepts(concepts_by_section: Dict[str, Sequence[str]]) -> List[str]:
    """Flatten section concept lists while preserving first-seen mapping order."""
    concepts: List[str] = []
    for section_concepts in concepts_by_section.values():
        for concept in section_concepts:
            if concept not in concepts:
                concepts.append(concept)
    return concepts


def validate_dense_concept_columns(df: pd.DataFrame, concept_columns: Sequence[str], dataset_label: str) -> None:
    missing_columns = [column for column in concept_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_label} is missing {len(missing_columns):,} standardized concept column(s): "
            f"{missing_columns}"
        )


def filter_dense_xbrl_filings(
    dense_df: pd.DataFrame,
    tag_audit_df: pd.DataFrame,
    key_column: str = CIK_ADSH_COLUMN,
) -> pd.DataFrame:
    """Filter dense rows to filings listed as XBRL in the dense tag audit output."""
    if key_column not in dense_df.columns:
        raise ValueError(f"Column '{key_column}' was not found in the dense dataset.")
    if key_column not in tag_audit_df.columns:
        raise ValueError(f"Column '{key_column}' was not found in the dense tag audit dataset.")

    audit_df = tag_audit_df
    if "isXBRL" in audit_df.columns:
        audit_df = audit_df.loc[audit_df["isXBRL"].map(_normalize_bool_value) == True]

    eligible_keys = set(audit_df[key_column].dropna().astype(str))
    filtered_df = dense_df.loc[dense_df[key_column].astype(str).isin(eligible_keys)].copy()
    excluded_rows = len(dense_df) - len(filtered_df)
    unmatched_audit_rows = len(eligible_keys - set(filtered_df[key_column].dropna().astype(str)))

    print(
        f"\nDense analysis uses {len(filtered_df):,} XBRL filings and excludes "
        f"{excluded_rows:,} non-XBRL dense rows."
    )
    if unmatched_audit_rows:
        print(f"  Audit keys not found in dense dataset: {unmatched_audit_rows:,}")

    if len(filtered_df) == 0:
        raise ValueError("No dense rows matched the XBRL tag audit output.")

    return filtered_df


def calculate_dense_completeness_summary(
    dense_xbrl_df: pd.DataFrame,
    concept_columns: Sequence[str],
) -> pd.DataFrame:
    """Calculate compact filing-level completeness metrics for the dense dataset."""
    present_mask = calculate_value_presence_mask(df=dense_xbrl_df, value_columns=concept_columns)
    values_per_filing = present_mask.sum(axis=1)
    observed_values = int(values_per_filing.sum())
    total_possible_values = len(dense_xbrl_df) * len(concept_columns)
    density = (observed_values / total_possible_values * 100) if total_possible_values else 0
    mean_values_per_filing = float(values_per_filing.mean()) if len(values_per_filing) else 0

    return pd.DataFrame(
        [
            {
                "metric": "Total standardized financial concepts",
                "value": f"{len(concept_columns):,}",
            },
            {
                "metric": "Total eligible filings (XBRL only)",
                "value": f"{len(dense_xbrl_df):,}",
            },
            {
                "metric": "Overall observed-value density (%)",
                "value": f"{density:.2f}",
            },
            {
                "metric": "Mean financial values per filing",
                "value": f"{mean_values_per_filing:.2f}",
            },
        ]
    )


def calculate_dense_section_coverage_summary(
    dense_xbrl_df: pd.DataFrame,
    concepts_by_section: Dict[str, Sequence[str]],
) -> pd.DataFrame:
    """Calculate observed-value coverage by standardized statement section."""
    rows = []
    for section_name, section_concepts in concepts_by_section.items():
        present_mask = calculate_value_presence_mask(df=dense_xbrl_df, value_columns=section_concepts)
        values_per_filing = present_mask.sum(axis=1)
        observed_values = int(values_per_filing.sum())
        possible_values = len(dense_xbrl_df) * len(section_concepts)
        density = (observed_values / possible_values * 100) if possible_values else 0

        rows.append(
            {
                "statement_section": SECTION_DISPLAY_NAMES[section_name],
                "standardized_concepts": len(section_concepts),
                "observed_values": observed_values,
                "possible_values": possible_values,
                "observed_value_density_pct": round(density, 2),
                "mean_values_per_filing": round(float(values_per_filing.mean()), 2)
                if len(values_per_filing)
                else 0,
            }
        )

    return pd.DataFrame(rows)


def calculate_dense_tag_mapping_diversity(
    tag_audit_df: pd.DataFrame,
    concepts_by_section: Dict[str, Sequence[str]],
) -> pd.DataFrame:
    """Count distinct SEC tags mapped to each standardized dense concept."""
    concept_to_section = {
        concept: SECTION_DISPLAY_NAMES[section_name]
        for section_name, section_concepts in concepts_by_section.items()
        for concept in section_concepts
    }
    concept_columns = list(concept_to_section)
    validate_dense_concept_columns(tag_audit_df, concept_columns, "Dense tag audit dataset")

    audit_df = tag_audit_df
    if "isXBRL" in audit_df.columns:
        audit_df = audit_df.loc[audit_df["isXBRL"].map(_normalize_bool_value) == True]

    rows = []
    for concept in concept_columns:
        mapped_tags = audit_df[concept].dropna().astype(str).str.strip()
        mapped_tags = mapped_tags.loc[mapped_tags.ne("")]
        tag_counts = mapped_tags.value_counts()

        most_frequent_tag = tag_counts.index[0] if not tag_counts.empty else ""
        most_frequent_count = int(tag_counts.iloc[0]) if not tag_counts.empty else 0
        mapped_filings = int(mapped_tags.count())
        most_frequent_share = (most_frequent_count / mapped_filings * 100) if mapped_filings else 0

        rows.append(
            {
                "standardized_concept": concept,
                "statement_section": concept_to_section[concept],
                "mapped_filings": mapped_filings,
                "distinct_sec_tags": int(mapped_tags.nunique()),
                "most_frequent_sec_tag": most_frequent_tag,
                "most_frequent_sec_tag_count": most_frequent_count,
                "most_frequent_sec_tag_share_pct": round(most_frequent_share, 2),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["distinct_sec_tags", "mapped_filings", "standardized_concept"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def write_dense_analysis_markdown(
    summary_df: pd.DataFrame,
    section_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    output_path: str,
    top_n: int = 20,
) -> None:
    """Write a short thesis-ready dense analysis section built around tables."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_display = summary_df.copy()
    summary_display["value"] = summary_display["value"].map(_format_number)

    section_display = section_df.copy()
    for column in ("standardized_concepts", "observed_values", "possible_values"):
        section_display[column] = section_display[column].map(_format_number)
    for column in ("observed_value_density_pct", "mean_values_per_filing"):
        section_display[column] = section_display[column].map(_format_number)

    diversity_display = diversity_df.head(top_n).copy()
    for column in ("mapped_filings", "distinct_sec_tags", "most_frequent_sec_tag_count"):
        diversity_display[column] = diversity_display[column].map(_format_number)
    diversity_display["most_frequent_sec_tag_share_pct"] = diversity_display[
        "most_frequent_sec_tag_share_pct"
    ].map(_format_number)

    markdown = f"""# Dense Dataset Analysis

This section summarizes the dense dataset after excluding filings that are not marked as XBRL in the dense tag-audit output. The goal is to describe completeness and mapping behavior with compact tables, rather than adding another set of frequency bar charts.

## Completeness Summary

{_dataframe_to_markdown(summary_display)}

## Coverage by Financial Statement Section

{_dataframe_to_markdown(section_display)}

## Tag-Audit Mapping Diversity

The tag-audit table records which original SEC tag was selected for each standardized concept in each XBRL filing. Concepts with many distinct mapped SEC tags illustrate heterogeneity in XBRL reporting and the practical complexity of standardizing SEC filings. These counts should be interpreted as evidence of reporting diversity, not as proof that the dense dataset is inherently superior.

{_dataframe_to_markdown(diversity_display)}
"""

    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(markdown)


def run_metadata_graphs(df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: str, show: bool) -> List[str]:
    print_duplicate_filing_keys(df=df, top_n=150)

    output_path = os.path.join(output_dir, "metadata_isxbrl_distribution_by_year.png")
    plot_isxbrl_distribution_by_year(df=df, output_path=output_path, show=show)
    return [output_path]


def run_ecl_graphs(df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: str, show: bool) -> List[str]:
    output_path = os.path.join(output_dir, "ecl_bankruptcy_distribution_by_year.png")
    plot_bankruptcy_distribution_by_year(df=df, output_path=output_path, show=show)
    return [output_path]


def run_sparse_graphs(df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: str, show: bool) -> List[str]:
    enriched_df = add_metadata_columns_by_filing_key(
        df=df,
        metadata_df=metadata_df,
        columns=["isXBRL"],
    )
    tag_columns = infer_sparse_tag_columns(df)
    print_sparse_dataset_summary(df=df, tag_columns=tag_columns)

    frequency_output_path = os.path.join(output_dir, "sparse_top_20_tag_frequencies.png")
    threshold_output_path = os.path.join(output_dir, "sparse_coverage_threshold_summary.png")
    full_frequency_df = calculate_sparse_tag_frequency(df=df, tag_columns=tag_columns)

    plot_sparse_top_tag_frequency(
        df=enriched_df,
        output_path=frequency_output_path,
        show=show,
        top_n=20,
    )
    plot_sparse_coverage_threshold_summary(
        frequency_df=full_frequency_df,
        output_path=threshold_output_path,
        show=show,
    )
    return [frequency_output_path, threshold_output_path]


def run_dense_analysis(df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: str, show: bool) -> List[str]:
    del metadata_df, show

    os.makedirs(output_dir, exist_ok=True)
    xbrl_mapping = load_xbrl_mapping()
    concepts_by_section = get_dense_concepts_by_section(xbrl_mapping=xbrl_mapping)
    concept_columns = flatten_concepts(concepts_by_section)
    validate_dense_concept_columns(df, concept_columns, "Dense dataset")

    tag_audit_df = load_csv_dataset(DEFAULT_DENSE_TAG_AUDIT_PATH, alias="dense_tag_audit")
    dense_xbrl_df = filter_dense_xbrl_filings(df, tag_audit_df)

    completeness_df = calculate_dense_completeness_summary(
        dense_xbrl_df=dense_xbrl_df,
        concept_columns=concept_columns,
    )
    section_df = calculate_dense_section_coverage_summary(
        dense_xbrl_df=dense_xbrl_df,
        concepts_by_section=concepts_by_section,
    )
    diversity_df = calculate_dense_tag_mapping_diversity(
        tag_audit_df=tag_audit_df,
        concepts_by_section=concepts_by_section,
    )

    completeness_output_path = os.path.join(output_dir, "dense_completeness_summary.csv")
    section_output_path = os.path.join(output_dir, "dense_section_coverage_summary.csv")
    diversity_output_path = os.path.join(output_dir, "dense_tag_mapping_diversity.csv")
    markdown_output_path = os.path.join(output_dir, "dense_dataset_analysis.md")

    completeness_df.to_csv(completeness_output_path, index=False)
    section_df.to_csv(section_output_path, index=False)
    diversity_df.to_csv(diversity_output_path, index=False)
    write_dense_analysis_markdown(
        summary_df=completeness_df,
        section_df=section_df,
        diversity_df=diversity_df,
        output_path=markdown_output_path,
    )

    print("\nDense completeness summary:")
    print(completeness_df.to_string(index=False))
    print("\nDense coverage by statement section:")
    print(section_df.to_string(index=False))
    print("\nDense tag-audit mapping diversity (top 20 by distinct SEC tags):")
    print(diversity_df.head(20).to_string(index=False))

    return [
        completeness_output_path,
        section_output_path,
        diversity_output_path,
        markdown_output_path,
    ]


def run_placeholder_graphs(
    dataset_name: str,
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: str,
    show: bool,
) -> List[str]:
    print(f"Loaded {dataset_name} dataset with {len(df):,} rows.")
    print(f"No experimental graphs are registered for '{dataset_name}' yet.")
    return []


def run_graphs_for_dataset(dataset_name: str, output_dir: str, show: bool = False) -> List[str]:
    df, metadata_df = load_selected_and_metadata_datasets(dataset_name)

    graph_runner_by_dataset: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, str, bool], List[str]]] = {
        "dense": run_dense_analysis,
        "ecl": run_ecl_graphs,
        "metadata": run_metadata_graphs,
        "sparse": run_sparse_graphs,
    }

    match dataset_name:
        case "dense":
            graph_runner = graph_runner_by_dataset["dense"]
        case "ecl":
            graph_runner = graph_runner_by_dataset["ecl"]
        case "metadata":
            graph_runner = graph_runner_by_dataset["metadata"]
        case "sparse":
            graph_runner = graph_runner_by_dataset["sparse"]
        case "text":
            graph_runner = lambda loaded_df, loaded_metadata_df, out_dir, should_show: run_placeholder_graphs(
                dataset_name,
                loaded_df,
                loaded_metadata_df,
                out_dir,
                should_show,
            )
        case _:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {sorted(DATASET_PATHS)}")

    return graph_runner(df, metadata_df, output_dir, show)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental analysis plots for generated datasets.")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="metadata",
        choices=sorted(DATASET_PATHS.keys()),
        help="Dataset to analyze. Defaults to metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated plots will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = run_graphs_for_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        show=args.show,
    )

    if output_paths:
        print("Generated outputs:")
        for output_path in output_paths:
            print(f"  {output_path}")


if __name__ == "__main__":
    main()
