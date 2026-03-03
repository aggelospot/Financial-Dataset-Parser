"""Helper script for basic analysis across generated datasets.

Current scope:
- Resolve a target dataset path (sparse/dense/metadata) from config, while
  still allowing explicit input path overrides.
- Infer metadata + label columns from the `ecl_companyfacts_no_text` dataset
  columns.
- Compute the distribution for the `isXBRL` boolean-like column.
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config
from tools.data_loader import DataLoader


DATASET_PATHS: Dict[str, str] = {
    "sparse": config.ECL_COMPANYFACTS_SPARSE_PATH,
    "dense": config.ECL_COMPANYFACTS_DENSE_PATH,
    "metadata": config.ECL_METADATA_PATH,
}


def resolve_dataset_path(dataset_name: str, input_path: Optional[str]) -> str:
    """Resolve analysis input path from dataset selector + optional override."""
    if input_path:
        return input_path

    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {sorted(DATASET_PATHS)}")

    return DATASET_PATHS[dataset_name]


def infer_metadata_and_numeric_columns(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Split columns into metadata/label vs numeric using ecl_companyfacts_no_text columns."""
    metadata_and_label_columns = [col for col in input_df.columns if col in reference_df.columns]
    numeric_columns = [col for col in input_df.columns if col not in reference_df.columns]

    return {
        "metadata_and_label_columns": metadata_and_label_columns,
        "numeric_columns": numeric_columns,
    }


def calculate_isxbrl_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Build count + percent table for the `isXBRL` column."""
    if "isXBRL" not in df.columns:
        raise ValueError("Column 'isXBRL' was not found in the selected dataset.")

    counts = df["isXBRL"].value_counts(dropna=False)
    percentages = (counts / len(df) * 100).round(4)

    summary = pd.DataFrame({
        "isXBRL": counts.index,
        "count": counts.values,
        "percentage": percentages.values,
    })
    return summary


def analyze_dataset(
    dataset_name: str,
    input_path: Optional[str] = None,
    reference_columns_path: str = config.ECL_COMPANYFACTS_DENSE_PATH,
) -> Dict[str, object]:
    """Run initial dataset analysis and return structured results."""
    loader = DataLoader()

    resolved_input_path = resolve_dataset_path(dataset_name=dataset_name, input_path=input_path)

    input_kwargs = {"lines": True} if resolved_input_path.lower().endswith(".json") else {}
    reference_kwargs = {"lines": True} if reference_columns_path.lower().endswith(".json") else {}

    input_df = loader.load_dataset(resolved_input_path, alias="target_dataset", **input_kwargs)
    reference_df = loader.load_dataset(reference_columns_path, alias="reference_dataset", **reference_kwargs)

    column_groups = infer_metadata_and_numeric_columns(input_df=input_df, reference_df=reference_df)
    isxbrl_distribution_df = calculate_isxbrl_distribution(input_df)

    return {
        "dataset": dataset_name,
        "input_path": resolved_input_path,
        "reference_columns_path": reference_columns_path,
        "row_count": len(input_df),
        "metadata_and_label_columns": column_groups["metadata_and_label_columns"],
        "numeric_columns": column_groups["numeric_columns"],
        "isXBRL_distribution": isxbrl_distribution_df.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sparse/dense/metadata datasets.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PATHS.keys()),
        required=True,
        help="Which dataset route to use from config.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional explicit path to override the config route for the selected dataset.",
    )
    parser.add_argument(
        "--reference-columns-path",
        default=config.ECL_COMPANYFACTS_DENSE_PATH,
        help="Path to ecl_companyfacts_no_text dataset used to infer metadata + label columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze_dataset(
        dataset_name=args.dataset,
        input_path=args.input,
        reference_columns_path=args.reference_columns_path,
    )

    print(f"Dataset: {results['dataset']}")
    print(f"Input path: {results['input_path']}")
    print(f"Reference columns path: {results['reference_columns_path']}")
    print(f"Rows analyzed: {results['row_count']}")
    print(f"Metadata + label columns: {len(results['metadata_and_label_columns'])}")
    print(f"Numerical columns: {len(results['numeric_columns'])}")

    print("\nisXBRL distribution:")
    for row in results["isXBRL_distribution"]:
        print(f"  isXBRL={row['isXBRL']!r}: count={row['count']}, pct={row['percentage']}")


if __name__ == "__main__":
    main()
