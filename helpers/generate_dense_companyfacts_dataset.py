"""Generate the dense ECL + SEC financial tags dataset.

This helper preserves the existing dense methodology used in ``main.py`` by:
1) loading the metadata dataset without text fields,
2) calling ``retrieve_sec_tags_and_values`` (which performs concept matching and
   value retrieval), and
3) saving the resulting dense dataset as CSV.
"""

import argparse
import json
import os
import sys
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from sec_data_parser import retrieve_sec_tags_and_values
from tools import config
from tools.data_loader import DataLoader


DEFAULT_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv")


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _get_config_financial_columns() -> List[str]:
    with open(config.XBRL_MAPPING_PATH, "r", encoding="utf-8") as file:
        xbrl_mapping = json.load(file)

    ordered_cols: List[str] = []
    for section in ["IncomeStatement", "BalanceSheet", "CashFlow", "StatementOfStockholdersEquity"]:
        for concept_name in xbrl_mapping.get(section, {}).keys():
            if concept_name not in ordered_cols:
                ordered_cols.append(concept_name)
    return ordered_cols


def postprocess_dense_csv(csv_path: str) -> None:
    """Keep only accession/label + configured financial columns and trim float tails."""
    df = pd.read_csv(csv_path, low_memory=False)

    keep_cols = ["accessionNumber", "label", *_get_config_financial_columns()]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

    # Ensure numeric columns are treated as numeric for CSV float formatting.
    for col in df.columns:
        if col in {"accessionNumber", "label"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # float_format trims trailing zeros (e.g. 2072000000.000000 -> 2072000000).
    df.to_csv(csv_path, index=False, float_format="%.15g")


def build_dense_raw(input_path: str, output_path: str) -> None:
    """Create the dense dataset using the existing retrieval/matching pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data_loader = DataLoader()
    ecl_metadata = data_loader.load_dataset(input_path, alias="ecl", lines=True)

    # Keep the historical methodology unchanged by delegating all retrieval/matching.
    retrieve_sec_tags_and_values(ecl_metadata, data_loader)

    # The retrieval function writes to the historical default output file.
    # If a custom output is provided, persist the same dataframe to that path as well.
    if os.path.abspath(output_path) != os.path.abspath(DEFAULT_OUTPUT_PATH):
        data_loader.save_dataset(ecl_metadata, out_path=output_path)


def create_dense_dataset(input_path: str, output_path: str, postprocess: bool = True) -> None:
    output_exists = os.path.isfile(output_path)

    if output_exists:
        print(f"Output already exists at {output_path}. Running postprocessing only.")
        postprocess_dense_csv(output_path)
        return

    build_dense_raw(input_path=input_path, output_path=output_path)

    if postprocess:
        postprocess_dense_csv(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense ECL + SEC financial tags dataset.")
    parser.add_argument(
        "--input",
        default=config.ECL_METADATA_NOTEXT_PATH,
        help="Path to metadata JSONL input (without opinion_text/item_7).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to dense CSV output.",
    )
    parser.add_argument(
        "--postprocess",
        type=_str_to_bool,
        default=True,
        help="Whether to drop non-required columns and trim trailing zeros on float values.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dense_dataset(input_path=args.input, output_path=args.output, postprocess=args.postprocess)
