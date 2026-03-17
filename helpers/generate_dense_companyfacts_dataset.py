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
from decimal import Decimal

import pandas as pd

from db_connection import retrieve_statement_taxonomies_by_accession_number2, close_connection, create_connection
from tools.utils import match_concept_in_section


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config
from tools.data_loader import DataLoader


DEFAULT_INPUT_PATH = getattr(config, "COMPANYFACTS_METADATA_PATH", config.ECL_METADATA_NOTEXT_PATH)
DEFAULT_OUTPUT_PATH = getattr(config, "COMPANYFACTS_DENSE_PATH", os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv"))


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _configured_financial_columns() -> List[str]:
    with open(config.XBRL_MAPPING_PATH, "r", encoding="utf-8") as file:
        xbrl_mapping = json.load(file)

    columns: List[str] = []
    for section in ["IncomeStatement", "BalanceSheet", "CashFlow", "StatementOfStockholdersEquity"]:
        for key in xbrl_mapping.get(section, {}).keys():
            if key not in columns:
                columns.append(key)
    return columns


def postprocess_dense_csv(output_path: str) -> None:
    print("postprocessing....")
    """Drop non-required columns and trim float tails in-place."""
    df = pd.read_csv(output_path, low_memory=False)

    keep_columns = ["accessionNumber", "label", *_configured_financial_columns()]
    keep_columns = [col for col in keep_columns if col in df.columns]
    df = df[keep_columns]


    for col in df.columns:
        if col in {"accessionNumber", "label"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # This trims trailing zeros after decimal point during CSV writing.
    df.to_csv(output_path, index=False, float_format="%.15g")
    print("postprocessing finished")

def create_dense_dataset(input_path: str, output_path: str, postprocess: bool = True) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If output already exists, do postprocessing only; do not rebuild raw output.
    if os.path.isfile(output_path):
        print(f"Output already exists at {output_path}. Running postprocessing only.")
        postprocess_dense_csv(output_path)
        return

    conn = None
    try:
        data_loader = DataLoader()
        conn = create_connection()

        metadata = data_loader.load_dataset(input_path, alias='ecl', lines=True)
        with open(config.XBRL_MAPPING_PATH, 'r', encoding='utf-8') as file:
            xbrl_mapping = json.load(file)

        tag_list = []
        for section in ["IncomeStatement", "BalanceSheet", "CashFlow", "StatementOfStockholdersEquity"]:
            for key, value in xbrl_mapping[section].items():
                if key not in tag_list:
                    tag_list.append(key)
        for tag in tag_list:
            metadata[tag] = None

        results = []
        filing_cache = {}
        for idx, row in metadata.iterrows():
            print(f"\rCurrent row: {idx}/{len(metadata)}", end='')
            if row['isXBRL'] == 0:
                continue

            adsh = row['accessionNumber']
            filing_cache[adsh] = {}

            def cache_statement(stmt_code, quarter_spec):
                tags, labels, vals = retrieve_statement_taxonomies_by_accession_number2(
                    conn,
                    adsh, stmt_code, quarter_spec)
                filing_cache[adsh].update(dict(zip(tags, vals)))
                return tags, labels

            is_tags, is_labels = cache_statement('IS', [4])
            matched_is_items = match_concept_in_section(
                xbrl_mapping['IncomeStatement'],
                is_tags, is_labels)

            bs_tags, bs_labels = cache_statement('BS', 0)
            matched_bs_items = match_concept_in_section(
                xbrl_mapping['BalanceSheet'],
                bs_tags, bs_labels)

            cf_tags, cf_labels = cache_statement('CF', [4])
            matched_cf_items = match_concept_in_section(
                xbrl_mapping['CashFlow'],
                cf_tags, cf_labels)

            eq_tags, eq_labels = cache_statement('EQ', [0])
            matched_eq_items = match_concept_in_section(
                xbrl_mapping['StatementOfStockholdersEquity'],
                eq_tags, eq_labels)

            metadata_row = {
                "accession_number": adsh,
                "isXBRL": 1,
                **matched_is_items,
                **matched_bs_items,
                **matched_cf_items,
                **matched_eq_items,
            }

            for concept_name, sec_tag in metadata_row.items():
                if concept_name in tag_list:
                    val = filing_cache[adsh].get(sec_tag)
                    metadata.at[idx, concept_name] = val

            results.append(metadata_row)

        result_df = pd.DataFrame(results)
        print("Resulting df: \n", result_df.head(1000))
        print("Resulting ecl df \n", metadata.head(100))

        data_loader.save_dataset(result_df, os.path.join(config.OUTPUT_DIR, "tags.csv"))
        data_loader.save_dataset(metadata, output_path)

        if postprocess:
            postprocess_dense_csv(output_path)



    finally:
        close_connection(conn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense ECL + SEC financial tags dataset.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
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
