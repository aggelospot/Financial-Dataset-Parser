"""Generate the dense ECL + SEC financial tags dataset.

This helper preserves the existing dense methodology used in ``main.py`` by:
1) loading the metadata dataset without text fields,
2) calling ``retrieve_sec_tags_and_values`` (which performs concept matching and
   value retrieval), and
3) saving the resulting dense dataset as CSV.
"""

import argparse
import os
import sys

import pandas as pd

from db_connection import retrieve_statement_taxonomies_by_accession_number2, close_connection, create_connection
# from financials_parser import match_concept_in_section
from tools.utils import match_concept_in_section


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sec_data_parser import retrieve_sec_tags_and_values
from tools import config
from tools.data_loader import DataLoader
import json


DEFAULT_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv")


def create_dense_dataset2(input_path: str, output_path: str) -> None:
    try:

        data_loader = DataLoader()
        conn = create_connection()

        ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)
        with open(config.XBRL_MAPPING_PATH, 'r') as file:
            xbrl_mapping = json.load(file)


        # METHOD 2: ATTEMPT TO STANDARDIZE THE TAGS
        tag_list = []
        for section in ["IncomeStatement", "BalanceSheet", "CashFlow", "StatementOfStockholdersEquity"]:
            for key, value in xbrl_mapping[section].items():
                if key not in tag_list:  # Avoid duplicates
                    tag_list.append(key)
        for tag in tag_list:
            ecl[tag] = None

        results = []
        filing_cache = {}  # {accession_number: {tag: value}}
        for idx, row in ecl.iterrows():
            print(f"\rCurrent row: {idx}/{len(ecl)}", end='')
            # if idx > 300: break
            if row['isXBRL'] == 0:
                continue  # skip non-XBRL filings

            adsh = row['accessionNumber']
            filing_cache[adsh] = {}  # start fresh for this filing

            # ------------------------------------------------
            # helper: store (tag → value) for later reuse
            # ------------------------------------------------
            def cache_statement(stmt_code, quarter_spec):
                tags, labels, vals = retrieve_statement_taxonomies_by_accession_number2(
                    conn,
                    adsh, stmt_code, quarter_spec)
                filing_cache[adsh].update(dict(zip(tags, vals)))
                return tags, labels  # keep old behaviour where needed

            # ---------------- Income Statement ----------------------
            is_tags, is_labels = cache_statement('IS', [4])
            matched_is_items = match_concept_in_section(
                xbrl_mapping['IncomeStatement'],
                is_tags, is_labels)

            # print(matched_is_items)
            # print("current cache", filing_cache[adsh])

            #  Balance Sheet
            bs_tags, bs_labels = cache_statement('BS', 0)
            matched_bs_items = match_concept_in_section(
                xbrl_mapping['BalanceSheet'],
                bs_tags, bs_labels)

            #  Cash-flow
            cf_tags, cf_labels = cache_statement('CF', [4])
            matched_cf_items = match_concept_in_section(
                xbrl_mapping['CashFlow'],
                cf_tags, cf_labels)

            #  Stockholders' Equity
            eq_tags, eq_labels = cache_statement('EQ', [0])
            matched_eq_items = match_concept_in_section(
                xbrl_mapping['StatementOfStockholdersEquity'],
                eq_tags, eq_labels)

            # assemble row using cached values
            metadata_row = {
                "accession_number": adsh,
                "isXBRL": 1,
                **matched_is_items,
                **matched_bs_items,
                **matched_cf_items,
                **matched_eq_items,
            }

            for concept_name, sec_tag in metadata_row.items():
                if concept_name in tag_list:  # skip meta fields
                    val = filing_cache[adsh].get(sec_tag)
                    ecl.at[idx, concept_name] = val

            results.append(metadata_row)

        result_df = pd.DataFrame(results)
        print("Resulting df: \n", result_df.head(1000))
        print("Resulting ecl df \n", ecl.head(100))
        data_loader.save_dataset(result_df, os.path.join(config.OUTPUT_DIR, "tags.csv"))
        data_loader.save_dataset(ecl, os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv"))



    finally:
        # 4. Close the connection regardless of success/failure
        close_connection(conn)

def create_dense_dataset(input_path: str, output_path: str) -> None:
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dense_dataset2(input_path=args.input, output_path=args.output)
