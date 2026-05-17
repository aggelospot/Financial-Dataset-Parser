"""Generate the dense ECL + SEC financial tags dataset.

This helper preserves the existing dense methodology used in ``main.py`` while
streaming metadata rows directly to CSV.
"""

import argparse
import csv
import json
import os
import sys
from typing import List
from decimal import Decimal

import pandas as pd

from db_connection import retrieve_statement_taxonomies_by_accession_number3, close_connection, create_connection
from tools.utils import match_concept_in_section


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config


DEFAULT_INPUT_PATH = getattr(config, "COMPANYFACTS_METADATA_PATH", config.ECL_METADATA_NOTEXT_PATH)
DEFAULT_OUTPUT_PATH = getattr(config, "COMPANYFACTS_DENSE_PATH", os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv"))
FINANCIAL_SECTIONS = (
    ("IncomeStatement", "IS", [4]),
    ("BalanceSheet", "BS", 0),
    ("CashFlow", "CF", [4]),
    ("StatementOfStockholdersEquity", "EQ", [0]),
)


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _configured_financial_columns() -> List[str]:
    with open(config.XBRL_MAPPING_PATH, "r", encoding="utf-8") as file:
        xbrl_mapping = json.load(file)

    return _configured_financial_columns_from_mapping(xbrl_mapping)


def _configured_financial_columns_from_mapping(xbrl_mapping: dict) -> List[str]:
    columns: List[str] = []
    for section, _, _ in FINANCIAL_SECTIONS:
        for key in xbrl_mapping.get(section, {}).keys():
            if key not in columns:
                columns.append(key)
    return columns


def _iter_metadata_rows(input_path: str, max_rows: int | None = None):
    _, extension = os.path.splitext(input_path)
    extension = extension.lower()
    rows_seen = 0

    if extension == ".csv":
        with open(input_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if max_rows is not None and rows_seen >= max_rows:
                    break
                rows_seen += 1
                yield row
        return

    if extension == ".json":
        with open(input_path, "r", encoding="utf-8") as file:
            for line in file:
                if max_rows is not None and rows_seen >= max_rows:
                    break
                if not line.strip():
                    continue
                rows_seen += 1
                yield json.loads(line)
        return

    raise ValueError(f"Unsupported file extension: {extension}")


def _is_xbrl_row(row: dict) -> bool:
    return str(row.get("isXBRL", 1)).strip().lower() not in {"0", "false", "nan", "none", ""}


def _csv_value(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return format(value, ".15g")
    if isinstance(value, float):
        return f"{value:.15g}"
    return value


def _write_row(writer: csv.DictWriter, row: dict) -> None:
    writer.writerow({key: _csv_value(value) for key, value in row.items()})


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

def create_dense_dataset(
    input_path: str,
    output_path: str,
    postprocess: bool = True,
    max_rows: int | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If output already exists, do postprocessing only; do not rebuild raw output.
    if os.path.isfile(output_path):
        print(f"Output already exists at {output_path}. Running postprocessing only.")
        if postprocess:
            postprocess_dense_csv(output_path)
        return

    conn = None
    try:
        conn = create_connection()

        with open(config.XBRL_MAPPING_PATH, 'r', encoding='utf-8') as file:
            xbrl_mapping = json.load(file)

        tag_list = _configured_financial_columns_from_mapping(xbrl_mapping)
        dense_fieldnames = ["accessionNumber", "label", *tag_list] if postprocess else None
        tags_fieldnames = ["accession_number", "isXBRL", *tag_list]
        tags_output_path = os.path.join(config.OUTPUT_DIR, "tags.csv")
        rows_written = 0
        printed_missing_role_warning = False

        with open(output_path, "w", encoding="utf-8", newline="") as dense_file, open(
            tags_output_path, "w", encoding="utf-8", newline=""
        ) as tags_file:
            dense_writer = None
            tags_writer = csv.DictWriter(tags_file, fieldnames=tags_fieldnames, extrasaction="ignore")
            tags_writer.writeheader()

            for row in _iter_metadata_rows(input_path, max_rows=max_rows):
                if dense_fieldnames is None:
                    dense_fieldnames = [*row.keys(), *[tag for tag in tag_list if tag not in row]]
                    dense_writer = csv.DictWriter(dense_file, fieldnames=dense_fieldnames, extrasaction="ignore")
                    dense_writer.writeheader()
                elif dense_writer is None:
                    dense_writer = csv.DictWriter(dense_file, fieldnames=dense_fieldnames, extrasaction="ignore")
                    dense_writer.writeheader()

                print(f"\rCurrent row: {rows_written}", end="")

                if "filing_role" not in row and not printed_missing_role_warning:
                    print("\nMetadata has no filing_role column; SEC numeric extraction will be skipped for all rows.")
                    printed_missing_role_warning = True

                if postprocess:
                    output_row = {"accessionNumber": row.get("accessionNumber"), "label": row.get("label")}
                else:
                    output_row = dict(row)
                output_row.update({tag: None for tag in tag_list})

                matched_items = {}
                filing_cache = {}
                is_primary_registrant = str(row.get("filing_role", "")).strip().lower() == "primary_registrant"

                if _is_xbrl_row(row) and is_primary_registrant:
                    adsh = row["accessionNumber"]

                    def cache_statement(stmt_code, quarter_spec):
                        tags, labels, vals = retrieve_statement_taxonomies_by_accession_number3(
                            conn,
                            adsh, row["cik"], row.get("filing_role"), stmt_code, quarter_spec)
                        filing_cache.update(dict(zip(tags, vals)))
                        return tags, labels

                    for section_name, stmt_code, quarter_spec in FINANCIAL_SECTIONS:
                        statement_tags, statement_labels = cache_statement(stmt_code, quarter_spec)
                        matched_items.update(match_concept_in_section(
                            xbrl_mapping[section_name],
                            statement_tags,
                            statement_labels,
                        ))

                    for concept_name, sec_tag in matched_items.items():
                        if concept_name in tag_list:
                            output_row[concept_name] = filing_cache.get(sec_tag)

                if _is_xbrl_row(row):
                    tags_writer.writerow({
                        "accession_number": row.get("accessionNumber"),
                        "isXBRL": 1,
                        **matched_items,
                    })

                _write_row(dense_writer, output_row)
                rows_written += 1

        print(f"\nFinished writing {rows_written:,} rows to {output_path}")



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
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of metadata rows to process for quick test runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dense_dataset(
        input_path=args.input,
        output_path=args.output,
        postprocess=args.postprocess,
        max_rows=args.max_rows,
    )
