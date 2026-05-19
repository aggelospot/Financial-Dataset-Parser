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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config


DEFAULT_INPUT_PATH = getattr(config, "COMPANYFACTS_METADATA_PATH", config.ECL_METADATA_NOTEXT_PATH)
DEFAULT_OUTPUT_PATH = getattr(config, "COMPANYFACTS_DENSE_PATH", os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv"))
ACCESSION_NUMBER_COLUMN = "accession_number"
CIK_ADSH_COLUMN = "cik_adsh"
BASE_OUTPUT_COLUMNS = (CIK_ADSH_COLUMN, "label", "year")
REQUIRED_METADATA_COLUMNS = ("cik", ACCESSION_NUMBER_COLUMN, CIK_ADSH_COLUMN, "label", "year", "filing_role", "isXBRL")
FINANCIAL_SECTIONS = (
    ("IncomeStatement", "IS", [4]),
    ("BalanceSheet", "BS", 0),
    ("CashFlow", "CF", [4]),
    ("StatementOfStockholdersEquity", "EQ", [0]),
)


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


def _ensure_required_metadata_columns(row: dict, row_number: int) -> None:
    missing_columns = sorted(column for column in REQUIRED_METADATA_COLUMNS if column not in row)
    if missing_columns:
        raise ValueError(
            f"Input metadata row {row_number} is missing required column(s): {missing_columns}. "
            "Regenerate metadata with helpers/generate_metadata_dataset.py before building the dense dataset."
        )


def create_dense_dataset(
    input_path: str,
    output_path: str,
    max_rows: int | None = None,
) -> None:
    from db_connection import close_connection, create_connection, retrieve_statement_taxonomies_by_accession_number3
    from tools.utils import match_concept_in_section

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    conn = None
    try:
        conn = create_connection()

        with open(config.XBRL_MAPPING_PATH, 'r', encoding='utf-8') as file:
            xbrl_mapping = json.load(file)

        tag_list = _configured_financial_columns_from_mapping(xbrl_mapping)
        dense_fieldnames = [*BASE_OUTPUT_COLUMNS, *tag_list]
        tags_fieldnames = [CIK_ADSH_COLUMN, "isXBRL", *tag_list]
        tags_output_path = os.path.join(config.OUTPUT_DIR, "tags.csv")
        rows_written = 0

        with open(output_path, "w", encoding="utf-8", newline="") as dense_file, open(
            tags_output_path, "w", encoding="utf-8", newline=""
        ) as tags_file:
            dense_writer = csv.DictWriter(dense_file, fieldnames=dense_fieldnames, extrasaction="ignore")
            dense_writer.writeheader()
            tags_writer = csv.DictWriter(tags_file, fieldnames=tags_fieldnames, extrasaction="ignore")
            tags_writer.writeheader()

            for row in _iter_metadata_rows(input_path, max_rows=max_rows):
                print(f"\rCurrent row: {rows_written}", end="")
                _ensure_required_metadata_columns(row, row_number=rows_written + 1)

                output_row = {column: row.get(column) for column in BASE_OUTPUT_COLUMNS}
                output_row.update({tag: None for tag in tag_list})

                matched_items = {}
                filing_cache = {}
                is_primary_registrant = str(row.get("filing_role", "")).strip().lower() == "primary_registrant"

                if _is_xbrl_row(row) and is_primary_registrant:
                    adsh = row[ACCESSION_NUMBER_COLUMN]

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
                        CIK_ADSH_COLUMN: row.get(CIK_ADSH_COLUMN),
                        "isXBRL": 1,
                        **matched_items,
                    })

                _write_row(dense_writer, output_row)
                rows_written += 1

        print(f"\nFinished writing {rows_written:,} rows to {output_path}")



    finally:
        if conn is not None:
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
        max_rows=args.max_rows,
    )
