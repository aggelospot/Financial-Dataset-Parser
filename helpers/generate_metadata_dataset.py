"""Build an ECL metadata dataset enriched with SEC submissions and filing-role fields.

This helper is designed to be executable as a standalone script and importable
by other dataset builders.

Enriched metadata columns include:
- accession_number
- cik_adsh
- reportDateIndex
- form
- primaryDocument
- isXBRL
- filing_role
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db_connection import close_connection, create_connection, retrieve_filing_role_by_accession_and_cik
from tools import config


DEFAULT_TEXT_COLUMNS_TO_DROP = ("opinion_text", "item_7")
ACCESSION_NUMBER_COLUMN = "accession_number"
CIK_ADSH_COLUMN = "cik_adsh"
FILING_ROLE_COLUMN = "filing_role"


def clean_cik(cik_value: Any) -> str:
    return str(cik_value).split(".")[0].strip().zfill(10)


def extract_year_from_filing_date(filing_date: Any) -> Optional[int]:
    match = re.match(r"^(\d{4})-\d{2}-\d{2}$", str(filing_date).strip())
    return int(match.group(1)) if match else None


def extract_accession_number_index(filename: str) -> str:
    return filename[-25:-5]


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_submissions_for_cik(
    cik_str: str,
    submissions_dir: str,
) -> Tuple[Optional[Dict[str, Any]], Iterable[Dict[str, Any]]]:
    main_data = load_json(os.path.join(submissions_dir, f"CIK{cik_str}.json"))
    split_data = [
        split_json
        for split_file in glob.glob(os.path.join(submissions_dir, f"CIK{cik_str}-submissions-*"))
        if (split_json := load_json(split_file)) is not None
    ]
    return main_data, split_data


def match_accession(metadata_source: Dict[str, Any], accession_number: str) -> Optional[Dict[str, Any]]:
    for report_index, existing_acc in enumerate(metadata_source.get("accessionNumber", [])):
        if existing_acc != accession_number:
            continue

        forms = metadata_source.get("form", [])
        docs = metadata_source.get("primaryDocument", [])
        is_xbrl = metadata_source.get("isXBRL", [])

        return {
            "form": forms[report_index] if report_index < len(forms) else None,
            "primaryDocument": docs[report_index] if report_index < len(docs) else None,
            "isXBRL": is_xbrl[report_index] if report_index < len(is_xbrl) else None,
            "reportDateIndex": report_index,
        }

    return None


def find_submissions_metadata(
    cik_value: Any,
    accession_number: str,
    cik_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Iterable[Dict[str, Any]]]],
    submissions_dir: str,
) -> Dict[str, Any]:
    cik_str = clean_cik(cik_value)
    if cik_str not in cik_cache:
        cik_cache[cik_str] = load_submissions_for_cik(cik_str, submissions_dir)

    main_data, split_data = cik_cache[cik_str]
    if main_data is not None:
        match = match_accession(main_data.get("filings", {}).get("recent", {}), accession_number)
        if match is not None:
            return match

    for split in split_data:
        match = match_accession(split, accession_number)
        if match is not None:
            return match

    return {
        "form": None,
        "primaryDocument": None,
        "isXBRL": None,
        "reportDateIndex": None,
    }


def find_filing_role(
    conn,
    accession_number: str,
    cik_value: Any,
    filing_role_cache: Dict[Tuple[str, int], str],
) -> str:
    metadata_cik = int(str(cik_value).split(".")[0].strip())
    cache_key = (accession_number, metadata_cik)

    if cache_key not in filing_role_cache:
        filing_role_cache[cache_key] = retrieve_filing_role_by_accession_and_cik(
            connection=conn,
            accession_number=accession_number,
            metadata_cik=metadata_cik,
        )

    return filing_role_cache[cache_key]


def build_cik_adsh(cik_value: Any, accession_number: Any) -> str:
    cik_part = str(cik_value).split(".")[0].strip()
    accession_part = str(accession_number).strip()
    return f"{cik_part}_{accession_part}"


def enrich_metadata_row(
    row: Dict[str, Any],
    cik_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Iterable[Dict[str, Any]]]],
    filing_role_cache: Dict[Tuple[str, int], str],
    submissions_dir: str,
    conn,
    drop_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS_TO_DROP,
) -> Dict[str, Any]:
    metadata_row = dict(row)

    for column in drop_columns:
        metadata_row.pop(column, None)

    metadata_row.pop("accessionNumber", None)
    metadata_row["year"] = extract_year_from_filing_date(metadata_row.get("filing_date"))
    metadata_row["cik"] = int(metadata_row["cik"])
    metadata_row["gvkey"] = int(metadata_row["gvkey"])

    accession_number = extract_accession_number_index(str(metadata_row.get("filename", "")))
    metadata_row[ACCESSION_NUMBER_COLUMN] = accession_number
    metadata_row[CIK_ADSH_COLUMN] = build_cik_adsh(
        cik_value=metadata_row["cik"],
        accession_number=accession_number,
    )

    metadata_row.update(
        find_submissions_metadata(
            cik_value=metadata_row["cik"],
            accession_number=accession_number,
            cik_cache=cik_cache,
            submissions_dir=submissions_dir,
        )
    )
    metadata_row[FILING_ROLE_COLUMN] = find_filing_role(
        conn=conn,
        accession_number=accession_number,
        cik_value=metadata_row["cik"],
        filing_role_cache=filing_role_cache,
    )

    return metadata_row


def create_metadata_dataset(
    input_path: str,
    output_path: str,
    submissions_dir: str = config.SEC_SUBMISSIONS_DIR,
    max_rows: Optional[int] = None,
    min_year: Optional[int] = 2000,
    drop_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS_TO_DROP,
) -> int:
    """Create metadata-enriched JSONL dataset from scratch."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cik_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Iterable[Dict[str, Any]]]] = {}
    filing_role_cache: Dict[Tuple[str, int], str] = {}
    rows_written = 0
    conn = create_connection()

    try:
        with open(input_path, "r", encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, start=1):
                if max_rows is not None and rows_written >= max_rows:
                    break

                if not line.strip():
                    continue

                metadata_row = enrich_metadata_row(
                    row=json.loads(line),
                    cik_cache=cik_cache,
                    filing_role_cache=filing_role_cache,
                    submissions_dir=submissions_dir,
                    conn=conn,
                    drop_columns=drop_columns,
                )

                if min_year is not None and metadata_row.get("year") is not None and metadata_row["year"] < min_year:
                    continue

                destination.write(json.dumps(metadata_row) + "\n")
                rows_written += 1
                print(f"\rRows written: {rows_written:,} | source line: {line_number:,}", end="")
    finally:
        close_connection(conn)

    print("")
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ECL metadata dataset from scratch.")
    parser.add_argument("--input", default=config.ECL_FILE_PATH, help="Path to source ECL JSONL file.")
    parser.add_argument(
        "--output",
        default=config.COMPANYFACTS_METADATA_PATH,
        help="Path for metadata-enriched JSONL output.",
    )
    parser.add_argument(
        "--submissions-dir",
        default=config.SEC_SUBMISSIONS_DIR,
        help="Directory containing SEC submissions JSON files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug limit: stop after X output rows.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2008,
        help="Optional year filter. Set to a negative value to disable filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_year = None if args.min_year < 0 else args.min_year

    rows_written = create_metadata_dataset(
        input_path=args.input,
        output_path=args.output,
        submissions_dir=args.submissions_dir,
        max_rows=args.max_rows,
        min_year=min_year,
    )

    print(f"Done. Wrote {rows_written:,} rows to {args.output}")


if __name__ == "__main__":
    main()
