"""Build an ECL metadata dataset enriched with SEC submissions fields.

This helper is designed to be:
1) executable as a standalone script, and
2) importable so other dataset builders (dense/sparse) can reuse its functions.

Enriched metadata columns include:
- accessionNumber
- reportDateIndex
- form
- primaryDocument
- isXBRL
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config
import re


DEFAULT_TEXT_COLUMNS_TO_DROP = ("opinion_text", "item_7")


def clean_cik(cik_value: Any) -> str:
    cik_str = str(cik_value).split(".")[0].strip()
    return cik_str.zfill(10)


def extract_year_from_filename(filename: str) -> Optional[int]:
    match = re.search(r"-(\d{2})-", filename)
    return int("20" + match.group(1)) if match else None


def extract_accession_number_index(filename: str) -> str:
    return filename[-25:-5]



def _load_json_if_exists(file_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def load_submissions_for_cik(cik_str: str, submissions_dir: str) -> Tuple[Optional[Dict[str, Any]], Iterable[Dict[str, Any]]]:
    """Load SEC submissions JSON files for a CIK (main + split files)."""
    main_file = os.path.join(submissions_dir, f"CIK{cik_str}.json")
    main_data = _load_json_if_exists(main_file)

    split_pattern = os.path.join(submissions_dir, f"CIK{cik_str}-submissions-*")
    split_data = []
    for split_file in glob.glob(split_pattern):
        split_json = _load_json_if_exists(split_file)
        if split_json is not None:
            split_data.append(split_json)

    return main_data, split_data


def _match_accession(metadata_source: Dict[str, Any], accession_number: str) -> Optional[Dict[str, Any]]:
    accession_numbers = metadata_source.get("accessionNumber", [])
    for report_index, existing_acc in enumerate(accession_numbers):
        if existing_acc == accession_number:
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
    """Resolve submissions metadata for a single (cik, accessionNumber) pair."""
    cik_str = clean_cik(cik_value)

    if cik_str not in cik_cache:
        cik_cache[cik_str] = load_submissions_for_cik(cik_str, submissions_dir)

    main_data, split_data = cik_cache[cik_str]

    if main_data is not None:
        recent = main_data.get("filings", {}).get("recent", {})
        match = _match_accession(recent, accession_number)
        if match:
            return match

    for split in split_data:
        match = _match_accession(split, accession_number)
        if match:
            return match

    return {
        "form": None,
        "primaryDocument": None,
        "isXBRL": None,
        "reportDateIndex": None,
    }


def build_metadata_row(
    row: Dict[str, Any],
    cik_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Iterable[Dict[str, Any]]]],
    submissions_dir: str,
    drop_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS_TO_DROP,
) -> Dict[str, Any]:
    """Create one enriched metadata row from an ECL source row."""
    output_row = dict(row)

    for column in drop_columns:
        output_row.pop(column, None)

    output_row["year"] = extract_year_from_filename(str(output_row.get("filename", "")))
    output_row["accessionNumber"] = extract_accession_number_index(str(output_row.get("filename", "")))

    submissions_metadata = find_submissions_metadata(
        cik_value=output_row.get("cik", ""),
        accession_number=output_row["accessionNumber"],
        cik_cache=cik_cache,
        submissions_dir=submissions_dir,
    )
    output_row.update(submissions_metadata)

    return output_row


def create_metadata_dataset(
    input_path: str,
    output_path: str,
    submissions_dir: str = config.SEC_SUBMISSIONS_DIR,
    max_rows: Optional[int] = None,
    min_year: Optional[int] = 2000,
    drop_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS_TO_DROP,
) -> int:
    """Create metadata-enriched JSONL dataset.

    Returns the number of rows written.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cik_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Iterable[Dict[str, Any]]]] = {}
    rows_written = 0

    with open(input_path, "r", encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as destination:

        for line_number, line in enumerate(source, start=1):
            print(f"\rCurrent row: {rows_written}", end='')
            if max_rows is not None and rows_written >= max_rows:
                break

            if not line.strip():
                continue

            raw_row = json.loads(line)
            metadata_row = build_metadata_row(
                row=raw_row,
                cik_cache=cik_cache,
                submissions_dir=submissions_dir,
                drop_columns=drop_columns,
            )

            if min_year is not None and metadata_row.get("year") is not None and metadata_row["year"] < min_year:
                continue

            destination.write(json.dumps(metadata_row) + "\n")
            rows_written += 1

            # if rows_written % 100 == 0:
            #     print(f"Processed {rows_written:,} rows (source line {line_number:,})", flush=True)

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create initial ECL metadata dataset from submissions files.")
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
        help="Optional debug limit: stop and save after X output rows.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2004,
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
