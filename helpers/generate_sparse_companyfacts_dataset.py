"""Generate a sparse ECL + SEC companyfacts dataset.

This script follows the historical pipeline documented in
`docs/sparse_dataset_pipeline.md`:
- Read ECL records.
- Remove large text fields early to keep memory usage low.
- Enrich each row with matching SEC companyfacts concepts from 10-K filings
  for the same fiscal year.
- Write line-delimited JSON output incrementally.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config


def clean_cik(cik_value: Any) -> str:
    """Normalize CIK as 10-digit zero-padded string."""
    cik_str = str(cik_value).split(".")[0].strip()
    return cik_str.zfill(10)


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract filing year from ECL filename pattern like *-YY-*."""
    import re

    match = re.search(r"-(\d{2})-", filename)
    return int("20" + match.group(1)) if match else None


DEFAULT_TEXT_COLUMNS_TO_DROP = ("opinion_text", "item_7")


def extract_fiscal_year(row: Dict[str, Any]) -> Optional[str]:
    """Extract the target fiscal year for matching SEC facts."""
    cik_year = row.get("cik_year")
    if cik_year:
        return str(cik_year).split("__")[-1]

    filename = row.get("filename")
    if filename:
        year = extract_year_from_filename(str(filename))
        if year is not None:
            return str(year)

    return None


def iter_data_points(fact_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield all data points across all units for a given SEC concept."""
    for unit_values in fact_obj.get("units", {}).values():
        for point in unit_values:
            yield point


def match_value_for_year(fact_obj: Dict[str, Any], fiscal_year: str) -> Any:
    """Return the first 10-K concept value matching the target fiscal year."""
    for point in iter_data_points(fact_obj):
        form_type = str(point.get("form", ""))
        if "10-K" not in form_type:
            continue

        if str(point.get("fy", "")) == fiscal_year:
            return point.get("val")

    return None


def load_companyfacts(cik_cleaned: str, cache: Dict[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Load a local SEC companyfacts file once per CIK and cache it."""
    if cik_cleaned in cache:
        return cache[cik_cleaned]

    file_name = f"CIK{cik_cleaned}.json"
    file_path = os.path.join(config.SEC_DATA_DIR, file_name)

    if not os.path.exists(file_path):
        cache[cik_cleaned] = None
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            cache[cik_cleaned] = json.load(handle)
    except json.JSONDecodeError:
        cache[cik_cleaned] = None

    return cache[cik_cleaned]


def build_sparse_dataset(
    input_path: str,
    output_path: str,
    max_rows: Optional[int] = None,
    drop_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS_TO_DROP,
) -> None:
    """Stream ECL rows, enrich them with SEC facts, and write sparse JSONL output."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sec_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    processed_rows = 0

    with open(input_path, "r", encoding="utf-8") as input_file, open(output_path, "w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if max_rows is not None and processed_rows >= max_rows:
                break

            if not line.strip():
                continue

            row = json.loads(line)

            # Drop large text columns immediately to minimize memory pressure.
            for column in drop_columns:
                row.pop(column, None)

            fiscal_year = extract_fiscal_year(row)
            if fiscal_year is None:
                output_file.write(json.dumps(row) + "\n")
                processed_rows += 1
                continue

            cik_cleaned = clean_cik(row.get("cik", ""))
            sec_data = load_companyfacts(cik_cleaned, sec_cache)

            if sec_data:
                facts = sec_data.get("facts", {})
                all_facts = {
                    **facts.get("us-gaap", {}),
                    **facts.get("dei", {}),
                }

                for concept_name, fact_obj in all_facts.items():
                    matched_value = match_value_for_year(fact_obj, fiscal_year)
                    if matched_value is not None:
                        row[concept_name] = matched_value

            output_file.write(json.dumps(row) + "\n")
            processed_rows += 1

            if processed_rows % 100 == 0:
                print(f"Processed {processed_rows:,} rows (last input line: {line_number:,})", flush=True)

    print(
        f"Done. Wrote {processed_rows:,} rows to {output_path}. "
        f"Rows limit: {max_rows if max_rows is not None else 'all'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sparse ECL + companyfacts dataset.")
    parser.add_argument(
        "--input",
        default=config.ECL_FILE_PATH,
        help="Path to source ECL JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(config.OUTPUT_DIR, "ecl_companyfacts_sparse.json"),
        help="Path to output sparse JSONL file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for test runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_sparse_dataset(input_path=args.input, output_path=args.output, max_rows=args.max_rows)
