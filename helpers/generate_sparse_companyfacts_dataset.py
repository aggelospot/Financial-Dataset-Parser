"""Generate a sparse ECL + SEC companyfacts CSV dataset."""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config

DEFAULT_TEXT_COLUMNS_TO_DROP = ("opinion_text", "item_7")
ACCESSION_NUMBER_COLUMN = "accession_number"
BASE_OUTPUT_COLUMNS = ("cik_adsh", "label", "year")


def clean_cik(cik_value: Any) -> str:
    """Normalize CIK as 10-digit zero-padded string."""
    cik_str = str(cik_value).split(".")[0].strip()
    return cik_str.zfill(10)


def iter_data_points(fact_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield all data points across all units for a given SEC concept."""
    for unit_values in fact_obj.get("units", {}).values():
        for point in unit_values:
            yield point


def match_value_for_accession(fact_obj: Dict[str, Any], accession_number: str) -> Any:
    """Return the first 10-K concept value reported by the target accession."""
    for point in iter_data_points(fact_obj):
        form_type = str(point.get("form", ""))
        if "10-K" not in form_type:
            continue

        if str(point.get("accn", "")).strip() == accession_number:
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


def get_input_columns(input_path: str, drop_columns: Iterable[str]) -> Set[str]:
    """Read first non-empty input row to identify original metadata columns."""
    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            for col in drop_columns:
                row.pop(col, None)
            return set(row.keys())
    return set()


def extract_numeric_facts_for_row(
    row: Dict[str, Any],
    sec_cache: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, float]:
    """Extract SEC numeric concepts for one input row."""
    accession_number = str(row.get(ACCESSION_NUMBER_COLUMN, "")).strip()
    if not accession_number:
        return {}

    cik_cleaned = clean_cik(row.get("cik", ""))
    sec_data = load_companyfacts(cik_cleaned, sec_cache)
    if not sec_data:
        return {}

    facts = sec_data.get("facts", {})
    all_facts = {
        **facts.get("us-gaap", {}),
        **facts.get("dei", {}),
    }

    numeric_facts: Dict[str, float] = {}
    for concept_name, fact_obj in all_facts.items():
        matched_value = match_value_for_accession(fact_obj, accession_number)
        if matched_value is None:
            continue

        numeric_value = pd.to_numeric(matched_value, errors="coerce")
        if pd.notna(numeric_value):
            numeric_facts[concept_name] = float(numeric_value)

    return numeric_facts


def determine_output_columns(
    input_path: str,
    max_rows: Optional[int],
    drop_columns: Iterable[str],
) -> list[str]:
    """First pass over input to determine CSV columns without materializing rows."""
    input_columns = get_input_columns(input_path=input_path, drop_columns=drop_columns)
    required_input_columns = {"cik", ACCESSION_NUMBER_COLUMN, *BASE_OUTPUT_COLUMNS}
    missing_input_columns = sorted(required_input_columns.difference(input_columns))
    if missing_input_columns:
        raise ValueError(
            f"Input metadata is missing required column(s): {missing_input_columns}. "
            "Regenerate metadata with helpers/generate_metadata_dataset.py before building the sparse dataset."
        )
    selected_base_columns = list(BASE_OUTPUT_COLUMNS)

    discovered_numeric_columns: Set[str] = set()
    sec_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    processed_rows = 0

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            print(f"\rDiscovering columns - current row: {line_number}", end="")
            if max_rows is not None and processed_rows >= max_rows:
                break
            if not line.strip():
                continue

            row = json.loads(line)
            for column in drop_columns:
                row.pop(column, None)

            numeric_facts = extract_numeric_facts_for_row(row=row, sec_cache=sec_cache)
            discovered_numeric_columns.update(numeric_facts.keys())
            processed_rows += 1

    print("")
    return selected_base_columns + sorted(discovered_numeric_columns)


def build_sparse_dataset_csv(
    input_path: str,
    output_path: str,
    max_rows: Optional[int],
    drop_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS_TO_DROP,
) -> None:
    """Two-pass streaming CSV writer for sparse companyfacts without loading full dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_columns = determine_output_columns(
        input_path=input_path,
        max_rows=max_rows,
        drop_columns=drop_columns,
    )

    sec_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    processed_rows = 0

    with open(input_path, "r", encoding="utf-8") as input_file, open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_columns)
        writer.writeheader()

        for line_number, line in enumerate(input_file, start=1):
            print(f"\rWriting CSV - current row: {line_number}", end="")
            if max_rows is not None and processed_rows >= max_rows:
                break
            if not line.strip():
                continue

            row = json.loads(line)
            for column in drop_columns:
                row.pop(column, None)

            numeric_facts = extract_numeric_facts_for_row(row=row, sec_cache=sec_cache)

            output_row: Dict[str, Any] = {}
            for col in BASE_OUTPUT_COLUMNS:
                if col in output_columns:
                    output_row[col] = row.get(col)

            output_row.update({key: format(value, ".15g") for key, value in numeric_facts.items()})
            writer.writerow(output_row)
            processed_rows += 1

    print(
        f"\nDone. Wrote {processed_rows:,} rows to {output_path}. "
        f"Rows limit: {max_rows if max_rows is not None else 'all'}"
    )


def resolve_input_path() -> str:
    if hasattr(config, "COMPANYFACTS_METADATA_PATH") and config.COMPANYFACTS_METADATA_PATH:
        return config.COMPANYFACTS_METADATA_PATH

    if os.path.isabs(config.ECL_FILE_PATH):
        return config.ECL_FILE_PATH
    return os.path.join(config.DATA_DIR, config.ECL_FILE_PATH)


def resolve_output_path(output_name: str) -> str:
    """Resolve output csv path under OUTPUT_DIR from any user-provided name."""
    base_name = os.path.basename(output_name)
    stem, ext = os.path.splitext(base_name)
    base_stem = stem if ext == ".csv" else base_name
    return os.path.join(config.OUTPUT_DIR, f"{base_stem}.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sparse companyfacts CSV dataset.")
    parser.add_argument(
        "--output",
        default="ecl_companyfacts_sparse.csv",
        help="Output CSV file name. File is written under outputs/.",
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

    input_path = resolve_input_path()
    output_csv_path = resolve_output_path(args.output)

    build_sparse_dataset_csv(
        input_path=input_path,
        output_path=output_csv_path,
        max_rows=args.max_rows,
    )
