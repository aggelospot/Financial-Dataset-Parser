"""Generate a sparse ECL + SEC companyfacts dataset."""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config

DEFAULT_TEXT_COLUMNS_TO_DROP = ("opinion_text", "item_7")
KEEP_COLUMNS_POSTPROCESS = ("accessionNumber", "label")


def clean_cik(cik_value: Any) -> str:
    """Normalize CIK as 10-digit zero-padded string."""
    cik_str = str(cik_value).split(".")[0].strip()
    return cik_str.zfill(10)


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract filing year from ECL filename pattern like *-YY-*."""
    import re

    match = re.search(r"-(\d{2})-", filename)
    return int("20" + match.group(1)) if match else None


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


def build_sparse_dataset(
    input_path: str,
    output_path: str,
    max_rows: Optional[int],
    drop_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS_TO_DROP,
) -> Set[str]:
    """Stream ECL rows, enrich them with SEC facts, and write sparse JSONL output."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sec_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    processed_rows = 0
    input_columns = get_input_columns(input_path=input_path, drop_columns=drop_columns)

    with open(input_path, "r", encoding="utf-8") as input_file, open(output_path, "w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(input_file, start=1):
            print(f"\rCurrent row: {line_number}", end="")
            if max_rows is not None and processed_rows >= max_rows:
                break

            if not line.strip():
                continue

            row = json.loads(line)

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

    print(
        f"\nDone. Wrote {processed_rows:,} rows to {output_path}. "
        f"Rows limit: {max_rows if max_rows is not None else 'all'}"
    )
    return input_columns


def postprocess_json_dataset(json_path: str, input_columns: Set[str]) -> None:
    """Keep only numeric added columns + accessionNumber + label in JSONL output."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON output not found: {json_path}")

    df = pd.read_json(json_path, lines=True)
    added_columns = [col for col in df.columns if col not in input_columns]
    numeric_added = [col for col in added_columns if pd.api.types.is_numeric_dtype(df[col])]

    keep_columns = [col for col in KEEP_COLUMNS_POSTPROCESS if col in df.columns] + numeric_added
    keep_columns = list(dict.fromkeys(keep_columns))

    processed_df = df[keep_columns] if keep_columns else pd.DataFrame()
    processed_df.to_json(json_path, orient="records", lines=True)
    print(f"Post-process complete for JSON. Kept columns: {len(keep_columns)}")


def convert_json_to_csv(json_path: str, csv_path: str) -> None:
    """Convert line-delimited JSON output to CSV."""
    df = pd.read_json(json_path, lines=True)
    df.to_csv(csv_path, index=False)
    print(f"Converted JSON to CSV: {csv_path}")


def resolve_input_path() -> str:
    if hasattr(config, "COMPANYFACTS_METADATA_PATH") and config.COMPANYFACTS_METADATA_PATH:
        return config.COMPANYFACTS_METADATA_PATH

    if os.path.isabs(config.ECL_FILE_PATH):
        return config.ECL_FILE_PATH
    return os.path.join(config.DATA_DIR, config.ECL_FILE_PATH)


def resolve_output_paths(output_name: str) -> Tuple[str, str]:
    """Resolve output json/csv paths under OUTPUT_DIR from any user-provided name."""
    base_name = os.path.basename(output_name)
    stem, ext = os.path.splitext(base_name)
    base_stem = stem if ext in (".json", ".csv") else base_name

    json_path = os.path.join(config.OUTPUT_DIR, f"{base_stem}.json")
    csv_path = os.path.join(config.OUTPUT_DIR, f"{base_stem}.csv")
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sparse companyfacts dataset.")
    parser.add_argument(
        "--output",
        default="ecl_companyfacts_sparse.json",
        help="Output file name (base). Files are written under outputs/.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for test runs.",
    )
    parser.add_argument(
        "--post-process",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Whether to keep only numeric added columns + accessionNumber + label (default: true).",
    )
    parser.add_argument(
        "--output-file-type",
        choices=["json", "csv"],
        default="json",
        help="Final file type to emit. If csv is selected, JSON is converted to CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = resolve_input_path()
    output_json_path, output_csv_path = resolve_output_paths(args.output)

    # If JSON already exists, skip generation and only postprocess/convert as requested.
    if os.path.exists(output_json_path):
        print(f"Found existing JSON output at {output_json_path}; skipping generation.")
        input_columns = get_input_columns(input_path=input_path, drop_columns=DEFAULT_TEXT_COLUMNS_TO_DROP)
    else:
        input_columns = build_sparse_dataset(
            input_path=input_path,
            output_path=output_json_path,
            max_rows=args.max_rows,
        )

    if args.post_process:
        postprocess_json_dataset(json_path=output_json_path, input_columns=input_columns)

    if args.output_file_type == "csv":
        convert_json_to_csv(json_path=output_json_path, csv_path=output_csv_path)
