"""Generate the text modality CSV for MDA + auditor opinion fields.

The helper streams the metadata dataset, resolves matching text rows from the
original ECL dataset using accession numbers extracted from ``filename``, and
writes a CSV with the columns:
- accessionNumber
- label
- item_7
- opinion_text
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterator, Optional, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config
from tools.utils import extract_accession_number_index

OUTPUT_COLUMNS = ("accessionNumber", "label", "item_7", "opinion_text")
JSON_CHUNK_SIZE = 65536


def _iter_jsonl_rows(file_path: str) -> Iterator[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {file_path}: {exc}") from exc



def _iter_json_array_rows(file_path: str) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()

    with open(file_path, "r", encoding="utf-8") as handle:
        first_non_ws = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first_non_ws = char
                break

        if first_non_ws != "[":
            raise ValueError(f"Expected a JSON array in {file_path}, found {first_non_ws!r} instead.")

        buffer = ""
        end_of_file = False

        while True:
            stripped = buffer.lstrip()
            if stripped != buffer:
                buffer = stripped

            if not buffer and not end_of_file:
                chunk = handle.read(JSON_CHUNK_SIZE)
                if chunk:
                    buffer += chunk
                    continue
                end_of_file = True

            if not buffer:
                if end_of_file:
                    raise ValueError(f"Unexpected end of file while parsing JSON array in {file_path}.")
                continue

            if buffer[0] == "]":
                return

            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            try:
                row, index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if end_of_file:
                    raise ValueError(f"Invalid JSON array contents in {file_path}.")
                chunk = handle.read(JSON_CHUNK_SIZE)
                if not chunk:
                    end_of_file = True
                else:
                    buffer += chunk
                continue

            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON objects inside {file_path}, found {type(row).__name__}.")

            yield row
            buffer = buffer[index:]



def iter_json_rows(file_path: str) -> Iterator[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        first_non_ws = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first_non_ws = char
                break

    if first_non_ws == "[":
        yield from _iter_json_array_rows(file_path)
        return

    yield from _iter_jsonl_rows(file_path)



def iter_limited_rows(file_path: str, max_rows: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    for row_number, row in enumerate(iter_json_rows(file_path), start=1):
        if max_rows is not None and row_number > max_rows:
            break
        yield row



def collect_metadata_accession_numbers(metadata_path: str, max_rows: Optional[int] = None) -> Set[str]:
    accession_numbers: Set[str] = set()

    for row_number, row in enumerate(iter_limited_rows(metadata_path, max_rows=max_rows), start=1):
        accession_number = str(row.get("accessionNumber", "")).strip()
        if accession_number:
            accession_numbers.add(accession_number)
        print(f"\rScanning metadata rows: {row_number}", end="")

    print("")
    return accession_numbers



def build_ecl_text_lookup(ecl_path: str, accession_numbers: Set[str]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}

    if not accession_numbers:
        return lookup

    for row_number, row in enumerate(iter_json_rows(ecl_path), start=1):
        filename = str(row.get("filename", ""))
        accession_number = extract_accession_number_index(filename).strip()
        if accession_number not in accession_numbers:
            continue

        lookup[accession_number] = {
            "item_7": row.get("item_7"),
            "opinion_text": row.get("opinion_text"),
        }
        print(
            f"\rBuilding ECL lookup - current row: {row_number} | matches: {len(lookup)}/{len(accession_numbers)}",
            end="",
        )

        if len(lookup) == len(accession_numbers):
            break

    print("")
    return lookup



def generate_text_modality_dataset(
    metadata_path: str,
    ecl_path: str,
    output_path: str,
    max_rows: Optional[int] = None,
) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    accession_numbers = collect_metadata_accession_numbers(metadata_path=metadata_path, max_rows=max_rows)
    ecl_lookup = build_ecl_text_lookup(ecl_path=ecl_path, accession_numbers=accession_numbers)

    rows_written = 0
    missing_matches = 0

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()

        for row in iter_limited_rows(metadata_path, max_rows=max_rows):
            accession_number = str(row.get("accessionNumber", "")).strip()
            matched_text = ecl_lookup.get(accession_number, {})
            if accession_number and accession_number not in ecl_lookup:
                missing_matches += 1

            writer.writerow(
                {
                    "accessionNumber": accession_number,
                    "label": row.get("label"),
                    "item_7": matched_text.get("item_7"),
                    "opinion_text": matched_text.get("opinion_text"),
                }
            )
            rows_written += 1
            print(f"\rWriting CSV rows: {rows_written}", end="")

    print("")
    print(
        f"Done. Wrote {rows_written:,} rows to {output_path}. "
        f"Missing ECL matches: {missing_matches:,}."
    )
    return rows_written



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MDA + auditor text modality dataset.")
    parser.add_argument(
        "--metadata",
        default=config.COMPANYFACTS_METADATA_PATH,
        help="Path to the metadata JSON/JSONL file.",
    )
    parser.add_argument(
        "--ecl-input",
        default=config.ECL_FILE_PATH,
        help="Path to the original ECL JSON/JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=config.MDA_AUDITOR_DATASET_PATH,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for test runs.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    generate_text_modality_dataset(
        metadata_path=args.metadata,
        ecl_path=args.ecl_input,
        output_path=args.output,
        max_rows=args.max_rows,
    )



if __name__ == "__main__":
    main()
