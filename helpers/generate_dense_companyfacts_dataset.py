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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sec_data_parser import retrieve_sec_tags_and_values
from tools import config
from tools.data_loader import DataLoader


DEFAULT_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv")


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
    create_dense_dataset(input_path=args.input, output_path=args.output)
