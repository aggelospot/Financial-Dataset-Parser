"""Streaming helpers for combining metadata columns into generated datasets.

Legacy dataframe utilities are preserved for notebooks/experiments, while the
module now also exposes a CLI consistent with the other helper scripts.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import config
from tools.config import OUTPUT_DIR
from tools.data_loader import DataLoader


data_loader = DataLoader()

DATASET_PATHS: Dict[str, str] = {
    "dense": config.COMPANYFACTS_DENSE_PATH,
    "sparse": config.COMPANYFACTS_SPARSE_PATH,
}


# ecl_meta = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)
def drop_non_numeric_cols(df):
    print("Numerical df's original columns:  ", df.columns)

    columns_to_drop = [
        'cik',
        "bankruptcy_prediction_split",
        'company', 'period_of_report', 'gvkey', 'filing_date', 'year', 'accessionNumber', 'reportDateIndex',
        'datadate', 'filename', 'can_label', 'qualified', 'cik_year', 'gc_list',
        'bankruptcy_date_1', 'bankruptcy_date_2', 'bankruptcy_date_3', 'form', 'primaryDocument',
        'isXBRL'
    ]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    print("Numerical df's columns after drop:  ", df.columns)

    # data_loader.save_dataset(key_or_df=df, out_path=os.path.join(OUTPUT_DIR, 'numerical_dense_merged_no_metadata.csv'))
    return df



def add_missing_uids_to_dense_df(dense_df, sparse_df):
    #  Keep ONLY those rows in `dense` whose uid exists in `sparse`
    #     Using .isin is vector-ised and avoids slow Python-level loops.
    dense_trimmed = dense_df[dense_df['uid'].isin(sparse_df['uid'])].copy()

    # Identify uids present in `sparse` but absent from the trimmed `dense`
    uids_to_add = sparse_df.loc[~sparse_df['uid'].isin(dense_trimmed['uid']), ['uid', 'label']]

    # Re-index the new rows so they have the same columns as `dense_trimmed`
    #     • Any columns other than 'uid' and 'label' become NaN (blank)
    rows_to_append = uids_to_add.reindex(columns=dense_trimmed.columns)

    # Concatenate: original kept rows + new blank-filled rows
    merged = pd.concat([dense_trimmed, rows_to_append], ignore_index=True)

    return merged


def add_uids_to_dense_df(dense_df):
    dense_df['uid'] = dense_df['cik'].astype(str) + '__' + pd.to_datetime(dense_df['period_of_report']).dt.strftime('%Y-%m-%d')



def add_missing_uids(dense_df, sparse_df):
    print("Dense df shape before merge: ", dense_df.shape)
    add_uids_to_dense_df(dense_df)
    dense_df = add_missing_uids_to_dense_df(dense_df, sparse_df)
    print("Dense df shape after merge: ", dense_df.shape)
    # data_loader.save_dataset(dense_df, os.path.join(config.OUTPUT_DIR, "numerical_dense_merged.csv"))
    return dense_df



def _iter_jsonl_rows(file_path: str) -> Iterator[Dict[str, object]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {file_path}: {exc}") from exc



def _next_metadata_row(metadata_iter: Iterator[Dict[str, object]], metadata_path: str, row_number: int) -> Dict[str, object]:
    try:
        return next(metadata_iter)
    except StopIteration as exc:
        raise ValueError(
            f"Metadata file ended early at dataset row {row_number}. "
            f"Expected at least as many metadata rows as dataset rows in {metadata_path}."
        ) from exc



def _ensure_metadata_columns_exist(sample_row: Dict[str, object], add_columns: Sequence[str], metadata_path: str) -> None:
    missing_columns = [column for column in add_columns if column not in sample_row]
    if missing_columns:
        raise ValueError(
            f"Metadata file {metadata_path} does not contain the requested column(s): {missing_columns}"
        )



def _resolve_dataset_path(dataset_name: str) -> str:
    configured_path = DATASET_PATHS[dataset_name]
    if os.path.exists(configured_path):
        return configured_path

    root, extension = os.path.splitext(configured_path)
    if extension.lower() != ".csv":
        csv_fallback = f"{root}.csv"
        if os.path.exists(csv_fallback):
            return csv_fallback

    raise FileNotFoundError(
        f"Configured {dataset_name} dataset path does not exist: {configured_path}"
    )



def _write_updated_csv_dataset(
    dataset_path: str,
    metadata_path: str,
    add_columns: Sequence[str],
) -> int:
    row_count = 0
    metadata_iter = _iter_jsonl_rows(metadata_path)
    first_metadata_row = _next_metadata_row(metadata_iter, metadata_path=metadata_path, row_number=1)
    _ensure_metadata_columns_exist(first_metadata_row, add_columns=add_columns, metadata_path=metadata_path)

    temp_fd, temp_path = tempfile.mkstemp(prefix="df_combiner_", suffix=".csv", dir=os.path.dirname(dataset_path) or ".")
    os.close(temp_fd)

    try:
        with open(dataset_path, "r", encoding="utf-8", newline="") as dataset_handle, open(
            temp_path, "w", encoding="utf-8", newline=""
        ) as output_handle:
            reader = csv.DictReader(dataset_handle)
            if reader.fieldnames is None:
                raise ValueError(f"Dataset {dataset_path} does not contain a CSV header.")

            output_columns = list(reader.fieldnames)
            for column in add_columns:
                if column not in output_columns:
                    output_columns.append(column)

            writer = csv.DictWriter(output_handle, fieldnames=output_columns)
            writer.writeheader()

            pending_metadata_row: Optional[Dict[str, object]] = first_metadata_row
            for row_count, dataset_row in enumerate(reader, start=1):
                metadata_row = pending_metadata_row or _next_metadata_row(
                    metadata_iter,
                    metadata_path=metadata_path,
                    row_number=row_count,
                )
                pending_metadata_row = None

                for column in add_columns:
                    dataset_row[column] = metadata_row.get(column)
                writer.writerow(dataset_row)
                print(f"\rWriting updated CSV row: {row_count}", end="")

            try:
                next(metadata_iter)
            except StopIteration:
                pass
            else:
                print("")
                raise ValueError(
                    f"Metadata file {metadata_path} contains more rows than dataset {dataset_path}."
                )

        os.replace(temp_path, dataset_path)
        print("")
        return row_count
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise



def _write_updated_jsonl_dataset(
    dataset_path: str,
    metadata_path: str,
    add_columns: Sequence[str],
) -> int:
    row_count = 0
    metadata_iter = _iter_jsonl_rows(metadata_path)
    first_metadata_row = _next_metadata_row(metadata_iter, metadata_path=metadata_path, row_number=1)
    _ensure_metadata_columns_exist(first_metadata_row, add_columns=add_columns, metadata_path=metadata_path)

    temp_fd, temp_path = tempfile.mkstemp(prefix="df_combiner_", suffix=".json", dir=os.path.dirname(dataset_path) or ".")
    os.close(temp_fd)

    try:
        with open(dataset_path, "r", encoding="utf-8") as dataset_handle, open(temp_path, "w", encoding="utf-8") as output_handle:
            pending_metadata_row: Optional[Dict[str, object]] = first_metadata_row
            for row_number, line in enumerate(dataset_handle, start=1):
                if not line.strip():
                    continue

                try:
                    dataset_row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {row_number} of {dataset_path}: {exc}") from exc

                row_count += 1
                metadata_row = pending_metadata_row or _next_metadata_row(
                    metadata_iter,
                    metadata_path=metadata_path,
                    row_number=row_count,
                )
                pending_metadata_row = None

                for column in add_columns:
                    dataset_row[column] = metadata_row.get(column)
                output_handle.write(json.dumps(dataset_row) + "\n")
                print(f"\rWriting updated JSONL row: {row_count}", end="")

            try:
                next(metadata_iter)
            except StopIteration:
                pass
            else:
                print("")
                raise ValueError(
                    f"Metadata file {metadata_path} contains more rows than dataset {dataset_path}."
                )

        os.replace(temp_path, dataset_path)
        print("")
        return row_count
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise



def add_metadata_columns_to_dataset(
    dataset_name: str,
    add_columns: Sequence[str],
    metadata_path: str = config.COMPANYFACTS_METADATA_PATH,
) -> int:
    """Stream requested metadata JSONL columns into the configured dense/sparse dataset."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {sorted(DATASET_PATHS)}")
    if not add_columns:
        raise ValueError("At least one metadata column must be provided.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    dataset_path = _resolve_dataset_path(dataset_name)
    _, extension = os.path.splitext(dataset_path)
    extension = extension.lower()

    unique_columns = list(dict.fromkeys(column.strip() for column in add_columns if column.strip()))
    if not unique_columns:
        raise ValueError("No valid metadata columns were provided.")

    if extension == ".csv":
        return _write_updated_csv_dataset(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            add_columns=unique_columns,
        )
    if extension == ".json":
        return _write_updated_jsonl_dataset(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            add_columns=unique_columns,
        )

    raise ValueError(f"Unsupported dataset extension for {dataset_path}: {extension}")



def add_metadata_to_original_ecl():
    import submissions_parser
    from tools.utils import extract_accession_number_index, extract_year_from_filename

    ecl = data_loader.load_dataset(config.ECL_FILE_PATH, alias='ecl_text', lines=True)
    ecl = ecl.drop('opinion_text', axis=1)
    ecl = ecl.drop('item_7', axis=1)
    print("after drop columns ", len(ecl.index), ecl.columns)

    original_shape = ecl.shape
    ecl["year"] = pd.to_numeric(ecl["filename"].apply(extract_year_from_filename))
    ecl["accessionNumber"] = ecl["filename"].apply(extract_accession_number_index)
    # ecl = ecl.loc[ecl["year"] >= 2007]  # Years before the threshold are dropped

    ecl_metadata = submissions_parser.add_submissions_metadata(ecl)
    ecl_metadata["accessionNumber"] = ecl_metadata['accessionNumber'].astype(str)
    data_loader.save_dataset(key_or_df=ecl_metadata, out_path=os.path.join(OUTPUT_DIR, 'ecl_metadata_complete.json'))

    # ecl_metadata_full = data_loader



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add metadata JSONL columns to the configured dense/sparse dataset.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PATHS.keys()),
        required=True,
        help="Which dataset route to update from config (dense or sparse).",
    )
    parser.add_argument(
        "--add-columns",
        required=True,
        help="Comma-separated metadata column names to append to the selected dataset.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    row_count = add_metadata_columns_to_dataset(
        dataset_name=args.dataset,
        add_columns=args.add_columns.split(","),
    )
    dataset_path = _resolve_dataset_path(args.dataset)
    print(
        f"Done. Added columns {args.add_columns} to {args.dataset} dataset at {dataset_path}. "
        f"Rows updated: {row_count:,}"
    )



if __name__ == "__main__":
    main()
