"""Generate a dense companyfacts dataset using k-means tag clusters.

This helper keeps the existing SEC statement retrieval path intact and replaces
the fuzzy concept mapping step with stable, per-section k-means clusters fitted
over observed SEC tag/label text.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from statistics import fmean
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers.generate_dense_companyfacts_dataset import (  # noqa: E402
    ACCESSION_NUMBER_COLUMN,
    BASE_OUTPUT_COLUMNS,
    CIK_ADSH_COLUMN,
    FINANCIAL_SECTIONS,
    _csv_value,
    _ensure_required_metadata_columns,
    _is_xbrl_row,
    _iter_metadata_rows,
    _write_row,
)
from tools import config  # noqa: E402


DEFAULT_INPUT_PATH = getattr(config, "COMPANYFACTS_METADATA_PATH", config.ECL_METADATA_NOTEXT_PATH)
DEFAULT_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "companyfacts_kmeans_dense.csv")
DEFAULT_INTERMEDIATE_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "kmeans_companyfacts_long.csv")
DEFAULT_TAG_CLUSTERS_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "kmeans_tag_clusters.csv")
DEFAULT_TAGS_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "kmeans_tags.csv")
DEFAULT_CLUSTER_COUNTS = {
    "IncomeStatement": 60,
    "BalanceSheet": 80,
    "CashFlow": 50,
    "StatementOfStockholdersEquity": 40,
}
LONG_FIELDNAMES = (CIK_ADSH_COLUMN, "label", "year", "section", "tag", "tlabel", "value")
TAG_CLUSTER_FIELDNAMES = ("section", "tag", "label", "cluster", "distance", "cluster_representative")
SECTION_PREFIXES = {section: code.lower() for section, code, _ in FINANCIAL_SECTIONS}
SECTION_ALIASES = {
    **{section.lower(): section for section, _, _ in FINANCIAL_SECTIONS},
    **{code.lower(): section for section, code, _ in FINANCIAL_SECTIONS},
    **{SECTION_PREFIXES[section]: section for section, _, _ in FINANCIAL_SECTIONS},
    "income": "IncomeStatement",
    "income_statement": "IncomeStatement",
    "balance": "BalanceSheet",
    "balance_sheet": "BalanceSheet",
    "cashflow": "CashFlow",
    "cash_flow": "CashFlow",
    "equity": "StatementOfStockholdersEquity",
    "stockholders_equity": "StatementOfStockholdersEquity",
}

CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
NON_WORD_RE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class TagRecord:
    section: str
    tag: str
    label: str


@dataclass(frozen=True)
class TagClusterAssignment:
    section: str
    tag: str
    label: str
    cluster: int
    distance: float
    cluster_representative: str


def make_feature_text(tag: object, label: object) -> str:
    """Normalize tag and label text before char n-gram vectorization."""
    parts: list[str] = []
    seen: set[str] = set()

    for value in (tag, label):
        text = "" if value is None else str(value)
        text = CAMEL_BOUNDARY_RE.sub(" ", text)
        text = NON_WORD_RE.sub(" ", text).strip().lower()
        if text and text not in seen:
            parts.append(text)
            seen.add(text)

    return " ".join(parts) or "missing"


def import_sklearn_components():
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for k-means clustering. "
            "Install the project requirements before running this generator."
        ) from exc

    return KMeans, TfidfVectorizer


def build_text_vectorizer():
    _, TfidfVectorizer = import_sklearn_components()
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))


def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def cluster_column_name(section: str, cluster: int) -> str:
    return f"{SECTION_PREFIXES[section]}_cluster_{cluster:03d}"


def assignment_key(section: object, tag: object, label: object) -> tuple[str, str, str]:
    return (
        "" if section is None else str(section),
        "" if tag is None else str(tag),
        "" if label is None else str(label),
    )


def cluster_representative_text(record: TagRecord) -> str:
    if record.label:
        return f"{record.tag} | {record.label}"
    return record.tag


def parse_cluster_overrides(values: Sequence[str] | None) -> dict[str, int]:
    cluster_counts = dict(DEFAULT_CLUSTER_COUNTS)
    if not values:
        return cluster_counts

    tokens: list[str] = []
    for value in values:
        tokens.extend(token for token in re.split(r"[,;]", value) if token.strip())

    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Invalid cluster override {token!r}. Use SectionName=COUNT.")
        raw_section, raw_count = token.split("=", 1)
        section_key = raw_section.strip().lower().replace("-", "_")
        section = SECTION_ALIASES.get(section_key)
        if section is None:
            valid_sections = ", ".join(DEFAULT_CLUSTER_COUNTS)
            raise ValueError(f"Unknown financial section {raw_section!r}. Valid sections: {valid_sections}.")

        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"Cluster count for {raw_section!r} must be an integer.") from exc
        if count <= 0:
            raise ValueError(f"Cluster count for {raw_section!r} must be positive.")

        cluster_counts[section] = count

    return cluster_counts


def extract_long_companyfacts_table(
    input_path: str,
    output_path: str,
    max_rows: int | None = None,
) -> int:
    """Write filing-level statement facts to a long CSV for clustering."""
    from db_connection import close_connection, create_connection, retrieve_statement_taxonomies_by_accession_number3

    ensure_parent_dir(output_path)
    conn = None
    long_rows_written = 0
    metadata_rows_seen = 0

    try:
        conn = create_connection()
        with open(output_path, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=LONG_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()

            for row in _iter_metadata_rows(input_path, max_rows=max_rows):
                metadata_rows_seen += 1
                print(f"\rExtracting k-means source facts - current row: {metadata_rows_seen}", end="")
                _ensure_required_metadata_columns(row, row_number=metadata_rows_seen)

                is_primary_registrant = str(row.get("filing_role", "")).strip().lower() == "primary_registrant"
                if not (_is_xbrl_row(row) and is_primary_registrant):
                    continue

                for section_name, stmt_code, quarter_spec in FINANCIAL_SECTIONS:
                    tags, labels, values = retrieve_statement_taxonomies_by_accession_number3(
                        conn,
                        row[ACCESSION_NUMBER_COLUMN],
                        row["cik"],
                        row.get("filing_role"),
                        stmt_code,
                        quarter_spec,
                    )
                    for tag, tag_label, value in zip(tags, labels, values):
                        if not tag:
                            continue
                        _write_row(
                            writer,
                            {
                                CIK_ADSH_COLUMN: row.get(CIK_ADSH_COLUMN),
                                "label": row.get("label"),
                                "year": row.get("year"),
                                "section": section_name,
                                "tag": tag,
                                "tlabel": tag_label,
                                "value": value,
                            },
                        )
                        long_rows_written += 1

        print(f"\nFinished writing {long_rows_written:,} long rows to {output_path}")
        return long_rows_written
    finally:
        if conn is not None:
            close_connection(conn)


def iter_long_rows(path: str) -> Iterable[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def collect_unique_tag_records(long_table_path: str) -> dict[str, list[TagRecord]]:
    records_by_section: dict[str, dict[tuple[str, str], TagRecord]] = {
        section: {} for section, _, _ in FINANCIAL_SECTIONS
    }

    for row in iter_long_rows(long_table_path):
        section = row.get("section", "")
        if section not in records_by_section:
            continue

        tag = row.get("tag", "")
        label = row.get("tlabel", "")
        if not tag:
            continue

        records_by_section[section][(tag, label)] = TagRecord(section=section, tag=tag, label=label)

    return {
        section: sorted(records.values(), key=lambda item: (make_feature_text(item.tag, item.label), item.tag, item.label))
        for section, records in records_by_section.items()
    }


def fit_section_clusters(
    section: str,
    records: Sequence[TagRecord],
    requested_cluster_count: int,
) -> list[TagClusterAssignment]:
    """Fit one section-level k-means model and return relabeled assignments."""
    if not records:
        return []

    if requested_cluster_count <= 0:
        raise ValueError("requested_cluster_count must be positive.")

    ordered_records = sorted(records, key=lambda item: (make_feature_text(item.tag, item.label), item.tag, item.label))
    feature_texts = [make_feature_text(record.tag, record.label) for record in ordered_records]
    n_clusters = min(requested_cluster_count, len(set(feature_texts)))

    KMeans, _ = import_sklearn_components()
    vectorizer = build_text_vectorizer()
    feature_matrix = vectorizer.fit_transform(feature_texts)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    original_labels = model.fit_predict(feature_matrix)
    distance_matrix = model.transform(feature_matrix)
    assigned_distances = [float(distance_matrix[row_index, cluster]) for row_index, cluster in enumerate(original_labels)]

    representatives: dict[int, int] = {}
    for original_cluster in sorted(set(int(label) for label in original_labels)):
        member_indices = [
            index for index, label in enumerate(original_labels) if int(label) == original_cluster
        ]
        representatives[original_cluster] = min(
            member_indices,
            key=lambda index: (
                assigned_distances[index],
                feature_texts[index],
                ordered_records[index].tag,
                ordered_records[index].label,
            ),
        )

    ordered_original_clusters = sorted(
        representatives,
        key=lambda original_cluster: (
            feature_texts[representatives[original_cluster]],
            ordered_records[representatives[original_cluster]].tag,
            ordered_records[representatives[original_cluster]].label,
        ),
    )
    relabel_map = {
        original_cluster: new_cluster
        for new_cluster, original_cluster in enumerate(ordered_original_clusters, start=1)
    }

    assignments: list[TagClusterAssignment] = []
    for index, record in enumerate(ordered_records):
        original_cluster = int(original_labels[index])
        representative = ordered_records[representatives[original_cluster]]
        assignments.append(
            TagClusterAssignment(
                section=section,
                tag=record.tag,
                label=record.label,
                cluster=relabel_map[original_cluster],
                distance=assigned_distances[index],
                cluster_representative=cluster_representative_text(representative),
            )
        )

    return sorted(assignments, key=lambda item: (item.cluster, item.distance, item.tag, item.label))


def fit_clusters_from_long_table(
    long_table_path: str,
    cluster_counts: Mapping[str, int],
) -> dict[tuple[str, str, str], TagClusterAssignment]:
    records_by_section = collect_unique_tag_records(long_table_path)
    assignments: dict[tuple[str, str, str], TagClusterAssignment] = {}

    for section, _, _ in FINANCIAL_SECTIONS:
        section_assignments = fit_section_clusters(
            section=section,
            records=records_by_section.get(section, []),
            requested_cluster_count=cluster_counts[section],
        )
        for assignment in section_assignments:
            assignments[assignment_key(assignment.section, assignment.tag, assignment.label)] = assignment

    return assignments


def actual_cluster_counts(
    assignments: Mapping[tuple[str, str, str], TagClusterAssignment],
) -> dict[str, int]:
    counts = {section: 0 for section, _, _ in FINANCIAL_SECTIONS}
    for assignment in assignments.values():
        counts[assignment.section] = max(counts[assignment.section], assignment.cluster)
    return counts


def cluster_columns_from_counts(cluster_counts: Mapping[str, int]) -> list[str]:
    columns: list[str] = []
    for section, _, _ in FINANCIAL_SECTIONS:
        columns.extend(cluster_column_name(section, cluster) for cluster in range(1, cluster_counts.get(section, 0) + 1))
    return columns


def write_tag_cluster_audit(
    assignments: Mapping[tuple[str, str, str], TagClusterAssignment],
    output_path: str,
) -> None:
    ensure_parent_dir(output_path)
    rows = sorted(
        assignments.values(),
        key=lambda item: (item.section, item.cluster, item.distance, item.tag, item.label),
    )

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TAG_CLUSTER_FIELDNAMES)
        writer.writeheader()
        for assignment in rows:
            writer.writerow(
                {
                    "section": assignment.section,
                    "tag": assignment.tag,
                    "label": assignment.label,
                    "cluster": assignment.cluster,
                    "distance": f"{assignment.distance:.12g}",
                    "cluster_representative": assignment.cluster_representative,
                }
            )


def parse_numeric_value(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return float(numeric_value)


def transform_filing_facts(
    facts: Sequence[Mapping[str, object]],
    assignments: Mapping[tuple[str, str, str], TagClusterAssignment],
    aggregation: str = "nearest",
) -> tuple[dict[str, object], dict[str, str]]:
    """Transform long facts for one filing into dense cluster values and tag audit cells."""
    nearest_candidates: dict[str, tuple[float, str, str, object]] = {}
    mean_values: dict[str, list[float]] = defaultdict(list)
    mean_tags: dict[str, set[str]] = defaultdict(set)

    for fact in facts:
        assignment = assignments.get(assignment_key(fact.get("section"), fact.get("tag"), fact.get("tlabel")))
        if assignment is None:
            continue

        column = cluster_column_name(assignment.section, assignment.cluster)
        tag = "" if fact.get("tag") is None else str(fact.get("tag"))
        label = "" if fact.get("tlabel") is None else str(fact.get("tlabel"))
        feature_text = make_feature_text(tag, label)

        if aggregation == "nearest":
            candidate = (assignment.distance, feature_text, tag, fact.get("value"))
            previous = nearest_candidates.get(column)
            if previous is None or candidate[:3] < previous[:3]:
                nearest_candidates[column] = candidate
        elif aggregation == "mean":
            numeric_value = parse_numeric_value(fact.get("value"))
            if numeric_value is not None:
                mean_values[column].append(numeric_value)
                mean_tags[column].add(tag)
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation}")

    if aggregation == "nearest":
        values = {column: candidate[3] for column, candidate in nearest_candidates.items()}
        tags = {column: candidate[2] for column, candidate in nearest_candidates.items()}
        return values, tags

    values = {column: fmean(column_values) for column, column_values in mean_values.items() if column_values}
    tags = {column: ";".join(sorted(column_tags)) for column, column_tags in mean_tags.items() if column_tags}
    return values, tags


def build_clustered_filing_maps(
    long_table_path: str,
    assignments: Mapping[tuple[str, str, str], TagClusterAssignment],
    aggregation: str,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, str]]]:
    facts_by_filing: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in iter_long_rows(long_table_path):
        filing_id = row.get(CIK_ADSH_COLUMN, "")
        if not filing_id:
            continue
        facts_by_filing[filing_id].append(row)

    values_by_filing: dict[str, dict[str, object]] = {}
    tags_by_filing: dict[str, dict[str, str]] = {}
    for filing_id, facts in facts_by_filing.items():
        values, tags = transform_filing_facts(facts=facts, assignments=assignments, aggregation=aggregation)
        values_by_filing[filing_id] = values
        tags_by_filing[filing_id] = tags

    return values_by_filing, tags_by_filing


def write_dense_clustered_csv(
    input_path: str,
    output_path: str,
    tags_output_path: str,
    cluster_columns: Sequence[str],
    values_by_filing: Mapping[str, Mapping[str, object]],
    tags_by_filing: Mapping[str, Mapping[str, str]],
    max_rows: int | None = None,
) -> int:
    ensure_parent_dir(output_path)
    ensure_parent_dir(tags_output_path)

    dense_fieldnames = [*BASE_OUTPUT_COLUMNS, *cluster_columns]
    tags_fieldnames = [*BASE_OUTPUT_COLUMNS, *cluster_columns]
    rows_written = 0

    with open(output_path, "w", encoding="utf-8", newline="") as dense_file, open(
        tags_output_path, "w", encoding="utf-8", newline=""
    ) as tags_file:
        dense_writer = csv.DictWriter(dense_file, fieldnames=dense_fieldnames, extrasaction="ignore")
        dense_writer.writeheader()
        tags_writer = csv.DictWriter(tags_file, fieldnames=tags_fieldnames, extrasaction="ignore")
        tags_writer.writeheader()

        for row in _iter_metadata_rows(input_path, max_rows=max_rows):
            rows_written += 1
            print(f"\rWriting k-means dense rows - current row: {rows_written}", end="")
            _ensure_required_metadata_columns(row, row_number=rows_written)

            filing_id = row.get(CIK_ADSH_COLUMN)
            dense_row = {column: row.get(column) for column in BASE_OUTPUT_COLUMNS}
            dense_row.update({column: None for column in cluster_columns})
            dense_row.update(values_by_filing.get(filing_id, {}))
            _write_row(dense_writer, dense_row)

            tags_row = {column: row.get(column) for column in BASE_OUTPUT_COLUMNS}
            tags_row.update({column: None for column in cluster_columns})
            tags_row.update(tags_by_filing.get(filing_id, {}))
            tags_writer.writerow({key: _csv_value(value) for key, value in tags_row.items()})

    print(f"\nFinished writing {rows_written:,} rows to {output_path}")
    return rows_written


def create_kmeans_dense_dataset(
    input_path: str,
    output_path: str,
    max_rows: int | None = None,
    cluster_counts: Mapping[str, int] | None = None,
    aggregation: str = "nearest",
    intermediate_output_path: str = DEFAULT_INTERMEDIATE_OUTPUT_PATH,
    tag_clusters_output_path: str = DEFAULT_TAG_CLUSTERS_OUTPUT_PATH,
    tags_output_path: str = DEFAULT_TAGS_OUTPUT_PATH,
) -> None:
    if aggregation not in {"nearest", "mean"}:
        raise ValueError("aggregation must be either 'nearest' or 'mean'.")

    import_sklearn_components()
    effective_cluster_counts = {**DEFAULT_CLUSTER_COUNTS, **dict(cluster_counts or {})}

    extract_long_companyfacts_table(
        input_path=input_path,
        output_path=intermediate_output_path,
        max_rows=max_rows,
    )
    assignments = fit_clusters_from_long_table(
        long_table_path=intermediate_output_path,
        cluster_counts=effective_cluster_counts,
    )
    write_tag_cluster_audit(assignments=assignments, output_path=tag_clusters_output_path)

    fitted_cluster_counts = actual_cluster_counts(assignments)
    cluster_columns = cluster_columns_from_counts(fitted_cluster_counts)
    values_by_filing, tags_by_filing = build_clustered_filing_maps(
        long_table_path=intermediate_output_path,
        assignments=assignments,
        aggregation=aggregation,
    )
    write_dense_clustered_csv(
        input_path=input_path,
        output_path=output_path,
        tags_output_path=tags_output_path,
        cluster_columns=cluster_columns,
        values_by_filing=values_by_filing,
        tags_by_filing=tags_by_filing,
        max_rows=max_rows,
    )

    print(f"Wrote tag cluster audit to {tag_clusters_output_path}")
    print(f"Wrote per-filing tag audit to {tags_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense companyfacts dataset using k-means tag clusters.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to metadata JSONL/CSV input.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to clustered dense CSV output.",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        default=None,
        help=(
            "Optional per-section cluster counts, for example "
            "IncomeStatement=60 BalanceSheet=80 CashFlow=50 StatementOfStockholdersEquity=40."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of metadata rows to process for quick test runs.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("nearest", "mean"),
        default="nearest",
        help="How to collapse multiple filing tags assigned to one cluster.",
    )
    parser.add_argument(
        "--intermediate-output",
        default=DEFAULT_INTERMEDIATE_OUTPUT_PATH,
        help="Path to the long k-means source table.",
    )
    parser.add_argument(
        "--tag-clusters-output",
        default=DEFAULT_TAG_CLUSTERS_OUTPUT_PATH,
        help="Path to the tag-to-cluster audit CSV.",
    )
    parser.add_argument(
        "--tags-output",
        default=DEFAULT_TAGS_OUTPUT_PATH,
        help="Path to the per-filing populated-tag audit CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_kmeans_dense_dataset(
        input_path=args.input,
        output_path=args.output,
        max_rows=args.max_rows,
        cluster_counts=parse_cluster_overrides(args.clusters),
        aggregation=args.aggregation,
        intermediate_output_path=args.intermediate_output,
        tag_clusters_output_path=args.tag_clusters_output,
        tags_output_path=args.tags_output,
    )
