import submissions_parser
from tools.config import OUTPUT_DIR, DATA_DIR
from tools.data_loader import DataLoader
import json
from tools import config, db_schema
from tools.utils import extract_year_from_filename, extract_accession_number_index
import pandas as pd
import os

data_loader = DataLoader()
# ecl_meta = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)

def drop_non_numeric_cols(df):
    print("Numerical df's original columns:  ", df.columns)

    columns_to_drop = [
        'cik',
        "bankruptcy_prediction_split",
        'company','period_of_report','gvkey','filing_date','year','accessionNumber','reportDateIndex',
        'datadate', 'filename', 'can_label', 'qualified', 'cik_year', 'gc_list',
        'bankruptcy_date_1','bankruptcy_date_2','bankruptcy_date_3','form','primaryDocument',
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


dense_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'ecl_with_financial_tags.csv'), low_memory=False)
sparse_df = pd.read_csv(os.path.join(config.DATA_DIR, 'numerical.csv'), low_memory=False)
dense_df = add_missing_uids(dense_df, sparse_df)
dense_df = drop_non_numeric_cols(dense_df)
data_loader.save_dataset(key_or_df=dense_df, out_path=os.path.join(OUTPUT_DIR, 'numerical_dense_merged_sept.csv'))


def add_metadata_to_original_ecl():
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

