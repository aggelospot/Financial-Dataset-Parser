import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import ModelWrapper
from models.utils import evaluate_bankruptcy_model

import os
import sys

from tools.utils import match_concept_in_section

notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from tools import config

def run_bankruptcy_experiments(model, df, numeric_cols, n_runs=5, test_size=0.2, drop_na=True, smote=True):
    all_accuracies = []
    all_avgprec = []
    all_recall100 = []
    all_rocauc = []
    all_accuracies_train = []
    all_avgprec_train = []
    all_recall100_train = []
    all_rocauc_train = []

    for seed in range(n_runs):
        print(f"\n===== RANDOM SEED: {seed} =====")

        # model = model_factory(seed)

        train_df = df[df['bankruptcy_prediction_split'] == 'train']
        test_df = df[df['bankruptcy_prediction_split'] == 'test']


        if drop_na:
            train_df = train_df.dropna(subset=numeric_cols)
            test_df = test_df.dropna(subset=numeric_cols)

        X_train = train_df.drop(columns=['cik','label','bankruptcy_prediction_split'], errors='ignore')
        y_train = train_df['label']
        X_test = test_df.drop(columns=['cik', 'label','bankruptcy_prediction_split'], errors='ignore')
        y_test = test_df['label']

        # print("TRAIN Label Distribution:\n", y_train.value_counts())
        # print("TEST Label Distribution:\n", y_test.value_counts())

        # =========== Under-sampling + SMOTE (for imbalance) ===========
        if smote:
            rus = RandomUnderSampler(sampling_strategy=0.10, random_state=seed)
            X_rus, y_rus = rus.fit_resample(X_train, y_train)

            sm = SMOTE(sampling_strategy='minority', random_state=seed)
            X_train_res, y_train_res = sm.fit_resample(X_rus, y_rus)
        else:
            X_train_res, y_train_res = X_train, y_train


        y_train_res = y_train_res.astype(int)

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        model.fit(X_train_res.values, y_train_res.values)

        # =========== Evaluate on test ===========
        y_proba_test = model.predict_proba(X_test.values)[:, 1]

        y_pred_test = model.predict(X_test.values)
        metrics_dict = evaluate_bankruptcy_model(
            y_true=y_test,
            y_pred=y_pred_test,
            y_proba=y_proba_test
        )
        all_accuracies.append(metrics_dict['accuracy'])
        all_avgprec.append(metrics_dict['average_precision'])
        all_recall100.append(metrics_dict['recall_at_100'])
        all_rocauc.append(metrics_dict['roc_auc'])

        # =========== Evaluate on train ===========
        y_proba_train = model.predict_proba(X_train.values)[:, 1]

        y_pred_train = model.predict(X_train.values)
        metrics_dict_train = evaluate_bankruptcy_model(
            y_true=y_train,
            y_pred=y_pred_train,
            y_proba=y_proba_train
        )
        all_accuracies_train.append(metrics_dict_train['accuracy'])
        all_avgprec_train.append(metrics_dict_train['average_precision'])
        all_recall100_train.append(metrics_dict_train['recall_at_100'])
        all_rocauc_train.append(metrics_dict_train['roc_auc'])

    print(f"\n===== AVERAGE METRICS ACROSS SEEDS =====")
    model_name = model.__class__.__name__
    print(f"\n===== MODEL: {model_name} =====")
    print(f"\n===== TEST DATA =====")
    print("Accuracy ", all_accuracies)
    print("Mean Accuracy:           {:.2f}%, std: {:.2f}%".format(np.mean(all_accuracies) * 100, np.std(all_accuracies) * 100))
    print("Mean Average Precision:  {:.2f}%, std: {:.2f}%".format(np.mean(all_avgprec) * 100, np.std(all_avgprec) * 100))
    print("Mean Recall@100:         {:.2f}%, std: {:.2f}%".format(np.mean(all_recall100) * 100, np.std(all_recall100) * 100))
    print("Mean ROC AUC:            {:.2f}%, std: {:.2f}%".format(np.mean(all_rocauc) * 100, np.std(all_rocauc) * 100))

    print(f"\n===== TRAIN DATA =====")
    print("Mean Accuracy:           {:.2f}%, std: {:.2f}%".format(np.mean(all_accuracies_train) * 100, np.std(all_accuracies_train) * 100))
    print("Mean Average Precision:  {:.2f}%, std: {:.2f}%".format(np.mean(all_avgprec_train) * 100, np.std(all_avgprec_train) * 100))
    print("Mean Recall@100:         {:.2f}%, std: {:.2f}%".format(np.mean(all_recall100_train) * 100, np.std(all_recall100_train) * 100))
    print("Mean ROC AUC:            {:.2f}%, std: {:.2f}%".format(np.mean(all_rocauc_train) * 100, np.std(all_rocauc_train) * 100))

# ======== Code used to generate the uids on a dataframe ============
from tools.data_loader import DataLoader
data_loader = DataLoader()
# df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'ecl_with_financial_tags.csv'), low_memory=False)

# exit()
# ==========================================================================

# add bankruptcy prediction split to numerical.csv
# df1 = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'ecl_with_financial_tags_uid.csv'), low_memory=False)
# df2 = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'numerical.csv'), low_memory=False)
# df2 = df2.merge(df1[['uid', 'bankruptcy_prediction_split']], on='uid', how='left')
# data_loader.save_dataset(df2, os.path.join(config.OUTPUT_DIR, "numerical_with_metadata.csv"))
# exit()

# ======== Code used merge finsim with ecl =======================================================================
# # df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'ecl_with_financial_tags.csv'), low_memory=False)
# df = pd.read_csv(os.path.join(config.DATA_DIR, 'ECL_filtered_subset_notext_uid.csv'), low_memory=False)
# df_fins = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'future_internet_fins.csv'))
#
# df['period_of_report'] = pd.to_datetime(df['period_of_report'], dayfirst=False)
# df['uid'] = df['cik'].astype(str) + '__' + df['period_of_report'].dt.strftime('%Y-%m-%d')
#
# df_subset = df[['uid', 'bankruptcy_prediction_split']]
# df_fins = df_fins.merge(df_subset, on='uid', how='left')
#
# output_path = os.path.join(config.OUTPUT_DIR, 'future_internet_fins_with_metadata.csv')
# df_fins.to_csv(output_path, index=False)

# ===============================================================================================================

def add_uids_to_dense_df(dense_df):
    dense_df['uid'] = df['cik'].astype(str) + '__' + pd.to_datetime(df['period_of_report']).dt.strftime('%Y-%m-%d')
    data_loader.save_dataset(df, os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags_uid.csv"))
    return dense_df

def add_missing_uids_to_dense_df(dense_df, sparse_df):
    # --- 1. Keep ONLY those rows in `dense` whose uid exists in `sparse`
    #     • Using .isin is vector-ised and avoids slow Python-level loops.
    dense_trimmed = dense_df[dense_df['uid'].isin(sparse_df['uid'])].copy()

    # --- 2. Identify uids present in `sparse` but absent from the trimmed `dense`
    uids_to_add = sparse_df.loc[~sparse_df['uid'].isin(dense_trimmed['uid']), ['uid', 'label']]

    # --- 3. Re-index the new rows so they have the same columns as `dense_trimmed`
    #     • Any columns other than 'uid' and 'label' become NaN (blank), as requested.
    rows_to_append = uids_to_add.reindex(columns=dense_trimmed.columns)

    # --- 4. Concatenate: original kept rows + new blank-filled rows
    merged = pd.concat([dense_trimmed, rows_to_append], ignore_index=True)

    return merged


# df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'future_internet_fins_with_metadata.csv'))
# dense_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'ecl_with_financial_tags_uid.csv'), low_memory=False)
# sparse_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'numerical.csv'), low_memory=False)
# df = df2[df2['uid'].isin(df['uid'])]
# df = df[df['uid'].isin(df2['uid'])]

# print("Dense df shape before merge: ", dense_df.shape)
# dense_df = add_missing_uids_to_dense_df(dense_df, sparse_df)
# print("Dense df shape after merge: ", dense_df.shape)
# data_loader.save_dataset(dense_df, os.path.join(config.OUTPUT_DIR, "numerical_dense_merged.csv"))
# exit()

# print("DFf shape before drop: ", df.shape)







def populate_dense_from_sparse(
    dense_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    xbrl_mapping: dict,
    *,
    meta_cols: tuple = ("uid", "label", "cik", "accessionNumber")
) -> pd.DataFrame:
    """
    Fill the NULL rows of *dense_df* with values taken from *sparse_df*.

    Parameters
    ----------
    dense_df : pd.DataFrame
        The “wide & dense” table that already has one column per standardised
        concept but still contains NaNs.
    sparse_df : pd.DataFrame
        The original “very‐wide & sparse” table coming straight from the SEC
        filings (one row per filing, one column per raw XBRL tag).
    xbrl_mapping : dict
        The JSON mapping that defines every standard concept, its candidate
        XBRL tags and the fuzzy-matching threshold.
    meta_cols : tuple[str], optional
        Column names that are **not** XBRL concepts (they should be ignored
        while building the tag list).  Extend this if your own dataframes
        contain additional metadata columns.

    Returns
    -------
    pd.DataFrame
        The same *dense_df* instance, updated in-place and also returned for
        convenience.
    """

    # ------------------------------------------------------------------
    # Pre-compute a flat view of the mapping so we don’t loop over the
    # same four sections for every single row.
    # ------------------------------------------------------------------
    section_maps = list(xbrl_mapping.values())   # 4 dicts (IS, BS, CF, EQ)

    i = 0
    na_rows = len(dense_df[dense_df["cik"].isna()])
    print(f"will parse {na_rows} rows")

    # Only iterate over rows that still need to be filled
    for idx, dense_row in dense_df[dense_df["cik"].isna()].iterrows():

        print(f"\rProgress: {(i/na_rows)*100}%", end='')
        i+=1
        uid = dense_row["uid"]
        s_row = sparse_df.loc[sparse_df["uid"] == uid]

        if s_row.empty:          # no counterpart → skip
            continue
        s_row = s_row.squeeze()  # convert 1-row DF → Series

        # --------------------------------------------------------------
        # 1) Build the list of *raw* tag names that actually have values
        #    in this sparse row (skip metadata columns and NaNs).
        # --------------------------------------------------------------
        concept_tag_list = [
            col for col in s_row.index
            if col not in meta_cols and pd.notna(s_row[col])
        ]

        # RapidFuzz only needs the tag strings; the 2nd argument in the
        # original signature (descriptions) is ignored, so we can pass [].
        matched = {}
        for section_map in section_maps:
            matched.update(
                match_concept_in_section(
                    section_map, concept_tag_list, []
                )
            )

        # --------------------------------------------------------------
        # 2) Write the matched values back into *dense_df*
        # --------------------------------------------------------------
        for std_name, raw_tag in matched.items():
            dense_df.at[idx, std_name] = s_row[raw_tag]

        # Also copy over the CIK (and any other meta fields you like)
        if "cik" in s_row.index:
            dense_df.at[idx, "cik"] = s_row["cik"]

    return dense_df

def normalize_new_rows(
    df: pd.DataFrame,
    *,
    eps_cols: tuple = ("earnings_per_share_basic",
                       "earnings_per_share_diluted",      # correct spelling
                       "earnings_per_share_dilluted"),    # in case of typo
) -> pd.DataFrame:
    """
    • Operates **only** on rows where ``cik`` is still NaN
    • Divides every numeric value by 1 000 000 and keeps up to six decimals
    • Skips the EPS columns listed in *eps_cols* (they stay at original scale)

    Returns the same dataframe modified in-place (and also returns it
    for fluent use, e.g. ``dense_df = normalize_new_rows(dense_df)``).
    """
    # ──────────────────────────────────────────────────────────────────────
    mask = df["cik"].isna()

    # columns eligible for scaling (exclude EPS + obvious metadata)
    cand_cols = [
        c for c in df.columns
        if c not in eps_cols and c not in ("uid", "label", "cik", "accessionNumber")
    ]

    # process each candidate column only if the masked block contains data
    for col in cand_cols:
        series = df.loc[mask, col]

        if series.notna().any():
            # 1️⃣ scale            2️⃣ format to ≤6 dp  3️⃣ back to float
            df.loc[mask, col] = (
                pd.to_numeric(series, errors="coerce")        # ensure numeric
                  .div(1_000_000)
                  .apply(lambda v:
                         float(f"{v:.6f}".rstrip('0').rstrip('.'))
                         if pd.notna(v) else v)
            )

    return df


# dense_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'dense_merged_filled.csv'), low_memory=False)
# dense_df = normalize_new_rows(dense_df)
# data_loader.save_dataset(dense_df, os.path.join(config.OUTPUT_DIR, "dense_merged_filled.csv"))
# exit()


import json


# with open(os.path.join('../tools', 'xbrl_mapping.json'), 'r') as file:
#     xbrl_mapping = json.load(file)


df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'dense_merged_filled.csv'), low_memory=False)
# dense_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'numerical_dense_merged.csv'), low_memory=False)
# sparse_df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'numerical.csv'), low_memory=False)
# dense_df = populate_dense_from_sparse(dense_df, sparse_df, xbrl_mapping)
#
# data_loader.save_dataset(dense_df, os.path.join(config.OUTPUT_DIR, "dense_merged_filled.csv"))
# exit()

# finsim_cols = ['current_assets', 'total_assets', 'cost_of_goods_sold',
#        'total_long_term_debt', 'depreciation_and_amortization', 'ebit',
#        'ebitda', 'gross_profit', 'inventory', 'total_current_liabilities',
#        'net_income', 'retained_earnings', 'total_receivables', 'total_revenue',
#        'market_value', 'total_liabilities', 'net_sales',
#        'total_operating_expenses', 'label']

columns_to_drop = [
    # 'uid',
    "bankruptcy_prediction_split",
    'company','period_of_report','gvkey','filing_date','year','accessionNumber','reportDateIndex',
    'datadate', 'filename', 'can_label', 'qualified', 'cik_year', 'gc_list',
    'bankruptcy_date_1','bankruptcy_date_2','bankruptcy_date_3','form','primaryDocument',
    'isXBRL'
]
# columns_to_drop = ["uid"]
df.dropna(subset=["bankruptcy_prediction_split"])
df.drop(columns=columns_to_drop, axis=1, inplace=True)
print("DFf shape after drop: ", df.shape)
print("cols remaining: ", df.columns)

data_loader.save_dataset(df, os.path.join(config.OUTPUT_DIR, "numerical.csv"))
exit()

df['bankruptcy_prediction_split'].fillna(value="train")
#
# numeric_cols = [
#    'revenues','operating_expenses', 'operating_income', 'net_income',
#    'earnings_per_share_basic', 'earnings_per_share_diluted',
#    'total_current_assets', 'total_noncurrent_assets', 'total_assets',
#    'total_current_liabilities', 'total_noncurrent_liabilities',
#    'total_liabilities', 'stockholders_equity', 'total_liabilities_equity',
#    'net_cash_from_operating_activities','net_cash_from_investing_activities',
#    'net_cash_from_financing_activities', 'cash','other_comprehensive_income'
# ]
#
#
# model=LogisticRegression(max_iter=10000, solver='liblinear', random_state=42)
# run_bankruptcy_experiments(model, df, df.columns, n_runs=5, smote=False)

model=XGBClassifier(eval_metric='logloss', random_state=42)
run_bankruptcy_experiments(model, df, df.columns, n_runs=5, drop_na=False, smote=False)

# model=AdaBoostClassifier(n_estimators=100, random_state=42)
# run_bankruptcy_experiments(model, df, df.columns, n_runs=5, smote=True)



# ------------------------------------------------------
# CatBoost
# catboost_model = CatBoostClassifier(
#     iterations=200,
#     learning_rate=0.05,
#     depth=6,
#     verbose=0,
#     random_state=42
# )
# run_bankruptcy_experiments(catboost_model, df, numeric_cols, n_runs=5)

# ------------------------------------------------------
# Stacking: Decision Tree + Random Forest → RidgeClassifier
# stacking_model = StackingClassifier(
#     estimators=[
#         ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
#         ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
#     ],
#     final_estimator=LogisticRegression(),
#     passthrough=True,
#     cv=5
# )
# run_bankruptcy_experiments(stacking_model, df, numeric_cols, n_runs=5)



# ------------------------------------------------------
# 1) Logistic Regression–style PyTorch model
# print("===Logistic regression===")
# torch_lr_model = ModelWrapper.GeneralizedTorchModel(
#     input_dim=len(numeric_cols),
#     architecture='logistic',
#     hidden_dim=0,      # not used in logistic
#     epochs=8000,
#     lr=1e-3,
#     # random_state=1
# )
# run_bankruptcy_experiments(torch_lr_model, df, numeric_cols, n_runs=5)

# 2) MLP–style PyTorch model
# print("===MLP===")
# torch_mlp_model = GeneralizedTorchModel(
#     input_dim=len(numeric_cols),
#     architecture='mlp',
#     hidden_dim=32,
#     epochs=20,
#     lr=1e-3,
#     random_state=42
# )
# run_bankruptcy_experiments(torch_mlp_model, df, numeric_cols, n_runs=5)

# You can easily add more architectures by expanding GeneralizedTorchModel.
