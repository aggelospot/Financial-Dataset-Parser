import pandas as pd
from datetime import datetime
import logging
import re
import os
import json
from fuzzywuzzy import fuzz
from tools import config
from tools.utils import extract_year_from_filename, match_concept_in_section, clean_cik, extract_accession_number_regex

logging.basicConfig(level=logging.INFO)

def get_unique_concepts_from_financials(financials_dir):
    """
    Parse all JSON files in the 'financials' directory (and its subdirectories),
    collecting unique 'concept' strings found under the 'bs', 'cf', and 'ic' sections
    of each file.

    If the concept contains a namespace prefix (e.g. 'nbbc:...'),
    we remove everything before the colon for a more standardized name.

    :param financials_dir: Path to the 'financials' directory.
    :return: A dictionary with keys 'bs', 'cf', 'ic' mapping to sorted lists of unique labels.
    """

    # Use sets to avoid duplicates
    unique_labels = {
        'bs': set(),
        'cf': set(),
        'ic': set()
    }

    # Walk through every file in 'financials' and its subdirectories
    for root, dirs, files in os.walk(financials_dir):
        print(f"\rCurrently processing : {root}", end='')
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)

                # Safely open and parse JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        data_sections = data.get('data', {})

                        # For each property (bs, cf, ic), gather unique concepts
                        for section in ['bs']:# , 'cf', 'ic']:
                            items = data_sections.get(section, [])
                            for item in items:
                                concept = item.get('concept')
                                if concept:
                                    # Split off namespace prefix if present
                                    if ':' in concept:
                                        concept = concept.split(':', 1)[1]
                                    unique_labels[section].add(concept)

                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Skip any file that is invalid JSON or has an unexpected structure
                        continue

    # Convert each set of labels into a sorted list for readability
    for section in unique_labels:
        unique_labels[section] = sorted(list(unique_labels[section]))

    return unique_labels



def parse_income_statement_matches2(financials_dir):
    """
    Parse all JSON files in 'financials_dir' and attempt to match each concept
    in the 'ic' section to one of three standardized items using fuzzy matching:
        1. earnings_per_share_diluted
        2. total_revenues
        3. operating_income

    Returns a pandas DataFrame with columns:
        - filename
        - earnings_per_share_diluted
        - total_revenues
        - operating_income
    """
    # Define the universal list of synonyms or near-synonyms for each concept
    # (feel free to expand or refine!)
    target_concepts = {
        "earnings_per_share_diluted": [
            "EarningsPerShareDiluted",
            "EarningsPerShareBasicAndDiluted",
            # "diluted earnings per share",
            # "DilutedShares",
            # "SharesDiluted"
            # "EarningsPerShareBasicAndDiluted"
            # "basic and diluted eps",
            # "basic and diluted earnings per share",
            # "eps diluted"
        ],
        "earnings_per_share_basic": [
            "EarningsPerShareBasic",
            # "EarningsPerShareBasicAndDiluted"
            # "earningspershare"
            # "NumberOfDilutedShares"
            # "basic and diluted eps",
            # "basic and diluted earnings per share",
            # "eps diluted"
        ],
        "total_revenues": [
            "Revenues",
            "SalesRevenueNet",
            # "RevenueFromContractWithCustomerExcludingAssessedTax" # if we include this, we need to sum it with other revenues
        ],
        "net_income": [
            "NetIncomeLoss",
            "ProfitLoss"
        ],
        "operating_income": [
            "OperatingIncomeLoss",
            "OperatingIncomeLossAfterEquityMethodInvestments",
            "income from operations",
            "operating profit"
        ]
    }

    # We will store results (one row per file) in a list of dicts
    results = []
    counter = 0

    # Read mapping from the config file
    xbrl_mapping = pd.read_json(config.XBRL_MAPPING_PATH)
    target_concepts = xbrl_mapping.get("IncomeStatement")


    for root, dirs, files in os.walk(financials_dir):
        # print("current root ", root)
        for filename in files:
            # print("current file: ", filename, counter)
            if counter > 113: break # return pd.DataFrame(results)
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        symbol = data.get("symbol")
                        quarter = data.get("quarter")
                        if quarter != "FY": continue
                        data_sections = data.get('data', {})
                        income_items = data_sections.get('ic', [])
                        symbol = data_sections.get('symbol', [])
                        print("symbol is ", symbol)

                        # Initialize a row for the DataFrame
                        row = {
                            "filename": filename,
                            # "earnings_per_share_diluted": None,
                            # "total_revenues": None,
                            # "operating_income": None
                        }
                        matched_ic_items = match_concept_in_section(target_concepts, income_items)
                        all_matches = {**row, **matched_ic_items}

                        # After scanning all concepts, add this row to results
                        results.append(all_matches)

                    except (json.JSONDecodeError, KeyError, TypeError):
                        counter += 1
                        print("error ")
                        # Malformed JSON or unexpected structure
                        continue
            counter+=1

    # Convert to a DataFrame
    df = pd.DataFrame(results)
    out_path = os.path.join(config.OUTPUT_DIR, 'financials_ic_tags.csv')
    print("Saving...")
    df.to_csv(out_path, index=False)
    return df


def extract_tags_for_year(submission_json: dict, target_year: int) -> list:
    """
    Walks through a flattened SEC-style submission JSON, where each key is a concept
    (e.g. 'AccountsPayableCurrent'). Each concept maps to:
      {
        "label": "...",
        "description": "...",
        "units": {
          "USD": [
            {
              "end": "2010-05-31",
              "val": 238466000,
              "accn": "...",
              "fy": 2011,
              "fp": "FY",
              "form": "10-K",
              "filed": "...",
              "frame": "...",
              ...
            },
            ...
          ],
          ...
        }
      }

    We look for fact entries whose 'fy' == target_year and whose 'form' == '10-K'
    and return them in a simplified list of dicts.
    """
    results = []

    for concept_name, concept_info in submission_json.items():
        # Fallback to concept_name if 'label' is missing
        label = concept_info.get("label", concept_name)

        # 'units' is typically a dict like {"USD": [ {...}, {...} ], "EUR": [ ... ], ...}
        units_dict = concept_info.get("units", {})
        # print("units dict", units_dict)
        # Loop through each unit
        for unit_name, fact_list in units_dict.items():
            # print("unit name, fact list: ", unit_name, fact_list)

            # fact_list is a list of individual fact dictionaries
            for fact_item in fact_list:
                # print(f"fact item ...  Form = {fact_item.get('form')}, {target_year} == {fact_item.get('fy')} ??")

                # Check if this fact matches our filter (year + form)
                if str(fact_item.get('fy')) in target_year and fact_item.get("form") in "10-K":
                    # print('Match')
                    # Build a simplified data dict
                    result = {
                        "label": label,
                        "concept": concept_name,
                        "unit": unit_name.lower(),
                        "value": fact_item.get("val")
                    }
                    results.append(result)

    # print("results before retuning: ", results)
    return results


def create_financials_dataframes(financials_dir):
    """
    Parses all JSON files under 'financials_dir', looking only for 10-K files
    (where top-level "quarter" == "FY"). For each such file, extracts the
    'bs', 'cf', and 'ic' items, using the 'concept' to name a column and
    storing the 'label' as that column's value. Returns three DataFrames.
    """

    # These lists will hold one dict-per-file (one "row" per file)
    bs_rows = []
    cf_rows = []
    ic_rows = []
    counter = 0

    for root_dir, dirs, files in os.walk(financials_dir):

        for filename in files:
            # print(f"\rCurrent folder: {root_dir} file: {filename}", end='')
            if filename.endswith('.json'):
                if counter > 5: break
                file_path = os.path.join(root_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        # Skip any file that isn't valid JSON
                        continue

                    # Check if it's a 10-K
                    if data.get('quarter') == 'FY':
                        # Prepare dicts that will become rows in each DF
                        # bs_dict = {}
                        # cf_dict = {}
                        ic_dict = {}

                        # Optionally store the file name, symbol, etc., if you want them in the DF:
                        # symbol = data.get('symbol', '')
                        # bs_dict['filename'] = filename
                        # bs_dict['symbol'] = symbol
                        # (Repeat for cf_dict, ic_dict if desired.)

                        data_sections = data.get('data', {})

                        # For each statement type, create columns keyed by concept (w/o namespace)
                        for section_name, row_dict in [('ic', ic_dict)]: #, [('bs', bs_dict), ('cf', cf_dict), ('ic', ic_dict)]:
                            items = data_sections.get(section_name, [])
                            for item in items:
                                concept = item.get('concept', '')
                                value = item.get('value', None)
                                if concept:
                                    # Strip off any prefix like "abc:" -> we keep only after the first colon
                                    if ':' in concept:
                                        concept = concept.split(':', 1)[1]
                                    # Store label in the dict
                                    row_dict[concept] = value

                        # Append these “rows” to our overall lists
                        # bs_rows.append(bs_dict)
                        # cf_rows.append(cf_dict)
                        ic_rows.append(ic_dict)
                        # print("IC ROWS ", ic_rows)
                        # aa = pd.DataFrame(ic_rows)
                        # print("bs_rows ", len(aa.columns), aa.columns)
                        counter+=1

    # Convert to DataFrames
    # df_bs = pd.DataFrame(bs_rows)
    # df_cf = pd.DataFrame(cf_rows)
    df_ic = pd.DataFrame(ic_rows)

    # # Print info on each DF
    # print("\n--- Balance Sheet (bs) ---")
    # print(df_bs.info())
    # print(df_bs.isna().sum())

    print("\n--- Cash Flow (cf) ---")
    # print(df_cf.info())
    # print(df_cf.isna().sum())
    #
    print("\n--- Income Statement (ic) ---")
    print(df_ic.info())
    print(df_ic.isna().sum())

    return df_ic # df_bs, df_cf, df_ic


def parse_financial_section(
        df_info,
        financials_dir,
        section='ic'
):
    """
    Given a DataFrame (df_info) that has at least 'year' and 'accessionNumber' columns,
    build a path to each JSON file using: {financials_dir}/{year}/{accessionNumber}.json

    Then, parse *only* the specified section (bs, cf, or ic) for each file,
    storing the 'label' values in columns named by the 'concept'.
    Only files with 'quarter' == 'FY' (10-K) are included.

    Returns a DataFrame, one row per file, columns corresponding
    to the 'concept' fields in that section.
    """
    rows = []
    counter = 0
    df_size = len(df_info.index)
    files_missing = 0
    files_missing_df = pd.DataFrame()
    files_missing_df['file_path'] = None
    files_missing_df['year'] = None
    files_missing_df['cik'] = None
    print("df head at start ", df_info.head(4))

    for _, row in df_info.iterrows():
        print(f"\rCurrent CIK: {row['cik']}. {counter} / {df_size}. Files missing: {files_missing}", end='')
        counter+=1
        year_val = str(extract_year_from_filename(row['filename']))
        if int(year_val) < 2013: continue
        accession_num = str(extract_accession_number_regex(row['filename']))


        # Construct the JSON path
        file_path = os.path.join(financials_dir, year_val, f"{accession_num}.json")

        if not os.path.isfile(file_path):
            # print("row is ", row)
            # print("year is ", year_val)
            # print("File doesnt exist: ", file_path)
            files_missing+=1
            files_missing_df.loc[len(files_missing_df)] = [file_path, year_val, row['cik']]
            # If the file doesn't exist, skip
            continue

        # Attempt to parse the JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError):
            print("error while loading file")
            print("files missing names ", )
            # Skip if we cannot open or parse properly
            continue

        # Check if this is a 10-K
        if data.get('quarter') == 'FY':
            # Prepare a dict for this file's row
            row_dict = {
                'year': year_val,
                'accessionNumber': accession_num
            }

            # Extract the items for the single requested section
            section_items = data.get('data', {}).get(section, [])
            for item in section_items:
                concept = item.get('concept', '')
                label = item.get('label', None)

                # Strip off namespace prefix if present (e.g. "abc:Something")
                if ':' in concept:
                    concept = concept.split(':', 1)[1]

                # You could store label or value; here we store label
                row_dict[concept] = label

            rows.append(row_dict)

    # Build a DataFrame out of all the row dictionaries
    df_section = pd.DataFrame(rows)

    # (Optional) Inspect missingness or summary info
    print(f"\n--- Summary for section: {section} ---")
    print(df_section.info())
    null_percentages = (df_section.isna().sum().sort_values(ascending=True)*100)/df_size
    null_percentages = null_percentages[null_percentages < 30]
    print("Null Percentages: \n", null_percentages)
    print(files_missing_df.head())
    print("Rows with missing files per year: \n", files_missing_df["year"].value_counts().sort_values(ascending=True))

    return df_section



def process_ecl_with_local_sec_data_test(df: pd.DataFrame ) -> pd.DataFrame: # xbrl_mapping: dict
    """
    Combines financials with local data
    """
    # TODO: Refractor and move to sec data parser

    # Caches loaded SEC data so we only read each CIK's JSON once.
    sec_data_cache = {}
    xbrl_mapping = pd.read_json(config.XBRL_MAPPING_PATH)

    # Create new columns in df for each “universal” item we want to store.
    for statement_label, items_map in xbrl_mapping.items():
        for universal_item in items_map.keys():
            if universal_item not in df.columns:
                df[universal_item] = None

    target_concepts = xbrl_mapping.get("IncomeStatement")

    """ Retrieve the appropriate companyfacts json """
    for idx, row in df.iterrows():
        current_cik = clean_cik(str(row.get('cik', '')))
        # if idx > 12: return df

        # TODO: test changing this with extract_year_from_filename
        cik_year_str = str(row.get('cik_year', ''))
        cik_year = cik_year_str.split('__')[-1] # if '__' in cik_year_str else cik_year_str

        # Load or retrieve SEC facts from cache
        if current_cik not in sec_data_cache:

            sec_file_name = f"CIK{current_cik}.json"
            sec_file_path = os.path.join(config.SEC_DATA_DIR, sec_file_name)

            if os.path.exists(sec_file_path):
                with open(sec_file_path, 'r') as sec_file:
                    try:
                        sec_data_cache[current_cik] = json.load(sec_file)
                    except json.JSONDecodeError:
                        logging.error(f"Could not decode JSON for {sec_file_path}")
                        sec_data_cache[current_cik] = None
            else:
                logging.warning(f"SEC data file not found for CIK {current_cik}")
                sec_data_cache[current_cik] = None
        sec_data = sec_data_cache[current_cik]

        """ Proceed to match items """
        if sec_data:
            facts_data = sec_data.get('facts', {})
            us_gaap_data = facts_data.get('us-gaap', {})
            dei_data = facts_data.get('dei', {})

            # Combine us-gaap and dei to search for keys in one place
            all_facts = {**us_gaap_data, **dei_data}

            # print("Calling with: ", str(cik_year))
            res= extract_tags_for_year(all_facts, str(cik_year))
            matched_concepts = match_concept_in_section(target_concepts, res)
            # print("Returns ", match_concept_in_section(target_concepts, res))


            for concept in matched_concepts:
                # print("concept:", )
                df.at[idx, concept] = matched_concepts.get(concept)

    return df


def match_concept_in_section(target_concepts, section_items):
    row = {}
    for item in section_items:
        concept = item.get('concept', '')

        # Remove any namespace prefix
        # e.g. 'nbhc:NetIncomeLoss' -> 'NetIncomeLoss'
        if ':' in concept:
            concept = concept.split(':', 1)[1]

        # Convert to lowercase for easier matching
        concept_lower = concept.lower()


        # For each "universal" item, see if there's a fuzzy match
        for col_name, synonyms in target_concepts.items():
            best_score = 0
            # Evaluate this concept against each synonym
            for synonym in synonyms:
                # print(f"mathing synonym: {synonym} of {synonyms}")
                # Use partial_ratio to allow substring matches
                score = fuzz.partial_ratio(synonym.lower(), concept_lower)
                if score > best_score:
                    best_score = score

            # If the best match is above a threshold, store
            if best_score >= 80:
                if col_name in row:
                    prev_score = fuzz.partial_ratio(row[col_name].lower(), concept_lower)
                    if best_score > prev_score:
                        # print(f"REPLACED {row[col_name]} with {concept}, prevScore: {prev_score}, new score: {best_score}")
                        row[col_name] = concept
                else:
                    # print(f"Matched {col_name} with {concept}. Score: {best_score}")
                    row[col_name] = concept

    return row
