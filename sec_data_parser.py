# Python packages
import json
import time
import logging
import os
from os import path
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# Our packages
from tools.utils import clean_cik, get_file_line_count, get_column_list_from_json, extract_year_from_filename
from tools import config
import db_connection


# Configure logging
logging.basicConfig(level=logging.INFO)


def process_ecl_with_local_sec_data(input_file_path, output_file_path, sec_data_dir):
    """
    Processes the ECL file, retrieves SEC data from local JSON files for each row,
    and adds 'us-gaap' and 'dei' variables to each row based on the matching date.
    Writes each updated row to the output file immediately.
    """
    row_counter = 0
    previous_cik = None
    sec_data = None

    # Ensure the error directory exists
    error_dir = './outputs/errors'
    os.makedirs(error_dir, exist_ok=True)
    error_file_path = os.path.join(error_dir, 'errors_log.txt')

    # Get the total number of rows in the input file to determine how many rows are remaining to be parsed.
    total_rows = get_file_line_count(input_file_path)

    # Open the error log and input files
    with open(error_file_path, 'w') as error_file:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                try:
                    row_counter += 1
                    # if row_counter > 100: break  # Break condition for testing
                    row = json.loads(line.strip())

                    # Extract the year from the 'cik_year' field
                    cik_year_str = extract_year(row.get('cik_year', ''))
                    cik_year = cik_year_str.split('__')[-1]  # Extract the year part (e.g., "2015")

                    if cik_year is None:
                        logging.warning(f"Invalid date format in row {row_counter}: {cik_year_str}")
                        continue

                    current_cik = row.get('cik', '')
                    cik_cleaned = clean_cik(current_cik)

                    # Display the current CIK being processed
                    print(f"\rProcessing CIK: {cik_cleaned} | Row: {row_counter} / {total_rows}", end='')

                    if previous_cik != cik_cleaned:
                        # Load the SEC data from local JSON file
                        sec_file_name = f"CIK{cik_cleaned}.json"
                        sec_file_path = os.path.join(sec_data_dir, sec_file_name)
                        if os.path.exists(sec_file_path):
                            with open(sec_file_path, 'r') as sec_file:
                                sec_data = json.load(sec_file)
                                error_file.write(f"CIK{cik_cleaned}: \n")
                        else:
                            error_file.write(f"CIK{cik_cleaned}:\n  - SEC data file not found. \n")
                            sec_data = None

                    previous_cik = cik_cleaned

                    num_variables_matched = 0  # Initialize counter for variables matched

                    if sec_data:
                        # Extract variables under 'us-gaap' and 'dei' keys in 'facts'
                        facts_data = sec_data.get('facts', {})
                        us_gaap_data = facts_data.get('us-gaap', {})
                        dei_data = facts_data.get('dei', {})

                        # Combine 'us-gaap' and 'dei' data
                        all_facts = {**us_gaap_data, **dei_data}

                        total_variables = len(all_facts)

                        # For each variable, find the value matching the ecl_date
                        for key, value in all_facts.items():
                            units = value.get('units', {})
                            data_points = []
                            for unit_key, unit_values in units.items():
                                data_points.extend(unit_values)
                            # Now data_points is a list of dictionaries
                            # Find the data point that matches the ecl_date or closest date within threshold
                            matching_val = None
                            for data_point in data_points:
                                """ Note: I used the 'in' keyword instead of '==' to match columns with strings like '10-K/A'. """
                                if ('10-K' not in data_point.get('form')): continue # Only check the data_points from 10-K reports.

                                # New method: Only match the fiscal year.
                                fiscal_year = data_point.get('fy', '')

                                if fiscal_year:
                                    if str(cik_year) == str(fiscal_year):  # Exact year match
                                        matching_val = data_point.get('val')
                                        break  # Exact match found


                            if matching_val is not None:
                                # Add the variable and its value to the row
                                row[key] = matching_val
                                num_variables_matched += 1

                        # If no variables were matched, log the period of report
                        if num_variables_matched == 0:
                            error_file.write(
                                f"  - Period '{cik_year}': No data found.\n")
                        #     f"CIK {cik_cleaned} - Period '{ecl_date_str}': No data found.\n")
                        else:
                            percent_matched = (num_variables_matched / total_variables) * 100 if total_variables > 0 else 0
                            error_file.write(f"  - Period '{cik_year}': {num_variables_matched}/{total_variables} variables matched ({percent_matched:.2f}%)\n")
                    else:
                        # SEC data not found for this CIK
                        pass

                    # Write the updated row to the output file
                    json.dump(row, output_file)
                    output_file.write('\n')

                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON on row {row_counter}")
                    continue

    # Clear the progress bar line after completion
    print("\rProcessing complete.                      ")
    print(f"\nTotal rows processed: {row_counter}")
    print(f"Error log written to: {error_file_path}")

def find_custom_revenue_concepts(calc_linkbase_path, target_concepts):
    import xml.etree.ElementTree as ET
    """
    target_concepts = list of standard revenue tags, e.g. ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", ...]
    """

    tree = ET.parse(calc_linkbase_path)
    root = tree.getroot()

    # Step 1: gather all <loc> elements -> map from label -> concept
    label_to_concept = {}
    for loc in root.findall('{http://www.xbrl.org/2003/linkbase}loc'):
        label = loc.get('{http://www.w3.org/1999/xlink}label')  # e.g. "us-gaap_Revenue_6369..."
        href  = loc.get('{http://www.w3.org/1999/xlink}href')   # e.g. "...#us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax"
        # parse the actual concept name from href
        # It's typically: "...#us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax"
        # or "...#air_CustomRevenueTag"

        # Extract the portion after the '#' (the ID part).
        # Then replace '_' with ':' if needed, or build a final string as "us-gaap:RevenueFrom..."
        # Often you can just treat it as "us-gaap_RevenueFrom..."
        concept_id = href.split('#')[-1]
        print("concept ids ", concept_id)
        # Might need additional string logic to produce the final concept name "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax"
        if '_' in concept_id and concept_id.startswith('us-gaap'):
            # The standard approach is to parse out up to the second underscore for the actual concept name
            # Or simpler: concept_id = concept_id.replace('_', ':', 1)
            concept_id = concept_id.replace('_', ':', 1)
            print("new concept id ", concept_id)
            pass

        # For demonstration, let's keep it simple:
        label_to_concept[label] = concept_id
        print("label_to_concept[label] = ", label_to_concept[label])

    # Step 2: find the <calculationLink> for the income statement
    # E.g. role="http://www.aarcorp.com/role/StatementConsolidatedStatementsOfIncome"
    # Then gather <calculationArc> from that link
    discovered = set()  # Will store newly discovered concepts

    for calc_link in root.findall('{http://www.xbrl.org/2003/linkbase}calculationLink'):
        role_val = calc_link.get('{http://www.w3.org/1999/xlink}role')

        print("role val ", role_val)
        # Check if it's the income statement link
        # For example, you might look for 'Income' in the role. Or you might directly compare the URI
        if 'StatementConsolidatedStatementsOfIncome' not in role_val:
            print("statement not in role val")
            continue

        # Now look for <calculationArc>
        arcs = calc_link.findall('{http://www.xbrl.org/2003/linkbase}calculationArc')
        print("all arcs: ", arcs)
        for arc in arcs:
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label   = arc.get('{http://www.w3.org/1999/xlink}to')

            from_concept = label_to_concept.get(from_label, "")
            to_concept   = label_to_concept.get(to_label, "")

            # If either side is one of our known revenue tags, the other side is relevant
            if from_concept in target_concepts:
                discovered.add(to_concept)
            if to_concept in target_concepts:
                discovered.add(from_concept)

    return list(discovered)
def process_ecl_with_local_sec_data_new(df: pd.DataFrame ) -> pd.DataFrame: # xbrl_mapping: dict
    """
    Refactored version
    """

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

            # For each universal item in the XBRL mapping, check if any synonyms match
            for statement_label, items_map in xbrl_mapping.items():
                for universal_item, synonyms_list in items_map.items():
                    if pd.notnull(df.at[idx, universal_item]):
                        continue # If we’ve already assigned a value for this row, skip re-check

                    matched_value = None

                    # Try each synonym in the company's facts
                    for synonym in synonyms_list:
                        fact_object = all_facts.get(synonym)
                        if not fact_object:
                            continue

                        units = fact_object.get('units', {})
                        # Flatten all data points across all reported units
                        data_points = []
                        for unit_values in units.values():
                            data_points.extend(unit_values)

                        # Iterate over data_points to find a 10-K for the correct FY
                        for dp in data_points:
                            form_type = dp.get('form', '')
                            fiscal_year = dp.get('fy', '')

                            # Only match on 10-K related forms (ex. "10-K", "10-K/S",...)
                            if '10-K' not in form_type:
                                continue

                            # Compare the row's year to the data_point's fy
                            if str(fiscal_year) == str(cik_year):
                                matched_value = dp.get('val')
                                break

                        if matched_value is not None:
                            break

                    # If any synonym matched, assign it to the row
                    df.at[idx, universal_item] = matched_value
    return df


def post_process_ecl(input_file_path, output_file_path):
    """
    Reads the combined ECL dataset and filters out rows that do not contain any financial variables.
    Writes the filtered dataset to a new file.

    Parameters:
    - input_file_path: Path to the combined ECL dataset file (ecl_combined.json).
    - output_file_path: Path to the output file for the filtered dataset.
    """
    # Define the original ECL keys to identify added financial variables
    original_keys = set([
        'bankruptcy_date_1', 'label', 'bankruptcy_date_2', 'filing_date', 'datadate',
        'bankruptcy_date_3', 'opinion_text', 'item_7', 'bankruptcy_prediction_split',
        'cik', 'company', 'period_of_report', 'cik_year', 'qualified', 'gc_list',
        'can_label', 'filename', 'gvkey'
    ])

    current_row = 0
    rows_with_financial_vars = 0
    total_rows = get_file_line_count(input_file_path)

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            try:
                print(f"\rProcessing file... Progress: {current_row/total_rows:.0%}", end='')
                row = json.loads(line.strip())
                current_keys = set(row.keys())

                # Identify financial variables by subtracting original keys from all keys
                financial_vars = current_keys - original_keys

                if financial_vars:
                    # Row contains financial variables, write it to the output file
                    json.dump(row, output_file)
                    output_file.write('\n')
                    rows_with_financial_vars += 1
                else:
                    # Row does not contain financial variables, skip it
                    pass

                current_row += 1

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON on line {total_rows + 1}")
                continue

    print(f"Total rows processed: {total_rows}")
    print(f"Rows with financial variables: {rows_with_financial_vars}")
    print(f"Rows without financial variables (omitted): {total_rows - rows_with_financial_vars}")


def drop_sec_variables_by_null_percentage(columns_stats_df, ecl_companyfacts, output_filename=config.POST_PROCESSED_DATASET_FILEPATH, max_null_percentage=22.0):
    """
    Reads the column statistics file, and drops all SEC variables that don't meet a certain max_null_percentage threshold.
    Writes the resulting DF to a file.

    :param columns_stats_df: The 'column_statistics' dataframe
    :param ecl_companyfacts: The ECL dataset with added SEC variables
    :param max_null_percentage: The threshold.
    """

    selected_columns = columns_stats_df[columns_stats_df['percentage_nulls'] <= max_null_percentage]['column'].tolist()
    print("Selected columns: ", selected_columns)
    numeric_columns = ecl_companyfacts[selected_columns].select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns: ", numeric_columns)


    # Combine numeric columns with original columns
    original_columns = ecl_companyfacts[get_column_list_from_json(config.ECL_FILE_PATH)]
    combined_dataset = pd.concat([original_columns, ecl_companyfacts[numeric_columns]], axis=1)

    # Drop rows where any of the selected numeric columns are null
    print("Shape before dropping NAs: ", combined_dataset.shape)
    cleaned_combined_dataset = combined_dataset.dropna(subset=numeric_columns)
    print("Shape after dropping NAs: ", cleaned_combined_dataset.shape)

    # Write the cleaned dataset to a CSV file
    cleaned_combined_dataset.to_csv(output_filename, index=False)

    print(f"Post processed dataset written to: {output_filename}")


def add_accession_and_form(df):
    """
    For each row in df, uses `df['cik']` and `df['period_of_report']` to look up
    the corresponding 'accessionNumber' and 'form' from the
    SEC submissions JSON file in `submissions_folder_path`.

    Parameters
    ----------
    df : pd.DataFrame
        A df which must have columns 'cik' and 'period_of_report'.

    Returns
    -------
    pd.DataFrame
        The same dataframe but with added columns 'accessionNumber' and 'form'.
    """
    print(df.head(1))
    # Ensure columns exist so we can store results
    if 'accessionNumber' not in df.columns:
        df['accessionNumber'] = None
    if 'form' not in df.columns:
        df['form'] = None

    for idx, row in df.iterrows():
        cik_str = clean_cik(row['cik'])
        period = row['period_of_report']

        # Build path to the JSON file for this CIK
        cik_file_name = f"CIK{cik_str}.json"
        cik_json_path = os.path.join(config.SEC_SUBMISSIONS_DIR, cik_file_name)

        # If the file doesn't exist, skip this row
        # print("Attempting to read file: ", cik_json_path, " -- ", os.path.isfile(cik_json_path))
        if not os.path.isfile(cik_json_path):
            print("File not found: ", cik_json_path)
            continue

        try:
            with open(cik_json_path, 'r') as f:
                print("File read.")
                data = json.load(f)

            # print("Data found: ", data)
            # 'filings' -> 'recent' is a dict of parallel arrays.
            # We want to find the index i where reportDate[i] = period_of_report
            report_dates = data['filings']['recent']['reportDate']

            # Check if our period_of_report is in the JSON's reportDate list
            if period in report_dates:
                i = report_dates.index(period)

                # Retrieve accessionNumber and form from the same index
                accession_numbers = data['filings']['recent']['accessionNumber']
                forms = data['filings']['recent']['form']

                df.at[idx, 'accessionNumber'] = accession_numbers[i]
                df.at[idx, 'form'] = forms[i]

        except Exception as e:
            # You could log the error if you want to debug
            # print(f"Error reading {cik_json_path}: {e}")
            pass

    return df


def add_report_date_index(df):

    #TODO: refractor this file and create the submissions parser file. 
    import glob
    """
    For each row in df, uses 'cik' and 'period_of_report' to look up
    the position (index) of that date in the JSON's accessionNumber list,
    and stores relevant fields (form, primaryDocument, isXBRL) in df.

    If a match is not found in the main file CIKxxxxx.json, it checks any
    additional split files (CIKxxxxx.json-submissions-001, etc.) for a match.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset, which must have columns 'cik' and 'period_of_report'.
    submissions_folder_path : str
        Path to the folder containing the SEC CIK JSON files, e.g. 'CIK0000012345.json'.

    Returns
    -------
    pd.DataFrame
        The same dataframe with (potentially) added or updated columns:
        'reportDateIndex', 'form', 'primaryDocument', 'isXBRL'.
    """

    # Ensure the new column exists to store the index
    if 'reportDateIndex' not in df.columns:
        df['reportDateIndex'] = None

    print("Retrieving metadata for all rows....")

    for idx, row in df.iterrows():
        cik_str = clean_cik(row['cik'])

        # The main JSON file
        main_file = f"CIK{cik_str}.json"
        main_path = os.path.join(config.SEC_SUBMISSIONS_DIR, main_file)

        match_idx = None  # We'll set this if we find a match

        # Attempt to open the main file
        if os.path.isfile(main_path):
            try:
                with open(main_path, 'r') as f:
                    data = json.load(f)

                acc_numbers = data['filings']['recent']['accessionNumber']
                # Try to match accessionNumber in the main file
                for i, acc_number in enumerate(acc_numbers):
                    if acc_number == row['accessionNumber']:
                        print("Matched ", acc_number)
                        match_idx = i
                        # Update relevant fields
                        df.at[idx, 'form'] = data['filings']['recent']['form'][i]
                        df.at[idx, 'primaryDocument'] = data['filings']['recent']['primaryDocument'][i]
                        df.at[idx, 'isXBRL'] = data['filings']['recent']['isXBRL'][i]
                        break
            except Exception as e:
                print(f"Error reading main file {main_file}: {e}")

        else:
            print(f"Main file not found for CIK {cik_str}: {main_path}")

        # If we did NOT find a match in the main file, look in the split files
        if match_idx is None:
            # Look for any files matching CIKxxxxx-submissions-*
            split_file_pattern = os.path.join(
                config.SEC_SUBMISSIONS_DIR,
                f"CIK{cik_str}-submissions-*"
            )
            split_files = glob.glob(split_file_pattern)
            print("Split files for this cik: ", split_files)

            for split_path in split_files:
                try:
                    with open(split_path, 'r') as sf:
                        data = json.load(sf)

                    # The split files have top-level arrays for accessionNumber, form, etc.
                    acc_numbers = data.get('accessionNumber', [])
                    # Try to match
                    for i, acc_number in enumerate(acc_numbers):
                        if acc_number == row['accessionNumber']:
                            match_idx = i
                            # Update relevant fields from the split file
                            df.at[idx, 'form'] = data['form'][i]
                            df.at[idx, 'primaryDocument'] = data['primaryDocument'][i]
                            df.at[idx, 'isXBRL'] = data['isXBRL'][i]
                            break
                    if match_idx is not None:
                        # If we found it in this file, stop searching further split files
                        break
                except Exception as e:
                    print(f"Error reading split file {split_path}: {e}")

            if match_idx is None:
                print(f"No match for {row['company']} , {row['year']} in main or split files.")

    return df


def fetch_calculation_linkbases(df):
    """
    Fetches XBRL calculation linkbase files from the SEC for each row in df.
    Expects df to have 'cik' and 'accessionNumber' columns.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame with columns ['cik', 'accessionNumber'] (others are ignored).
    """
    # Create the output directory if it doesn't already exist
    os.makedirs("./data/calc_linkbases", exist_ok=True)

    # SEC recommends providing a descriptive User-Agent in HTTP requests
    headers = {
        "User-Agent": "MyAppName/1.0 (email@example.com)"
    }

    base_url = "https://www.sec.gov/Archives/edgar/data"

    total_rows = df.shape[0]
    # df = df.copy()  # To avoid SettingWithCopyWarning.......
    if 'calc_linkbase' not in df.columns:
        df['calc_linkbase'] = None

    for idx, row in df.iterrows():
        # if idx > 100: break

        # if row['calc_linkbase'] != None:
        #     print("Linkbase already exists")
        #     continue

        cik = str(clean_cik(row["cik"]))
        if row["accessionNumber"]:
            accession_no_dashes = row["accessionNumber"].replace("-", "")
        else:
            continue  # skip rows without an accession number

        # Build the index.json URL
        index_json_url = f"{base_url}/{cik}/{accession_no_dashes}/index.json"

        print(f"Processing year {row['period_of_report']} of CIK: {cik}. Current Row: {idx}")

        # Fetch index.json
        try:
            r = requests.get(index_json_url, headers=headers, timeout=10)
            r.raise_for_status()  # Raises if the response wasn't successful
        except requests.RequestException as e:
            print(f"[Error] Could not retrieve {index_json_url}: {e}")
            continue

        # Throttle to 5 requests/second
        time.sleep(0.2)

        # Parse the JSON to find the _cal.xml file
        try:
            data = r.json()
        except ValueError as e:
            print(f"[Error] Could not parse JSON from {index_json_url}: {e}")
            continue

        items = data.get("directory", {}).get("item", [])
        cal_filename = None

        for item in items:
            name = item.get("name", "").lower()
            if "_cal.xml" in name:
                cal_filename = item["name"]
                break

        if not cal_filename:
            print(f"[Info] No '_cal.xml' file found for {index_json_url}")
            continue

        # Save the file if the linkbases directory, if it doesn't already exist.
        save_path = os.path.join("./data/calc_linkbases", cal_filename)
        # if os.path.exists(save_path):
            # df['calc_linkbase'][idx] = cal_filename # Add the filename to the appropriate row.
            # print("set new linkbase: ", df['calc_linkbase'][idx])
            # print("[Info] File already present, skipped.")
            # continue

        # # Throttle to 5 requests/second
        # time.sleep(0.2)

        # Build the final URL and fetch the _cal.xml file
        cal_url = f"{base_url}/{cik}/{accession_no_dashes}/{cal_filename}"
        if not os.path.isfile(save_path):
            try:
                cal_resp = requests.get(cal_url, headers=headers, timeout=10)
                cal_resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[Error] Could not retrieve {cal_url}: {e}")
                continue

            # Save the file under ./data/calc_linkbases/
            with open(save_path, "wb") as f:
                f.write(cal_resp.content)

        print("About to save: ", cal_filename)
        df.at[idx, 'calc_linkbase'] = cal_filename
        # df['calc_linkbase'][idx] = cal_filename
        print(f"[Success] Saved calculation linkbase to {save_path}. Filename in df: {df['calc_linkbase'][idx]}")
    return df



def retrieve_sec_tags_and_values(ecl, data_loader):
    conn = db_connection.create_connection()
    print("created connection")
    try:
        with open(config.XBRL_MAPPING_PATH, 'r') as file:
            xbrl_mapping = json.load(file)

        tag_list = []
        for section in ["IncomeStatement", "BalanceSheet", "CashFlow", "StatementOfStockholdersEquity"]:
            for key, value in xbrl_mapping[section].items():
                if key not in tag_list:  # Avoid duplicates
                    tag_list.append(key)
        for tag in tag_list:
            ecl[tag] = None

        results = []

        for idx, row in ecl.iterrows():
            print(f"\rCurrent row: {idx}", end='')
            metadata_row = {
                "accession_number": row['accessionNumber'],
                "isXBRL": row['isXBRL']
            }
            if row['isXBRL'] == 0: continue
            result = db_connection.retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'IS', [0,4])
            target_concepts = xbrl_mapping.get("IncomeStatement")
            matched_is_items = db_connection.match_concept_in_section(target_concepts, result)

            result = db_connection.retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'BS', 0)
            target_concepts = xbrl_mapping.get("BalanceSheet")
            matched_bs_items = db_connection.match_concept_in_section(target_concepts, result)

            result = db_connection.retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'CF', [0, 4])
            target_concepts = xbrl_mapping.get("CashFlow")
            matched_cf_items = db_connection.match_concept_in_section(target_concepts, result)

            result = db_connection.retrieve_statement_taxonomies_by_accession_number(row['accessionNumber'], 'EQ', [0, 4])
            target_concepts = xbrl_mapping.get("StatementOfStockholdersEquity")
            matched_eq_items = db_connection.match_concept_in_section(target_concepts, result)


            metadata_row = {**metadata_row, **matched_is_items, **matched_bs_items, **matched_cf_items, **matched_eq_items}
            all_matches = {**matched_is_items, **matched_bs_items, **matched_cf_items, **matched_eq_items}

            # if all_matches:
            for tag in tag_list:
                tag_synonym = all_matches.get(tag)
                if tag_synonym is not None:
                    ecl.at[idx, tag] = db_connection.get_tags_value_by_accession_number(connection=conn, accession_number=row['accessionNumber'], tag=tag_synonym)

            results.append(metadata_row)

        result_df = pd.DataFrame(results)
        # print("Resulting df: \n", result_df.head(1000))
        # print("Resulting ecl df \n", ecl.head(100))


        data_loader.save_dataset(result_df, os.path.join(config.OUTPUT_DIR, "tags.csv"))
        data_loader.save_dataset(ecl, os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv"))

    finally:
        print("attempting to close connection")
        db_connection.close_connection(conn)