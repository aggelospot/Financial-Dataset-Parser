import json
import requests
import time
import logging
import os
import pandas as pd
from tools import config

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the User-Agent header as per SEC's requirements
HEADERS = {
    'User-Agent': 'MyAppName/1.0 (myemail@example.com)'
}

def clean_cik(cik_value):
    """
    Cleans and formats the CIK value to a 10-digit string with leading zeros.
    """
    cik_value = str(cik_value).split('.')[0]  # Remove decimal point if present
    cik_value = cik_value.strip()
    cik_value = cik_value.zfill(10)  # Pad with leading zeros to make it 10 digits
    return cik_value

def get_sec_data(cik_value):
    """
    Retrieves data from the SEC API for a given CIK value.
    """
    sec_api_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_value}.json"
    try:
        response = requests.get(sec_api_url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            logging.warning(f"Failed to retrieve data for CIK {cik_value}: HTTP {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.error(f"Request exception for CIK {cik_value}: {e}")
        return None


def getCompanyFacts(cik_value):
    """
    Retrieves data from the SEC API for a given CIK value.
    """
    cik_value = clean_cik(cik_value)
    sec_api_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_value}.json"
    print(f"Processing URL: {sec_api_url}", end='')

    try:
        response = requests.get(sec_api_url, headers=HEADERS)
        if response.status_code == 200:
            logging.warning(f"{cik_value} Retrieved. HTTP: {response.status_code}")
            return response.json()
        else:
            logging.warning(f"Failed to retrieve data for CIK {cik_value}: HTTP {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.error(f"Request exception for CIK {cik_value}: {e}")
        return None


def process_ecl_file(file_path, num_rows=2):
    """
    Processes the ECL file, retrieves SEC data for the first num_rows rows,
    and adds 'us-gaap' variables to each row.
    """
    updated_rows = []
    row_counter = 0
    previous_cik = 0
    sec_data = {}

    with open(file_path, 'r') as f:
        for line in f:
            if row_counter >= num_rows:
                break
            try:
                row = json.loads(line.strip())

                # Get date and cik key for this row
                ecl_date = row.get('period_of_report', '')
                current_cik = row.get('cik', '')

                if (previous_cik != current_cik):
                    sec_data = getCompanyFacts(current_cik)

                previous_cik = current_cik


                # Display the current CIK being processed
                print(f"\rParsing: {current_cik}", end='')
                if sec_data:
                    # Extract all variables under the 'us-gaap' key in 'facts'
                    us_gaap_data = sec_data.get('facts', {}).get('us-gaap', {})
                    # Add each 'us-gaap' variable to the row data
                    for key, value in us_gaap_data.items():
                        print("Row: ", key, value)
                        # Here I want to add the logic to retrieve only the value of the row matching the ecl_date

                updated_rows.append(row)
                row_counter += 1
                # Wait before the next request to comply with SEC's rate limiting
                time.sleep(0.5)  # Sleep for half a second
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON on row {row_counter + 1}")
                continue
    # Clear the progress bar line after completion
    print("\rProcessing complete.                      ")
    return updated_rows



def main():
    # Set the path to your ECL dataset file
    file = config
    file_path = './data/ECL_AA_subset.json'

    # Process the ECL file and get updated rows
    updated_rows = process_ecl_file(file_path)

    # Ensure the output directory exists
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ecl_combined.json')

    # Save the updated data to the output file
    with open(output_file, 'w') as f:
        for row in updated_rows:
            json.dump(row, f)
            f.write('\n')

    # Load the data into a pandas DataFrame
    df = pd.DataFrame(updated_rows)

    # Print relevant information about the data
    print("\nData Overview:")
    print(df.info())
    print("\nData Columns:")
    print(df.columns.tolist())

if __name__ == '__main__':
    main()
