import os

# Get the root directory of the project by navigating up from the current file's directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Navigate up one level

# Define other directories based on the root directory
DATA_DIR = os.path.join(BASE_DIR, 'data')              # Data directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')         # Output directory
SEC_DATA_DIR = os.path.join(DATA_DIR, 'companyfacts')  # SEC data directory
SEC_SUBMISSIONS_DIR = os.path.join(DATA_DIR, 'submissions')  # SEC data directory
CALC_LINKBASES_DIR = os.path.join(DATA_DIR, 'calc_linkbases')  # Calculation linkbases directory
FINANCIALS_DIR = os.path.join(DATA_DIR, 'financials')  #  'financials' directory
IMPORTS_DIR = os.path.join(DATA_DIR, 'db_imports')
# IMPORTS_DIR = os.path.join(DATA_DIR, 'test')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
ECL_FILE_PATH = os.path.join(DATA_DIR, 'ECL_AA_subset.json')                          # ECL dataset
ECL_FILTERED_PATH = os.path.join(DATA_DIR, 'ECL_filtered_subset.json')                # ECL filtered for year > 2008
ECL_SUBMISSIONS_METADATA_PATH = os.path.join(OUTPUT_DIR, 'ECL_submissions_metadata.json') # ECL_w_metadata
ECL_METADATA_PATH = os.path.join(OUTPUT_DIR, 'ECL_metadata.json')
ECL_METADATA_NOTEXT_PATH = os.path.join(OUTPUT_DIR, 'ECL_metadata_no_text.json')
ECL_COMPANYFACTS_NOTEXT_PATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts_no_text.json')
ECL_COMPANYFACTS_SPARSE_PATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts_sparse.json')
ECL_COMPANYFACTS_DENSE_PATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts_dense.json')
RAW_DATASET_FILEPATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts_raw.json')          # Combined ECL + companyfacts dataset
ALL_VARS_DATASET_FILEPATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts.json')                  # ECL + companyfacts (no nulls)
POST_PROCESSED_DATASET_FILEPATH = os.path.join(OUTPUT_DIR, 'ecl_companyfacts_processed.csv')  # Final processed dataset
COLUMN_STATS_FILEPATH = os.path.join(OUTPUT_DIR, 'column_statistics.csv')             # Column statistics file



XBRL_MAPPING_PATH = os.path.join('tools', 'xbrl_mapping.json')
