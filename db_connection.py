import psycopg2
from tools import config, db_schema
import pandas as pd
import os
from tools.utils import match_concept_in_section
from dotenv import load_dotenv
from decimal import Decimal
import string

load_dotenv()

def create_connection():
    """
    Creates and returns a psycopg2 connection to the PostgreSQL database.
    Replace the placeholders with your own connection parameters.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("HOST"),
            database=os.getenv("DATABASE_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("PASSWORD")
        )
        return conn
    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
        raise

def close_connection(conn):
    if conn:
        conn.close()

def import_csv_to_sec_num(conn, file_path):
    """
    Imports CSV data into the 'sec_num' table using the COPY command.
    Assumes the CSV file has a header row matching the columns of sec_num.
    """
    try:
        with conn.cursor() as cur:
            copy_sql = """
                COPY sub
                FROM STDIN
                WITH (
                    FORMAT CSV,
                    DELIMITER \'\t\',
                    HEADER TRUE
                )
            """
            with open(file_path, 'r') as f:
                cur.copy_expert(copy_sql, f)
        # commit so changes are persisted
        conn.commit()
    except psycopg2.Error as e:
        print("Error importing CSV to sec_num:", e)
        raise


def import_all_num_csv(conn, db_imports_dir):
    """
    Recursively traverses the db_imports_dir and its subdirectories,
    looking for 'num.csv' files and importing them to the sec_num table.

    :param conn: psycopg2 connection object
    :param db_imports_dir: The path to the directory containing subdirectories (e.g., db_imports/2020q1, db_imports/2020q2, etc.)
    """
    count = 0
    for root, dirs, files in os.walk(db_imports_dir):
        print(f"root={root}, dir={dirs}, files={files}")
        for file_name in files:
            if file_name.lower() == "tag.txt":
                file_path = os.path.join(root, file_name)
                print(f"Found sub.csv: {file_path}. Importing to database...")

                # uncomment to import sub, then pre. Add more import_sub_related_table calls for other files
                # import_sub_table(conn,file_path)
                # import_sub_related_table(conn,file_path,table_name="tag")
                import_tag_table(conn,file_path,table_name="tag")
                print("CSV successfully imported!")
                # if count > 3: return


def import_sub_table(conn, file_path):
    """
    Imports rows from a CSV file into the 'sub' table, filtering only
    those rows whose 'form' contains '10-K'.
    """
    staging_table_ddl = db_schema.STAGING_TABLE_SUB_SCHEMA

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS staging_sub;")

        cur.execute(staging_table_ddl)

        copy_sql = """
            COPY staging_sub
            FROM STDIN
            WITH (
                FORMAT CSV,
                DELIMITER '\t',
                HEADER TRUE
            );
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            cur.copy_expert(copy_sql, f)

        # Insert only the rows that have '10-K' in the 'form' column into the final 'sub' table.
        insert_sql = """
            INSERT INTO sub 
            SELECT *
            FROM staging_sub
            WHERE form LIKE '%10-K%';
        """
        cur.execute(insert_sql)

        cur.execute("DROP TABLE staging_sub;")

    conn.commit()

def get_table_schema(table_name):
    if table_name == 'num':
        return db_schema.STAGING_TABLE_NUM_SCHEMA
    if table_name == 'pre':
        return db_schema.STAGING_TABLE_PRE_SCHEMA
    if table_name == 'tag':
        return db_schema.STAGING_TABLE_TAG_SCHEMA

def import_sub_related_table(conn, file_path, table_name):
    """
    Imports rows from a CSV file into a target table,
    but only for those rows where 'adsh' is already present in the 'sub' table.
    """

    drop_staging_sql = f"DROP TABLE IF EXISTS staging_{table_name};"
    create_staging_sql = get_table_schema(table_name)

    copy_sql = f"""
        COPY staging_{table_name}
        FROM STDIN
        WITH (
            FORMAT CSV,
            DELIMITER '\t',
            HEADER TRUE
        );
    """

    insert_sql = f"""
            WITH filtered_sub AS (
            SELECT s.*
            FROM staging_{table_name} AS s
            JOIN sub ON s.adsh = sub.adsh
        )
        INSERT INTO {table_name}
        SELECT * FROM filtered_sub;
        """

    drop_staging_after_sql = f"DROP TABLE staging_{table_name};"

    with conn.cursor() as cur:
        cur.execute(drop_staging_sql)

        cur.execute(create_staging_sql)

        # Bulk load CSV into staging table
        with open(file_path, 'r', encoding='utf-8') as f:
            cur.copy_expert(copy_sql, f)

        # Insert only matching records into the real table
        cur.execute(insert_sql)

        # Drop the staging table
        cur.execute(drop_staging_after_sql)

    conn.commit()


def import_tag_table(conn, file_path, table_name):
    """
    Imports rows from a TSV file into the 'tag' table using a staging table
    to avoid inserting duplicates based on (tag, version).
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS staging_tag;")
        cur.execute(f"CREATE TEMP TABLE staging_tag (LIKE {table_name} INCLUDING ALL);")

        # Load into staging
        with open(file_path, 'r', encoding='utf-8') as f:
            print("Copying into staging table...")
            cur.copy_expert(f"""
                COPY staging_tag FROM STDIN WITH (
                    FORMAT CSV,
                    DELIMITER '\t',
                    HEADER TRUE
                );
            """, f)

        # Insert only new rows
        cur.execute(f"""
            INSERT INTO {table_name}
            SELECT DISTINCT *
            FROM staging_tag
            ON CONFLICT (tag, version) DO NOTHING;
        """)

    conn.commit()



def get_tag_values_by_accession_number(connection, accession_number, tags, num_quarters=4):
    """
    Fetches the values from a list of tags. example: tags=['NetIncomeLoss','EarningsPerShareBasic']
    """

    select_sql = f"""
    SELECT DISTINCT ON (tag) tag,value
    FROM public.num
    WHERE adsh = %s
    AND qtrs = %s
    AND tag = ANY(%s)
    ORDER BY tag, ddate DESC;
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(select_sql, (accession_number, num_quarters, tags))
            results = cursor.fetchall()
            return results
    except Exception as e:
        print("Error executing query:", e)
        return None

def get_tags_value_by_accession_number(connection, accession_number, tag, num_quarters=[0,4]):
    # Ensure tags are unique and properly formatted for SQL
    select_sql = f"""
    SELECT DISTINCT ON (tag) value
    FROM public.num
    WHERE adsh = %s
    AND qtrs = ANY(%s)
    AND tag = %s
    ORDER BY tag, ddate DESC;
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(select_sql, (accession_number, num_quarters, tag))
            result = cursor.fetchone()  # Fetch a single row instead of fetchall()

            if result:  # Check if there's a result
                return float(result[0]) if isinstance(result[0], Decimal) else result[0]
            else:
                return None  # If no result found, return None
    except Exception as e:
        print("Error executing query:", e)
        return None


_TRANSLATOR = str.maketrans('', '', string.punctuation)
def retrieve_statement_taxonomies_by_accession_number(accession_number, financial_statement, num_quarters):
    """
    Retrieves a financial statement's unique taxonomies found in the report using the accession_number.

    Options for financial statements:
        BS = BalanceSheet, IS = IncomeStatement, CF = CashFlow,
        EQ = Equity, CI = ComprehensiveIncome, SI = Schedule of Investments,
        UN = Unclassifiable Statement

    Use num_quarters = 0 for the balance sheet, and num_quarters = [0,4] for IS, CF, and EQ.
    """

    # Ensure num_quarters is iterable
    if isinstance(num_quarters, list):
        num_quarters_clause = " OR ".join(f"num.qtrs = {q}" for q in num_quarters)
        where_qtrs_clause = f"({num_quarters_clause})"
    else:
        where_qtrs_clause = f"num.qtrs = {num_quarters}"

    # Ensure num_quarters is iterable
    if isinstance(financial_statement, list):
        financial_statement_clause = " OR ".join(f"pre.stmt = {s}" for s in financial_statement)
        where_fin_statement_clause = f"({financial_statement_clause})"
    else:
        where_fin_statement_clause = f"pre.stmt = {financial_statement}"

    query_sql = f"""
        SELECT pre.tag, tag.tlabel, num.value tag.version, num.qtrs, num.segments
        FROM public.num AS num
        LEFT JOIN public.pre AS pre
          ON num.adsh = pre.adsh AND num.tag = pre.tag
        JOIN public.tag AS tag
          ON pre.tag = tag.tag AND pre.version = tag.version
        WHERE num.adsh = '{accession_number}'
          AND {where_fin_statement_clause}
          AND {where_qtrs_clause}
          AND num.segments IS NULL
          AND num.ddate = (
            SELECT MAX(ddate)
            FROM public.num AS n2
            WHERE n2.adsh = num.adsh AND n2.tag = num.tag
          )
        ORDER BY num.tag;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query_sql)
            result = cur.fetchall()

            return (
                [row[0] for row in result],
                [row[1].translate(_TRANSLATOR) for row in result]  # stripped punctuation
            )

    except psycopg2.Error as e:
        print("Error executing query on sec_num:", e)
        raise


def retrieve_statement_taxonomies_by_accession_number2(
        accession_number: str,
        financial_statement,
        num_quarters
    ):
    """
    Returns three parallel lists:
        tags, labels, values (millions – only when uom == 'USD')
    """

    # build the quarter filter
    if isinstance(num_quarters, list):
        num_quarters_clause = " OR ".join(f"num.qtrs = {q}" for q in num_quarters)
        where_qtrs_clause = f"({num_quarters_clause})"
    else:
        where_qtrs_clause = f"num.qtrs = {num_quarters}"

    # Ensure num_quarters is iterable
    # if isinstance(financial_statement, list):
    #     financial_statement_clause = " OR ".join(f"pre.stmt = {s}" for s in financial_statement)
    #     where_fin_statement_clause = f"({financial_statement_clause})"
    # else:
    #     where_fin_statement_clause = f"pre.stmt = '{financial_statement}'"

    if isinstance(financial_statement, list):
        stmt_list = ", ".join("%s" for _ in financial_statement)
        where_fin_statement_clause = f"pre.stmt IN ({stmt_list})"
        stmt_params = tuple(financial_statement)
    else:
        where_fin_statement_clause = "pre.stmt = %s"
        stmt_params = (financial_statement,)

    # select num.value and num.uom so we can normalise
    query_sql = f"""
        SELECT pre.tag,
               tag.tlabel,
               num.value,
               tag.datatype,                 
               num.qtrs,
               num.segments
        FROM   public.num AS num
               LEFT JOIN public.pre  AS pre  ON num.adsh = pre.adsh
                                            AND num.tag  = pre.tag
               JOIN public.tag  AS tag  ON pre.tag     = tag.tag
                                         AND pre.version = tag.version
        WHERE  num.adsh = %s
          AND  {where_fin_statement_clause}
          AND  {where_qtrs_clause}
          AND  num.segments IS NULL
          AND  num.ddate = (SELECT MAX(ddate)
                            FROM   public.num n2
                            WHERE  n2.adsh = num.adsh
                              AND  n2.tag  = num.tag)
        ORDER BY pre.tag;
    """

    # to remove custom tags: AND tag.version != pre.adsh
    try:
        with conn.cursor() as cur:
            cur.execute(query_sql, (accession_number, *stmt_params))
            rows = cur.fetchall()

        tags, labels, values = [], [], []
        for tag, label, raw_val, datatype, *_ in rows:
            # ➋  normalise only when the facts are in USD
            val = raw_val # / 1_000_000 if datatype == 'monetary' and raw_val is not None else raw_val
            val = float(f"{val:.6f}".rstrip('0').rstrip('.')) if isinstance(val, float) else val
            tags.append(tag)
            labels.append(label)
            values.append(val)

        return tags, labels, values

    except psycopg2.Error as e:
        print("Error executing query on sec_num:", e)
        raise


from decimal import Decimal
import json
from pathlib import Path
from tqdm import tqdm

def append_raw_statement_data_stream(
    ecl_df,
    *,
    statements=['BS','CF','IS','EQ'],
    quarter_spec=[0,4],
    output_path='ecl_all_statements.jsonl',
    batch=1_000
):
    out = Path(output_path).open('w', encoding='utf-8')
    buffer = []

    # helper that turns Decimal → float
    def to_jsonable(x):
        if isinstance(x, Decimal):
            return float(x)
        return x

    for _, row in tqdm(ecl_df.iterrows(), total=len(ecl_df), desc="Fetching facts"):
        if not row.get('isXBRL', 1):
            continue

        tags, _, values = retrieve_statement_taxonomies_by_accession_number2(
            row['accessionNumber'], statements, quarter_spec)

        # ✨ make sure every value is JSON-safe
        rec = {"accession_number": row['accessionNumber'],
               **{k: to_jsonable(v) for k, v in zip(tags, values)}}

        buffer.append(rec)

        if len(buffer) >= batch:
            for rec in buffer:
                out.write(json.dumps(rec) + '\n')
            buffer.clear()

    # flush remainder
    for rec in buffer:
        out.write(json.dumps(rec) + '\n')
    out.close()


from collections import Counter
from tqdm import tqdm

def collect_unique_tags(
    ecl_df,
    *,
    statements: list[str] = ['BS', 'CF', 'IS', 'EQ'],
    quarter_spec = [0, 4],
    show_progress: bool = True,
):
    """
    Returns
    -------
    tag_set           : set[str]
        All distinct XBRL tags encountered.
    tag_frequencies   : Counter
        How many times each tag appeared across all filings.
    """

    unique_tags      : set[str] = set()
    tag_frequencies  : Counter  = Counter()

    iterator = tqdm(ecl_df.itertuples(index=False),
                    total=len(ecl_df),
                    disable=not show_progress,
                    desc="Scanning tags")

    for row in iterator:
        if getattr(row, 'isXBRL', 1) == 0:
            continue

        adsh = row.accessionNumber

        tags, _, _ = retrieve_statement_taxonomies_by_accession_number2(
            adsh,
            financial_statement=statements,
            num_quarters=quarter_spec
        )

        unique_tags.update(tags)      # set keeps only new names
        tag_frequencies.update(tags)  # track how often each shows up

    # simple, informative summary
    print(f"Filings scanned   : {len(ecl_df):,}")
    print(f"Unique tags found : {len(unique_tags):,}")
    print(f"Top 10 most common tags:")
    for tag, cnt in tag_frequencies.most_common(10):
        print(f"  {tag:<40} {cnt:,}")

    return unique_tags, tag_frequencies

conn = create_connection()
pd.options.display.max_colwidth = 1500
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

try:

    # file_path = os.path.join(os.path.join(config.IMPORTS_DIR, "2020q4(1)"), "num.csv")
    # import_all_num_csv(conn, config.IMPORTS_DIR)

    from tools.data_loader import DataLoader
    import json
    data_loader = DataLoader()

    ecl = data_loader.load_dataset(config.ECL_METADATA_NOTEXT_PATH, alias='ecl', lines=True)
    with open(config.XBRL_MAPPING_PATH, 'r') as file:
        xbrl_mapping = json.load(file)

    # collect_unique_tags(ecl_df=ecl)
    # METHOD 1: CREATE A SPARSE DATAFRAME OF THE RAW DATA
    # output_file = os.path.join(config.OUTPUT_DIR, "ecl_all_statements.jsonl")
    # append_raw_statement_data_stream(ecl, output_path=output_file)

    # ecl_all = pd.read_json(output_file, lines=True)
    # print(f"Sparse dataframe shape: {ecl_all.shape}")
    # print(f"Saved to: {output_file}")

    # METHOD 2: ATTEMPT TO STANDARDIZE THE TAGS
    tag_list = []
    for section in ["IncomeStatement", "BalanceSheet", "CashFlow", "StatementOfStockholdersEquity"]:
        for key, value in xbrl_mapping[section].items():
            if key not in tag_list:  # Avoid duplicates
                tag_list.append(key)
    for tag in tag_list:
        ecl[tag] = None

    results = []
    filing_cache = {}  # {accession_number: {tag: value}}
    for idx, row in ecl.iterrows():
        print(f"\rCurrent row: {idx}/{len(ecl)}", end='')
        # if idx > 300: break
        if row['isXBRL'] == 0:
            continue  # skip non-XBRL filings

        adsh = row['accessionNumber']
        filing_cache[adsh] = {}  # start fresh for this filing


        # ------------------------------------------------
        # helper: store (tag → value) for later reuse
        # ------------------------------------------------
        def cache_statement(stmt_code, quarter_spec):
            tags, labels, vals = retrieve_statement_taxonomies_by_accession_number2(
                adsh, stmt_code, quarter_spec)
            filing_cache[adsh].update(dict(zip(tags, vals)))
            return tags, labels  # keep old behaviour where needed


        # ---------------- Income Statement ----------------------
        is_tags, is_labels = cache_statement('IS', [4])
        matched_is_items = match_concept_in_section(
            xbrl_mapping['IncomeStatement'],
            is_tags, is_labels)

        # print(matched_is_items)
        # print("current cache", filing_cache[adsh])

        #  Balance Sheet
        bs_tags, bs_labels = cache_statement('BS', 0)
        matched_bs_items = match_concept_in_section(
            xbrl_mapping['BalanceSheet'],
            bs_tags, bs_labels)

        #  Cash-flow
        cf_tags, cf_labels = cache_statement('CF', [4])
        matched_cf_items = match_concept_in_section(
            xbrl_mapping['CashFlow'],
            cf_tags, cf_labels)

        #  Stockholders' Equity
        eq_tags, eq_labels = cache_statement('EQ', [0])
        matched_eq_items = match_concept_in_section(
            xbrl_mapping['StatementOfStockholdersEquity'],
            eq_tags, eq_labels)

        # assemble row using cached values
        metadata_row = {
            "accession_number": adsh,
            "isXBRL": 1,
            **matched_is_items,
            **matched_bs_items,
            **matched_cf_items,
            **matched_eq_items,
        }

        for concept_name, sec_tag in metadata_row.items():
            if concept_name in tag_list:  # skip meta fields
                val = filing_cache[adsh].get(sec_tag)
                ecl.at[idx, concept_name] = val

        results.append(metadata_row)

    result_df = pd.DataFrame(results)
    print("Resulting df: \n", result_df.head(1000))
    print("Resulting ecl df \n", ecl.head(100))
    data_loader.save_dataset(result_df, os.path.join(config.OUTPUT_DIR, "tags.csv"))
    data_loader.save_dataset(ecl, os.path.join(config.OUTPUT_DIR, "ecl_with_financial_tags.csv"))








    # print("Query results : ", result)
    # print("Target concepts: \n", target_concepts)
    # print("End!\n", matched_ic_items)



finally:
    # 4. Close the connection regardless of success/failure
    close_connection(conn)