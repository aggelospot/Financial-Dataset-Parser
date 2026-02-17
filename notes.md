==================================================================

EXAMPLES 

SELECT tag.tlabel, pre.tag, tag.version, num.qtrs, num.segments
FROM public.num AS num
LEFT JOIN public.pre AS pre
  ON num.adsh = pre.adsh AND num.tag = pre.tag
JOIN public.tag AS tag
  ON pre.tag = tag.tag AND pre.version = tag.version
WHERE num.adsh = '0001047469-15-006136'
  AND (pre.stmt = 'IS')
  AND (num.qtrs = 4)
  AND num.segments IS NULL
  AND num.ddate = (
    SELECT MAX(ddate)
    FROM public.num AS n2
    WHERE n2.adsh = num.adsh AND n2.tag = num.tag
  )
ORDER BY num.tag;


SELECT DISTINCT ON (tag) tag, tlabel,custom, version, doc
FROM public.tag
WHERE tag.tlabel LIKE 'Net Income'

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4613085


Example for tag Revenues on company AAR CORP (2nd row of dataset)
https://www.sec.gov/Archives/edgar/data/1750/000104746911006302/index.json

calc linkbase: run extract_parent_concepts.py with params: C:\\Users\Angelo\PycharmProjects\financial_dataset\downloads\air-20110531_cal.xml



Scanning tags: 100%|██████████| 36788/36788 [13:56<00:00, 43.99it/s] 
Filings scanned   : 36,788
Unique tags found : 93,381
Top 10 most common tags:
  CashAndCashEquivalentsAtCarryingValue    73,467
  NetIncomeLoss                            63,014
  StockholdersEquity                       62,312
  StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest 40,096
  ProfitLoss                               37,562
  Assets                                   31,577
  LiabilitiesAndStockholdersEquity         31,348
  CommonStockSharesOutstanding             31,022
  RetainedEarningsAccumulatedDeficit       29,649
  IncomeTaxExpenseBenefit                  29,506
Saved to: C:\\Users\\Angelo\\PycharmProjects\\financial_dataset\\outputs\\ecl_all_statements.jsonl


SELECT DISTINCT(pre.tag)
FROM public.pre AS pre



TODO: Exclude custom variables, see if that improves the data

SELECT DISTINCT version
FROM num
WHERE version != adsh


TODOS: 

- 
- Kmeans with way more clusters
- 




- TODOS this week 

standardization / imputation, multimodal models with concatenated mda/auditor or split. 

fbeta optimize (during validation)
# Plot PR curves
    p, r, thresholds = precision_recall_curve(y_test, pred_proba)
    # Find the threshold that gives the best F1-score
    numerator = 2 * r * p
    denom = r + p
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    best_f1 = np.max(f1_scores)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1-score: {best_f1:.2f}")

























