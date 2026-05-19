# Dense companyfacts matching process

This note documents how the dense companyfacts dataset matches SEC filing data
to standardized concept columns.

## Input rows

The dense generator reads the metadata JSONL configured by
`config.COMPANYFACTS_METADATA_PATH`. Each row must contain:

- `cik`, passed to the SEC database lookup.
- `accession_number`, passed as the SEC accession / `adsh` for the filing lookup.
- `cik_adsh`, the filing-level identifier written to the output dataset.
- `label` and `year`, which are retained as output columns.
- `filing_role`, used to keep extraction to primary registrant filings.
- `isXBRL`, used to skip rows that do not have XBRL filing data.

The accession number is still needed to retrieve the filing from the SEC data,
but it is no longer the dataset identifier. The output dataset is keyed by
`cik_adsh`.

## SEC database lookup

For each metadata row, the generator attempts SEC numeric extraction only when:

- `isXBRL` is truthy.
- `filing_role` equals `primary_registrant`.

The generator calls `retrieve_statement_taxonomies_by_accession_number3(...)`
with the row's `accession_number`, CIK, filing role, statement code, and quarter
specification.

The statement passes are:

- Income statement: `IS`, quarter spec `[4]`.
- Balance sheet: `BS`, quarter spec `0`.
- Cash flow statement: `CF`, quarter spec `[4]`.
- Statement of stockholders' equity: `EQ`, quarter spec `[0]`.

Each pass returns SEC tags, labels, and values for that statement. Values are
cached by SEC tag for the current filing.

## Standardized concept matching

The dense generator loads standardized concept definitions from
`tools/xbrl_mapping.json`. For each financial statement section, SEC tags and
labels from the filing are matched against the configured concept rules using
`match_concept_in_section(...)`.

When a standardized concept matches a SEC tag, the concept column receives the
cached value for that tag. If no SEC tag matches a standardized concept for the
filing, that concept column remains blank.

## Rationale for the standardized concept set

The standardized columns in the dense dataset are best understood as a
research-oriented canonical schema rather than a complete replacement for the
SEC or FASB taxonomies. The goal is to convert heterogeneous XBRL filings into a
compact set of recurring financial statement concepts that are usable for
cross-filing empirical analysis and modelling.

The selected concepts were guided by common broker and financial-data-provider
templates: recurring line items such as revenue, net income, assets, liabilities,
cash-flow measures, and equity measures are widely used in fundamental analysis
and appear frequently enough to support cross-company comparison. This market
practice was used as a pragmatic starting point because the dataset is intended
for analytical modelling, not for reconstructing every possible disclosure in
the original filing.

The initial concept list was then constrained by matchability. Concepts were
retained when they could be mapped to SEC filing tags with reasonable consistency
using the configured concept aliases, labels, financial statement section, and
matching thresholds. Concepts that were highly industry-specific, depended on
company-specific presentation, or required extensive manual judgment were
excluded. This exclusion is a methodological choice: it prioritizes
reproducibility and lower semantic noise over exhaustive coverage.

This framing is important for interpretation. A populated standardized concept
means that the automated matching process found a plausible SEC tag for that
filing and concept. It should not be read as proof that the dense schema captures
all accounting nuance from the original filing. The dense dataset therefore
trades some filing-specific detail for a smaller, auditable feature space.

## Relation to XBRL standardization literature

The implementation follows a practical version of a broader problem discussed in
XBRL research and regulatory guidance: XBRL is standardized, but it is also
extensible. Filers may use standard taxonomy elements, but they may also create
custom elements when an appropriate standard element does not exist. This makes
machine-readable financial data easier to collect, while still leaving data
users with a mapping and comparability problem.

SEC guidance supports the motivation for a standardization layer. The SEC notes
that custom tags are permitted only in limited circumstances and that unnecessary
customization can reduce comparability across filers. SEC staff observations
also emphasize that element selection is not a simple string-search task:
filers and data users need to consider definitions, attributes, taxonomy
sections, and context rather than relying only on tag names or labels.

Academic work makes the same point from a data-integration perspective. Etudo,
Yoon, and Liu propose Financial Concept Element Mapper (FinCEM) to address
semantic heterogeneity across XBRL filings by mapping heterogeneous elements to
common financial concepts. Wang similarly treats custom-tag standardization as a
practical problem faced by analysts, investors, regulators, and researchers, and
uses NLP methods to map custom XBRL tags to standard taxonomy tags. These works
support the premise that standardizing XBRL concepts is a legitimate empirical
data-engineering step, even when a particular thesis implementation uses a
simpler rule-based matching strategy.

Other literature also suggests why the concept set should not be presented as
definitive. Valentinetti and Rea describe XBRL as a customizable standard: a
strict taxonomy improves comparability but may lose filing-specific information,
while allowing extensions preserves idiosyncratic disclosure at the cost of
comparability. Walton, Yang, and Zhang show that extensions are not always just
noise; in some contexts, extension tags can carry useful information about
complex or specialized reporting. For this dataset, the dense schema is
therefore positioned as a modelling-oriented normalization layer, not as a claim
that excluded or custom concepts are unimportant.

## Output shape

The output CSV starts with:

```text
cik_adsh,label,year
```

All standardized concepts from `tools/xbrl_mapping.json` are appended after
those columns, preserving the mapping order across income statement, balance
sheet, cash flow, and stockholders' equity sections.

The dense generator writes the compact schema directly. There is no separate
post-process step or CLI flag.

## Tag audit output

The generator also writes `outputs/tags.csv`. This file is keyed by `cik_adsh`
and records which SEC tag was selected for each standardized concept when a row
has XBRL data. It is useful for auditing the mapping decisions without changing
the dense dataset schema.

The tag audit is also used as a limitation check. If many distinct SEC tags map
to the same standardized concept, that should be discussed as evidence of XBRL
reporting heterogeneity and matching complexity. It should not be used as proof
that the dense dataset is superior to the sparse representation. Instead, it
shows where the standardization layer is doing the most work and where manual
review would be most valuable.

## Sources

- SEC Division of Economic and Risk Analysis, [IFRS - XBRL Custom Tags Trend for
  2021-2023](https://www.sec.gov/data-research/structured-data/ifrs-xbrl-custom-tags-trend-2021-2023).
  Supports the regulatory background that custom tags are permitted when the
  standard taxonomy lacks an appropriate element, while also noting that custom
  tags can reduce inter-company comparability.
- SEC staff, [Staff Observations From Review of Interactive Data Financial
  Statements](https://www.sec.gov/about/divisions-offices/division-economic-risk-analysis/office-structured-disclosure-staff/osd_staffobs_11-01-10).
  Supports the matching caveat that element selection should consider taxonomy
  definitions, attributes, and context, not only tag names or labels.
- Mark J. Flannery, SEC Chief Economist, [The Commission's Production and Use of
  Structured Data](https://www.sec.gov/newsroom/speeches-statements/2014-spch093014mjf).
  Supports the comparability motivation: broader use of standard tags makes
  inter-company comparison easier for investors, analysts, and researchers.
- Ugochukwu Etudo, Victoria Yoon, and Dapeng Liu, [Financial Concept Element
  Mapper (FinCEM) for XBRL interoperability: Utilizing the M3 Plus
  method](https://www.sciencedirect.com/science/article/abs/pii/S0167923617300659).
  Supports the idea that heterogeneous XBRL elements can be mapped to common
  financial concepts as an interoperability problem.
- Richard Wang, [Standardizing XBRL Financial Reporting Tags with Natural
  Language Processing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4613085).
  Supports the claim that standardizing custom XBRL tags is a practical problem
  for financial analysts, investors, regulators, and researchers using XBRL data
  for financial analysis and modelling.
- Daniela Valentinetti and Michele A. Rea, [Critical reflection on XBRL: A
  "customisable standard" for financial
  reporting?](https://www.macrothink.org/journal/index.php/ijafr/article/view/3870).
  Supports the tradeoff framing: strict standardization improves comparability,
  while extensibility preserves firm-specific disclosure but can reduce
  comparability.
- Stephanie Walton, Liu Yang, and Yiyang Zhang, [XBRL Tag Extensions and Tax
  Accrual Quality](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3700173).
  Supports the limitation that extension tags are not always merely errors or
  noise; they can sometimes reflect specialized reporting information.
