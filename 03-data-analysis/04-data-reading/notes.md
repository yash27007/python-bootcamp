# Reading Data

## Problem

Almost no data starts life as a Python object in memory. It starts as bytes on disk or over a
network — a CSV file, a JSON blob from an API, an Excel workbook, rows in a SQL database. Getting
it into a usable, correctly-typed in-memory `DataFrame` requires answering a series of questions
the raw bytes don't answer for you: what character encoding are these bytes in? what delimiter
separates fields, and what happens when a field's *value* contains that delimiter? what type
should each column be? Reading data is the process of answering all of those questions —
correctly, ideally automatically — before any analysis can begin.

## Intuition

Imagine transcribing a paper form into a spreadsheet by hand: you have to know what language the
handwriting is in (encoding), where one answer ends and the next begins (delimiting — tricky if
someone writes a comma inside their answer), and whether "007" should become the number `7` or
stay the text `"007"` (type inference). A file-reading library like Pandas' `read_csv`/`read_json`
family automates exactly this transcription — with sensible defaults for the common case, and
explicit parameters (`encoding=`, `sep=`, `dtype=`) for the cases where the default guess is
wrong.

## Why simpler approaches fail

The most naive CSV reader — `line.split(',')` for every line — breaks the instant any field's
*value* legitimately contains a comma (e.g. a free-text note field: `"likes coffee, tea"`), because
a plain `split` can't distinguish a delimiter comma from a data comma; it needs to track whether
it is currently "inside" a quoted field. Writing that quote-tracking state machine by hand, for
every file format (CSV, TSV, then separately for JSON's nesting, then separately for Excel's
per-sheet/per-cell-type structure), means re-deriving and re-testing the same edge cases (embedded
delimiters, embedded quotes, mixed encodings, inconsistent types down a column) over and over.
`pd.read_csv`/`read_json`/`read_excel`/`read_sql` exist so each format's edge cases are
implemented once, correctly, and reused — while still exposing the parameters (`encoding`,
`dtype`, `parse_dates`) needed for the genuinely ambiguous cases no default can resolve for you.

## Conceptual foundation

*(Substituting for "Mathematical foundation" per the template's documented substitution
allowance — this topic's foundation is a data-representation model, not a numeric derivation.)*

**Bytes on disk are not self-describing without an encoding.** A file is a sequence of bytes;
turning it back into text requires knowing which byte-to-character mapping (encoding) was used to
write it. UTF-8 and Latin-1 (ISO-8859-1) agree on the ASCII range (bytes 0–127) but diverge above
that — the same byte can be a legitimate character start in one encoding and an invalid/different
character in the other. Reading with the wrong encoding either raises `UnicodeDecodeError` (bytes
that aren't valid under the assumed encoding) or — more dangerously — succeeds but produces the
*wrong characters* (mojibake), because both encodings can independently produce syntactically
valid, but semantically wrong, text for many byte sequences.

**Type inference is a heuristic over a column's values, not a guarantee.** `pd.read_csv` decides
each column's dtype by scanning its values and picking the narrowest type all of them fit — a
column of `"00123", "00456"` "looks like" integers, so it becomes `int64`, silently discarding the
leading zeros a human reading the same text would have kept (because "as an integer, `00123` and
`123` are the same value" — the parser cannot know the leading zero was semantically meaningful,
e.g. a zip code or an ID, rather than incidental).

## Algorithm

Quote-aware CSV field-splitting, generically (what `pd.read_csv` does per line, before type
inference):
1. Scan the line character by character, tracking one boolean: "currently inside a quoted field."
2. A `"` character toggles that boolean (open/close a quoted region) rather than being emitted as
   data.
3. A delimiter (`,`) ends the current field **only** if the boolean is currently false; inside a
   quoted region, a `,` is ordinary data.
4. At end of line, emit the field in progress.

## From-scratch implementation

A minimal quote-aware CSV parser, compared against a naive `split(',')` parser and against
`pd.read_csv`, on the same text (`data_reading.ipynb`, "1b. From-Scratch"):

```python
def naive_split_parse(csv_text):
    return [line.split(',') for line in csv_text.strip().split('\n')]

def quote_aware_parse(csv_text):
    rows = []
    for line in csv_text.strip().split('\n'):
        fields, field, in_quotes = [], [], False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == ',' and not in_quotes:
                fields.append(''.join(field)); field = []
            else:
                field.append(ch)
        fields.append(''.join(field))
        fields = [f[1:-1] if f.startswith('"') and f.endswith('"') else f for f in fields]
        rows.append(fields)
    return rows

sample_csv = 'name,city,note\nAlice,NYC,"likes coffee, tea"\nBob,LA,"no comment"'
```

Actual output:

```
Naive split (breaks on the embedded comma -- 4 fields instead of 3):
['name', 'city', 'note']
['Alice', 'NYC', '"likes coffee', ' tea"']
['Bob', 'LA', '"no comment"']

Quote-aware parse (correct -- 3 fields per row):
['name', 'city', 'note']
['Alice', 'NYC', 'likes coffee, tea']
['Bob', 'LA', 'no comment']

pd.read_csv (handles this automatically):
    name city               note
0  Alice  NYC  likes coffee, tea
1    Bob   LA         no comment
```

The naive parser splits Alice's row into 4 fields instead of 3 — the embedded comma is
mistaken for a delimiter, and the quote characters leak into the field values. The
hand-rolled quote-aware parser and `pd.read_csv` both produce the correct 3-field split,
confirming what quote-awareness — the one piece implemented here — buys over naive splitting.
(`pd.read_csv` additionally infers types, handles multiple quote/escape conventions, and much
more, which this minimal parser does not attempt to replicate.)

## Practical implementation

`data_reading.ipynb` covers the practical surface: CSV (`read_csv`/`to_csv`, `usecols`, `dtype`
override, `parse_dates`, chunked reading, alternate delimiters), JSON (`read_json`/`to_json`,
`pd.json_normalize` for nested API responses), Excel (`read_excel`/`to_excel`, multi-sheet
writing), SQLite (`read_sql`/`to_sql`), Parquet (dtype-preserving columnar format, with a real
measured CSV-vs-Parquet read/write time comparison), and HTML tables (`read_html`) — plus, added
in this pass, the from-scratch quote-aware parser above and the two failure-mode demonstrations
below. (This pass also fixed two pre-existing bugs surfaced by executing every cell: a
`usecols`/`parse_dates` column-name mismatch, and `pd.read_html` needing its input string wrapped
in `io.StringIO` under this environment's pandas 3.0.2 — both fixed in place, with `lxml` and
`html5lib` added as project dependencies since `read_html` requires one of them.)

## Experiment

**Hypothesis:** the only functional difference between the naive and quote-aware parsers, on
text containing one embedded-comma field, is field-count correctness — the quote-aware version
recovers the same field boundaries `pd.read_csv` does.

**Setup:** the 2-row `sample_csv` above, one row with a comma inside a quoted `note` field.

**Actual result:** naive parser: 4 fields for Alice's row (wrong). Quote-aware parser and
`pd.read_csv`: 3 fields for Alice's row, with the `note` value read as the single string
`'likes coffee, tea'` (correct, matching).

**Interpretation:** confirms the conceptual claim that comma-splitting alone is insufficient
without a quote-awareness pass, using both fields' text output as the correctness check.

**Limitations:** the from-scratch parser doesn't handle escaped quotes inside a quoted field
(`""` as a literal `"`), doesn't do type inference, and doesn't handle encoding — it isolates one
piece (quote-aware delimiter splitting) of what `pd.read_csv` does as a whole.

## Failure modes

- **Encoding mismatch (UTF-8 vs Latin-1).** A file written in Latin-1, read assuming UTF-8, can
  either raise or silently mangle text. Measured: writing `'Héllo Wörld - café'` to a file with
  `encoding='latin-1'`, then reading with `encoding='utf-8'` raised
  `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 1: invalid continuation
  byte` — the `é` byte under Latin-1 is not a valid UTF-8 continuation byte. Reading the same
  file with `encoding='latin-1'` (matching the write) correctly recovered
  `'Héllo Wörld - café'`. (This particular pair of encodings happened to *raise* rather than
  silently mangle for this text — other byte sequences produce syntactically valid but wrong
  text under the mismatched encoding instead, which is the more dangerous case because nothing
  signals the error.)
- **Silent type misinference on a numeric-looking ID column.** Measured:
  `employee_id` values `'00123', '00456', '00789'` read with default `pd.read_csv` become
  `int64` `[123, 456, 789]` — the leading zeros are gone, silently, with no error or warning.
  Forcing `dtype={'employee_id': str}` correctly preserves them as `['00123', '00456', '00789']`.
  This matters wherever leading zeros are semantically meaningful (zip codes, employee/product
  IDs, phone extensions) — the integer `123` and the string `'00123'` are *not* the same key for
  a downstream join or lookup, and the corruption happens with no exception to catch.

## Real-world usage

Every ML/DS pipeline begins with a read step: training data usually starts as CSV/Parquet on
disk or query results from a warehouse; production features are frequently read from
database/API sources with unpredictable encodings; ID columns (customer IDs, SKUs, zip codes)
are one of the most common sources of the silent-int-coercion bug in production pipelines,
because they look numeric but are not meant to be treated as numbers. Parquet's popularity in
big-data pipelines over CSV is a direct consequence of the dtype-preservation issue above —
Parquet stores the dtype alongside the data, so it can't silently reinterpret a string ID column
as an integer the way a CSV round-trip can.

## Mental model

Reading data correctly means answering three questions the raw bytes never answer for you:
*what encoding are these bytes in, where does one field end and the next begin, and what type
should this column really be* — and the two failure modes in this topic are exactly the second
and third of those questions answered wrong, silently, by a reasonable-looking default.

## Questions to think about

1. The Latin-1/UTF-8 mismatch above happened to raise `UnicodeDecodeError` rather than silently
   produce wrong characters. Under what condition (in terms of which specific byte values appear
   in the file) would a UTF-8-vs-Latin-1 mismatch instead succeed silently with mojibake?
2. `pd.read_csv(..., dtype={'employee_id': str})` fixes the leading-zero problem for a fresh
   read. If the ID column had *already* been read once as `int64` and saved back out to CSV,
   could `dtype=str` on the next read still recover the original `'00123'`? Why or why not?
3. The from-scratch quote-aware parser toggles `in_quotes` on every `"` character with no
   distinction between "open quote" and "close quote." What real CSV content would break this
   simplification that a full CSV grammar (like the one `pd.read_csv` implements) handles
   correctly?
4. Why does Parquet avoid the leading-zero-loss failure mode that CSV has, structurally — what
   does Parquet store that a CSV file does not?
5. If you had to choose a `chunksize` for reading a 50 GB CSV file on a machine with 8 GB of RAM,
   what would decide the value — and what does the from-scratch parser's line-by-line loop
   structure suggest about why chunked reading is even possible for CSV specifically?
