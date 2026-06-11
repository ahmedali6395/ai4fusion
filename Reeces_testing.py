# Cleaned and simplified data inspection / cleaning script
# Reads the CSV, reports NaN counts, drops specified columns and
# columns with too many NaNs, then drops any rows containing NaNs.

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

# Load data
df = pd.read_csv("data/raw_data/cmod.csv", sep=',', index_col=None)

# Report NaN counts per column
nan_counts = df.isna().sum()
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print("NaN count per column:")
    print(nan_counts)

# Drop a few known non-data columns if present
cols_to_drop_first = ["Unnamed: 0", "shot", "time_until_disrupt"]
df_reduced = df.drop(columns=cols_to_drop_first, errors="ignore")

# Drop any columns with >100000 NaNs
cols_with_many_nans = df_reduced.isna().sum()[lambda x: x > 100000].index
df_clean = df_reduced.drop(columns=cols_with_many_nans)
if len(cols_with_many_nans) > 0:
    print("Columns dropped for too many NaNs:", list(cols_with_many_nans))

# Drop rows with any NaN
df_no_nans = df_clean.dropna()

# Summary relative to original
orig_rows, orig_cols = df.shape
final_rows, final_cols = df_no_nans.shape
orig_entries = df.size
final_entries = df_no_nans.size

rows_pct = 100 * final_rows / orig_rows
cols_pct = 100 * final_cols / orig_cols
entries_pct = 100 * final_entries / orig_entries

print(f"Rows remaining vs original:    {final_rows}/{orig_rows}  ({rows_pct:.2f}%)")
print(f"Columns remaining vs original: {final_cols}/{orig_cols}  ({cols_pct:.2f}%)")
print(f"Entries remaining vs original: {final_entries}/{orig_entries}  ({entries_pct:.2f}%)")