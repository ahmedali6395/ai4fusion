import pandas as pd
import os

# Candidate paths (workspace-relative)
CANDIDATES = [
    os.path.join("data", "raw_data", "cmod.csv"),
    os.path.join("data", "raw_data", "cmod_clean_200ms.csv"),
    os.path.join("data", "raw_data", "cmod_clean.csv"),
    "cmod_clean_200ms.csv",
]

for path in CANDIDATES:
    if os.path.exists(path):
        DATA_PATH = path
        break
else:
    raise FileNotFoundError(f"CSV not found; tried: {CANDIDATES}")

print("Loading:", DATA_PATH)

df = pd.read_csv(DATA_PATH, sep=',', index_col=None)

if df.shape[1] < 2:
    print("CSV has fewer than 2 columns; nothing to drop")
else:
    df_cut = df.iloc[:, 2:].copy()  # drop first two columns by position
    print("Dropped first two columns. New shape:", df_cut.shape)
    print("\nSnippet (first 10 rows):")
    print(df_cut.head(10).to_string(index=False))
