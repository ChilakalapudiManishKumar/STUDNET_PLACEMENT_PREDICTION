"""
Student Placement Prediction - Data Cleaning Pipeline

FOLDER SETUP (all files in the SAME folder as this script):
    your_project_folder/
        pipeline.py                          <- this file
        student_placement_dataset.csv        <- your raw dataset

Running this script creates:
        student_placement_cleaned.csv        <- cleaned output, used by later scripts
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Portable paths: everything is relative to THIS script's location,
# not to whatever folder you happen to launch python from.
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "student_placement_dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "student_placement_cleaned.csv")

# ============================================================
# STEP 1: LOAD
# ============================================================
df = pd.read_csv(INPUT_CSV)
print("Raw shape:", df.shape)

# ============================================================
# STEP 3: DATA CLEANING
# ============================================================

# --- 3a. Remove exact duplicate rows ---
n_dupe_rows = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n_dupe_rows} fully duplicate rows -> {len(df)} rows remain")

# --- 3b. Handle duplicate Student_ID (keep first occurrence, drop the rest) ---
id_dupe_count = df['Student_ID'].duplicated().sum()
df = df.drop_duplicates(subset='Student_ID', keep='first').reset_index(drop=True)
print(f"Removed {id_dupe_count} rows with repeated Student_ID -> {len(df)} rows remain")

# --- 3c. Clean numeric columns: strip %, spaces, commas-as-decimals, word numbers ---
word_to_num = {
    "nine": 9, "ninety": 90, "eighty": 80, "eighty five": 85,
    "two": 2, "five": 5, "none": 0
}

def clean_numeric(series):
    def _clean(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s == "":
            return np.nan
        s_lower = s.lower()
        if s_lower in word_to_num:
            return word_to_num[s_lower]
        s = s.replace("%", "")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(_clean)

numeric_cols = ["CGPA", "10th_Percentage", "12th_Percentage", "Internships",
                "Coding_Score", "Aptitude_Score", "Projects", "Backlogs"]

for col in numeric_cols:
    df[col] = clean_numeric(df[col])

# --- 3d. Enforce valid ranges (invalid values -> NaN, to be imputed) ---
valid_ranges = {
    "CGPA": (5.0, 10.0),
    "10th_Percentage": (45, 100),
    "12th_Percentage": (45, 100),
    "Internships": (0, 4),
    "Coding_Score": (30, 100),
    "Aptitude_Score": (30, 100),
    "Projects": (0, 8),
    "Backlogs": (0, 8),
}

invalid_counts = {}
for col, (lo, hi) in valid_ranges.items():
    invalid_mask = ~df[col].between(lo, hi) & df[col].notna()
    invalid_counts[col] = int(invalid_mask.sum())
    df.loc[invalid_mask, col] = np.nan

print("\nInvalid (out-of-range) values found & nulled per column:")
for c, n in invalid_counts.items():
    print(f"  {c}: {n}")

# --- 3e. Standardize Communication ---
def standardize_communication(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    mapping = {
        "poor": "Poor", "poorr": "Poor",
        "average": "Average", "avarage": "Average", "avg": "Average",
        "good": "Good", "goood": "Good", "gd": "Good",
        "excellent": "Excellent", "exellent": "Excellent", "excelent": "Excellent",
    }
    return mapping.get(s, np.nan)

df["Communication"] = df["Communication"].apply(standardize_communication)

# --- 3f. Standardize Placement ---
def standardize_placement(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    yes_set = {"yes", "y", "placed", "selected"}
    no_set = {"no", "n", "rejected", "not placed"}
    if s in yes_set:
        return "Yes"
    if s in no_set:
        return "No"
    return np.nan

df["Placement"] = df["Placement"].apply(standardize_placement)

print("\nAfter standardizing text columns:")
print(df["Communication"].value_counts(dropna=False))
print(df["Placement"].value_counts(dropna=False))

# --- 3g. Drop rows with missing target (can't train on unknown label) ---
before = len(df)
df = df[df["Placement"].notna()].reset_index(drop=True)
print(f"\nDropped {before - len(df)} rows with missing/unrecoverable Placement label")

# --- 3h. Impute missing values ---
for col in numeric_cols:
    median_val = df[col].median()
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    print(f"Imputed {n_missing} missing values in {col} with median={median_val:.2f}")

mode_val = df["Communication"].mode()[0]
n_missing_comm = df["Communication"].isna().sum()
df["Communication"] = df["Communication"].fillna(mode_val)
print(f"Imputed {n_missing_comm} missing Communication values with mode='{mode_val}'")

# --- 3i. Fix dtypes ---
df["Internships"] = df["Internships"].round().astype(int)
df["Projects"] = df["Projects"].round().astype(int)
df["Backlogs"] = df["Backlogs"].round().astype(int)
df["Student_ID"] = df["Student_ID"].astype(int)

print("\nFinal cleaned shape:", df.shape)
print(df.dtypes)

# --- 3j. Outlier detection using IQR (cap rather than delete) ---
def iqr_cap(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    capped = series.clip(lower, upper)
    n_outliers = ((series < lower) | (series > upper)).sum()
    return capped, n_outliers, lower, upper

outlier_report = {}
for col in ["CGPA", "10th_Percentage", "12th_Percentage", "Coding_Score", "Aptitude_Score"]:
    capped, n_out, lo, hi = iqr_cap(df[col])
    outlier_report[col] = (n_out, lo, hi)
    df[col] = capped

print("\nIQR outlier capping report (col: count, lower, upper):")
for c, (n, lo, hi) in outlier_report.items():
    print(f"  {c}: {n} outliers capped to [{lo:.2f}, {hi:.2f}]")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved cleaned dataset to: {OUTPUT_CSV}")
