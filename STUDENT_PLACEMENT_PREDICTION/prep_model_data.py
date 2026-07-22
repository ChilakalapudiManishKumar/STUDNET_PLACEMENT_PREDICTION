"""
Student Placement Prediction - Feature Engineering, Encoding, Scaling, Split

FOLDER SETUP:
    your_project_folder/
        prep_model_data.py                 <- this file
        student_placement_cleaned.csv      <- created by pipeline.py (run that first!)

Creates in the same folder:
        student_placement_engineered.csv
        X_train.csv, X_test.csv, y_train.csv, y_test.csv
        scaler.joblib
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "student_placement_cleaned.csv")

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Could not find {INPUT_CSV}\n"
        "Run pipeline.py first — it creates student_placement_cleaned.csv from the raw dataset."
    )

df = pd.read_csv(INPUT_CSV)

# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================
df["Academic_Score"] = (df["CGPA"] * 10 + df["10th_Percentage"] + df["12th_Percentage"]) / 3
df["Experience_Score"] = df["Internships"] * 2 + df["Projects"]
df["Skill_Index"] = (df["Coding_Score"] + df["Aptitude_Score"]) / 2 - df["Backlogs"] * 3

print("New engineered features preview:")
print(df[["Academic_Score", "Experience_Score", "Skill_Index"]].describe())

# ============================================================
# STEP 6: ENCODING
# ============================================================
comm_map = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
df["Communication_Encoded"] = df["Communication"].map(comm_map)
df["Placement_Encoded"] = df["Placement"].map({"No": 0, "Yes": 1})

feature_cols = [
    "CGPA", "10th_Percentage", "12th_Percentage", "Internships",
    "Coding_Score", "Aptitude_Score", "Communication_Encoded",
    "Projects", "Backlogs", "Academic_Score", "Experience_Score", "Skill_Index"
]

X = df[feature_cols].copy()
y = df["Placement_Encoded"].copy()

# ============================================================
# STEP 7: FEATURE SCALING
# ============================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# ============================================================
# STEP 8: TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train placement rate:", y_train.mean().round(3), " Test placement rate:", y_test.mean().round(3))

joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.joblib"))
X_train.to_csv(os.path.join(BASE_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(BASE_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(BASE_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(BASE_DIR, "y_test.csv"), index=False)
df.to_csv(os.path.join(BASE_DIR, "student_placement_engineered.csv"), index=False)
print(f"\nSaved train/test splits, scaler, and engineered dataset to: {BASE_DIR}")
