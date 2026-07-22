"""
Student Placement Prediction - EDA

FOLDER SETUP:
    your_project_folder/
        eda.py                            <- this file
        student_placement_cleaned.csv     <- created by pipeline.py (run that first!)
        plots/                            <- created automatically, plots saved here
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # safe non-interactive backend; plots are saved to file, not popped up
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "student_placement_cleaned.csv")
PLOTS = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Could not find {INPUT_CSV}\n"
        "Run pipeline.py first — it creates student_placement_cleaned.csv from the raw dataset."
    )

df = pd.read_csv(INPUT_CSV)

num_cols = ["CGPA", "10th_Percentage", "12th_Percentage", "Internships",
            "Coding_Score", "Aptitude_Score", "Projects", "Backlogs"]

# 1. Histograms
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "01_histograms.png"), dpi=110)
plt.close()

# 2. Count plot: Placement
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Placement", hue="Placement", palette="Set2", legend=False)
plt.title("Placement Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "02_placement_countplot.png"), dpi=110)
plt.close()

# 3. Box plots: numeric vs Placement
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, num_cols):
    sns.boxplot(data=df, x="Placement", y=col, hue="Placement", ax=ax, palette="Set2", legend=False)
    ax.set_title(f"{col} vs Placement")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "03_boxplots_by_placement.png"), dpi=110)
plt.close()

# 4. Scatter: CGPA vs Coding_Score colored by Placement
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="CGPA", y="Coding_Score", hue="Placement", alpha=0.6, palette="Set1")
plt.title("CGPA vs Coding Score by Placement")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "04_scatter_cgpa_coding.png"), dpi=110)
plt.close()

# 5. Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "05_correlation_heatmap.png"), dpi=110)
plt.close()

# 6. Pair plot (subset, for speed/readability)
pp = sns.pairplot(df[["CGPA", "Coding_Score", "Aptitude_Score", "Backlogs", "Placement"]],
                   hue="Placement", palette="Set1", diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
pp.savefig(os.path.join(PLOTS, "06_pairplot.png"), dpi=110)
plt.close()

# 7. Violin plot: Backlogs vs Placement
plt.figure(figsize=(6, 5))
sns.violinplot(data=df, x="Placement", y="Backlogs", hue="Placement", palette="Set2", legend=False)
plt.title("Backlogs Distribution by Placement")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "07_violin_backlogs.png"), dpi=110)
plt.close()

# 8. Bar chart: Placement rate by Communication level
comm_order = ["Poor", "Average", "Good", "Excellent"]
rate = df.groupby("Communication")["Placement"].apply(lambda s: (s == "Yes").mean()).reindex(comm_order)
plt.figure(figsize=(6, 5))
rate.plot(kind="bar", color="#55A868")
plt.ylabel("Placement Rate")
plt.title("Placement Rate by Communication Level")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "08_bar_communication_placement_rate.png"), dpi=110)
plt.close()

# 9. Bar chart: Placement rate by Internship count
plt.figure(figsize=(6, 5))
rate2 = df.groupby("Internships")["Placement"].apply(lambda s: (s == "Yes").mean())
rate2.plot(kind="bar", color="#C44E52")
plt.ylabel("Placement Rate")
plt.title("Placement Rate by Number of Internships")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "09_bar_internships_placement_rate.png"), dpi=110)
plt.close()

print(f"All EDA plots saved to: {PLOTS}")

print("\nPlacement rate overall:", round((df['Placement']=='Yes').mean(),3))
target_numeric = (df["Placement"] == "Yes").astype(int)
print("\nCorrelation of each numeric feature with Placement (Yes=1):")
for col in num_cols:
    print(f"  {col}: {df[col].corr(target_numeric):.3f}")

print("\nMean CGPA - Placed vs Not:")
print(df.groupby("Placement")["CGPA"].mean())
print("\nMean Backlogs - Placed vs Not:")
print(df.groupby("Placement")["Backlogs"].mean())
