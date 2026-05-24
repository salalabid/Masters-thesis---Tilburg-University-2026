import pandas as pd

# Load data
INPUT_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()

# Drop the corrupted rows to ensure accurate math
df.dropna(subset=['EmploymentStatus'], inplace=True)

# UPDATED: Included RecruitmentSource in the high-cardinality analysis
target_enc_cols = ['State', 'ManagerID', 'Position', 'Department', 'RecruitmentSource']

print("--- Median Group Sizes for M-Value Optimization ---")
for col in target_enc_cols:
    if col in df.columns:
        counts = df[col].value_counts()
        print(f"{col}:")
        print(f"  Unique Categories: {len(counts)}")
        print(f"  Median Size (Suggested m): {counts.median():.1f}")
        print(f"  Min/Max: {counts.min()} / {counts.max()}\n")