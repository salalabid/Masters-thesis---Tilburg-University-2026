import pandas as pd

# Path to your RAW data
path = 'D:/Masters/Thesis/Block 3/Thesis work/Data/raw/kaggle5.csv'
df = pd.read_csv(path)

# Count actual nulls (which we will later turn into -1)
manager_null_count = df['ManagerID'].isna().sum()
total_records = len(df)

print(f"Number of records with missing ManagerID (NaN): {manager_null_count}")
print(f"Percentage of total data: {(manager_null_count / total_records) * 100:.2f}%")