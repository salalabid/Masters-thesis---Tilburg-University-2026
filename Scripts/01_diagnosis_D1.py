import pandas as pd

INPUT_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    unique_vals = df[col].dropna().unique()
    print(f"{col}: {sorted([str(x) for x in unique_vals])}")
df_protocol = df[df['department'] != 'temp'].copy() 
diag_map = {'Employed': 0, 'Left': 1}
df_protocol['status_num'] = df_protocol['status'].map(diag_map)
total_pre = df_protocol['status_num'].value_counts()
missing_eval = df_protocol[df_protocol['last_evaluation'].isnull()]
missing_by_class = missing_eval['status_num'].value_counts()

for label, code in diag_map.items():
    total = total_pre.get(code, 0)
    missing = missing_by_class.get(code, 0)
    print(f"Class '{label}': {missing} missing out of {total} ({(missing/total)*100:.2f}% loss)")
df_final = df_protocol.dropna(subset=['last_evaluation'])
total_post = df_final['status_num'].value_counts()
final_minority_pct = (total_post.get(1) / len(df_final)) * 100

print(f"Total Records Remaining: {len(df_final)}")
print(f"Employed (Target 0): {total_post.get(0)}")
print(f"Left (Target 1):     {total_post.get(1)}")
print(f"Minority Class Proportion: {final_minority_pct:.2f}%")