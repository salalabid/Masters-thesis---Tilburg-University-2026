import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset2"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)
D1_COLOR = "#008CA8"
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['Department'] = df['Department'].str.strip().str.title().replace({'It/Is': 'IT', 'It': 'IT'})

df['RaceDesc'] = df['RaceDesc'].str.strip()
race_dept_pct = pd.crosstab(df['RaceDesc'], df['Department'], normalize='index') * 100
race_dept_pct.to_csv(os.path.join(TABLE_PATH, "Table_26_D2_Race_Dept_Distribution_Pct.csv"))

plt.figure(figsize=(14, 8))
sns.heatmap(race_dept_pct, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Percentage of Race Group (%)'})

plt.title('Employee Concentration: % of Each Race per Department (D2)', fontsize=15, pad=20)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Race', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "14_D2_Race_Dept_Concentration.png"), dpi=300)