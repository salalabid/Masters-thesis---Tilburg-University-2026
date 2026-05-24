import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
REF_DATE = pd.to_datetime('2021-04-29')
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#EB9698", "#87CCDA"] 
df_raw = pd.read_csv(RAW_PATH)
df_raw.columns = df_raw.columns.str.strip()

df_raw['DOB'] = pd.to_datetime(df_raw['DOB'], dayfirst=True, errors='coerce')
df_raw['DOB'] = df_raw['DOB'].apply(lambda x: x - pd.DateOffset(years=100) if x.year > 2021 else x)
df_raw['Age'] = REF_DATE.year - df_raw['DOB'].dt.year
df_raw['DateofHire'] = pd.to_datetime(df_raw['DateofHire'], dayfirst=True, errors='coerce')
df_raw['Tenure'] = (REF_DATE - df_raw['DateofHire']).dt.days / 365.25

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df_raw[df_raw['EmploymentStatus'].isin(valid_statuses)].copy()
df = df[(df['Age'] >= 18) & (df['Tenure'] >= 0)].copy()
df['status'] = df['EmploymentStatus'].apply(lambda x: 1 if 'Terminated' in x else 0)
mapping = {1: 'Yes', 0: 'No'}
df['display_status'] = df['status'].map(mapping)

counts = df['display_status'].value_counts().reindex(['Yes', 'No']).fillna(0)
total = len(df)
percentages = (counts / total) * 100

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, 
            edgecolor="white", ax=ax, legend=False)
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper left')
for i, (label, count) in enumerate(counts.items()):
    pct = percentages.iloc[i]
    ax.text(i, count/2, f"{int(count)} ({pct:.1f}%)", 
            ha='center', va='center', fontsize=14, color='black')
plt.title('Employee Attrition: Count and Percentage (D2)', fontsize=16, pad=20)
ax.set_xlabel('Attrition Outcome', fontsize=12)
ax.set_ylabel('Number of Employees', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "01_D2_Attrition.png"), dpi=300)