import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset1"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset1"
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#EB9698", "#008CA8"] 
df = pd.read_csv(RAW_PATH)

if 'department' in df.columns:
    df['department'] = df['department'].str.strip().str.lower()
    dept_map = {
        'information_technology': 'IT',
        'it': 'IT'
    }
    df['department'] = df['department'].replace(dept_map)
    df['department'] = df['department'].str.title().replace({'It': 'IT'})
if 'department' in df.columns:
    df = df[df['department'].str.strip().str.upper() != 'TEMP'].copy()
df = df.dropna(subset=['last_evaluation']).copy()
df['status_label'] = df['status'].map({'Employed': 'No Attrition', 'Left': 'Attrition'})
status_order = ['Attrition', 'No Attrition']

contingency = pd.crosstab(df['department'], df['status']) #Cramer's V
chi2, p, dof, ex = chi2_contingency(contingency)
n = contingency.sum().sum()
v_val = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
cramers_data = {
    "Variable": ["Department"],
    "Cramér's V": [round(v_val, 6)],
    "p-value": ["< 0.001" if p < 0.001 else round(p, 6)]
}
cramers_results_table = pd.DataFrame(cramers_data)

dept_dist = df.groupby(['department', 'status_label'], observed=False).size().unstack(fill_value=0)
dept_dist = dept_dist.reindex(columns=status_order)
dept_dist['Total'] = dept_dist.sum(axis=1)
dept_dist['Attrition within group'] = (dept_dist['Attrition'] / dept_dist['Total'] * 100).round(1).astype(str) + '%'
dept_dist['% of Total Attrition'] = (dept_dist['Attrition'] / dept_dist['Attrition'].sum() * 100).round(1).astype(str) + '%'
dept_dist.to_csv(os.path.join(TABLE_PATH, "Table_11_D1_Department_Dist.csv"))
cramers_results_table.to_csv(os.path.join(TABLE_PATH, "Table_12_D1_CramersV.csv"), index=False)

plt.figure(figsize=(14, 8))
ax = sns.countplot(data=df, x='department', hue='status_label', hue_order=status_order, palette=colors, edgecolor='white')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=9)
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper right')
plt.title('Attrition by Department (D1)', fontsize=15, pad=20)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.xticks(rotation=45)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "12_D1_Department.png"), dpi=300)