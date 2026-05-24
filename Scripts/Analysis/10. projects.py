import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset1"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset1"
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#EB9698", "#008CA8"] 
df = pd.read_csv(RAW_PATH)
if 'department' in df.columns:
    df = df[df['department'].str.strip().str.lower() != 'temp'].copy()
df = df.dropna(subset=['last_evaluation']).copy()
df['status_label'] = df['status'].map({'Employed': 'No Attrition', 'Left': 'Attrition'})
status_order = ['Attrition', 'No Attrition']

diag_table = df.groupby(['n_projects', 'status_label'], observed=False).size().unstack(fill_value=0)
diag_table = diag_table.reindex(columns=status_order)
diag_table['Total'] = diag_table.sum(axis=1)
diag_table['Attrition within group'] = (diag_table['Attrition'] / diag_table['Total'] * 100).round(1).astype(str) + '%'
diag_table['% of Total Attrition'] = (diag_table['Attrition'] / diag_table['Attrition'].sum() * 100).round(1).astype(str) + '%'
diag_table.to_csv(os.path.join(TABLE_PATH, "Table_9_D1_Projects.csv"))

plt.figure(figsize=(12, 7))
ax = sns.countplot(data=df, x='n_projects', hue='status_label', hue_order=status_order, palette=colors, edgecolor='white')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper right')
plt.title('Attrition by Number of Projects (D1)', fontsize=15, pad=20)
plt.xlabel('Number of Projects', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "10_D1_Projects.png"), dpi=300)