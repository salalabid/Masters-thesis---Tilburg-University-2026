import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset1"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset1"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
colors = ["#EB9698", "#008CA8"] 
df = pd.read_csv(RAW_PATH)

if 'department' in df.columns: # removing temp department
    df = df[df['department'].str.strip().str.lower() != 'temp'].copy()
df = df.dropna(subset=['last_evaluation']).copy() #remove missing evaluations

df['status_label'] = df['status'].map({'Employed': 'No Attrition', 'Left': 'Attrition'})
status_order = ['Attrition', 'No Attrition']

bins = [80, 120, 160, 200, 240, 280, 320]
labels = ['80-120', '120-160', '160-200', '200-240', '240-280', '280-320']
df['hours_bins'] = pd.cut(df['avg_monthly_hrs'], bins=bins, labels=labels)

diag_table = df.groupby(['hours_bins', 'status_label'], observed=False).size().unstack(fill_value=0)
diag_table = diag_table.reindex(columns=status_order)
diag_table['Total'] = diag_table.sum(axis=1)
diag_table['Attrition within group'] = (diag_table['Attrition'] / diag_table['Total'] * 100).round(1).astype(str) + '%'
diag_table['% of Total Attrition'] = (diag_table['Attrition'] / diag_table['Attrition'].sum() * 100).round(1).astype(str) + '%'
plt.figure(figsize=(12, 7))
ax = sns.countplot(data=df, x='hours_bins', hue='status_label', hue_order=status_order, palette=colors, edgecolor='white')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper right')
plt.title('Attrition Distribution by Average Monthly Hours (D1)', fontsize=15, pad=20)
plt.xlabel('Average Monthly Hours', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "07_D1_Avg_Monthly_Hrs.png"), dpi=300)
diag_table.to_csv(os.path.join(TABLE_PATH, "Table_5_D1_Avg_Monthly_Hrs.csv"))