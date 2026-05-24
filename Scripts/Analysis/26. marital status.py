import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset2"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#EB9698", "#008CA8"] 
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['status_label'] = df['EmploymentStatus'].apply(lambda x: 'Attrition' if 'Terminated' in x else 'No Attrition')
status_order = ['Attrition', 'No Attrition']
df['MaritalDesc'] = df['MaritalDesc'].str.strip()
marital_table = df.groupby(['MaritalDesc', 'status_label'], observed=True).size().unstack(fill_value=0)
marital_table = marital_table.reindex(columns=status_order)
if 'Attrition' not in marital_table.columns:
    marital_table['Attrition'] = 0
marital_table['Total'] = marital_table.sum(axis=1)
marital_table['Attrition within group'] = (marital_table['Attrition'] / marital_table['Total'] * 100).round(1).astype(str) + '%'
marital_table['% of Total Attrition'] = (marital_table['Attrition'] / marital_table['Attrition'].sum() * 100).round(1).astype(str) + '%'
marital_table.to_csv(os.path.join(TABLE_PATH, "Table_27_D2_Marital_Dist.csv"))

plt.figure(figsize=(14, 8))
ax = sns.countplot(data=df, x='MaritalDesc', hue='status_label', hue_order=status_order, palette=colors, edgecolor='white')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper right')
plt.title('Attrition Distribution by Marital Status (D2)', fontsize=15, pad=20)
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "16_D2_MaritalStatus.png"), dpi=300)