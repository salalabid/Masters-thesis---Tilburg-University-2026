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
df = df.dropna(subset=['last_evaluation']).copy() #remove missing evaluation

df['status_label'] = df['status'].map({'Employed': 'No Attrition', 'Left': 'Attrition'})
hue_order = ['Attrition', 'No Attrition']

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
          '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
df['satisfaction_bins'] = pd.cut(df['satisfaction'], bins=bins, labels=labels, include_lowest=True)
diag_table = df.groupby(['satisfaction_bins', 'status_label'], observed=False).size().unstack(fill_value=0)
diag_table['Total'] = diag_table['No Attrition'] + diag_table['Attrition']
diag_table['Attrition within group'] = (diag_table['Attrition'] / diag_table['Total'] * 100).round(1).astype(str) + '%'
diag_table['% of Total Attrition'] = (diag_table['Attrition'] / diag_table['Attrition'].sum() * 100).round(1).astype(str) + '%'

plt.figure(figsize=(14, 7))
ax = sns.countplot(data=df, x='satisfaction_bins', hue='status_label', 
                   hue_order=hue_order, palette=colors, edgecolor='white')

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=9)

attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper left')
plt.title('Attrition Distribution by Satisfaction Level (D1)', fontsize=15, pad=20)
plt.xlabel('Satisfaction Level', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "04_D1_Satisfaction.png"), dpi=300)
diag_table.to_csv(os.path.join(TABLE_PATH, "Table_2_D1_Satisfaction.csv"))