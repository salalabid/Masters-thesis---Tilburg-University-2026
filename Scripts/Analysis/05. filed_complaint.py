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

if 'department' in df.columns: #removing temp department
    df = df[df['department'].str.strip().str.lower() != 'temp'].copy()
df = df.dropna(subset=['last_evaluation']).copy() #remove missing evaluations

df['filed_complaint'] = df['filed_complaint'].fillna(0).astype(int) #replacing blanks with 0
df['complaint_label'] = df['filed_complaint'].map({0: 'No Complaint', 1: 'Filed Complaint'})
df['status_label'] = df['status'].map({'Employed': 'No Attrition', 'Left': 'Attrition'})
hue_order = ['Attrition', 'No Attrition']

diag_table = df.groupby(['complaint_label', 'status_label'], observed=False).size().unstack(fill_value=0)
diag_table['Total'] = diag_table['No Attrition'] + diag_table['Attrition']
diag_table['Attrition within group'] = (diag_table['Attrition'] / diag_table['Total'] * 100).round(1).astype(str) + '%'
diag_table['% of Total Attrition'] = (diag_table['Attrition'] / diag_table['Attrition'].sum() * 100).round(1).astype(str) + '%'

plt.figure(figsize=(10, 7))
ax = sns.countplot(data=df, x='complaint_label', hue='status_label', 
                   hue_order=hue_order, palette=colors, edgecolor='white')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=11)
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper right')
plt.title('Attrition Distribution by Complaint Status (D1)', fontsize=15, pad=20)
plt.xlabel('Complaint Status', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "05_D1_Filed_Complaint.png"), dpi=300)
diag_table.to_csv(os.path.join(TABLE_PATH, "Table_3_D1_Filed_Complaint.csv"))