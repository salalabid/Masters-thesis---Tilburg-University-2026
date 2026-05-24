import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset2"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
colors = ["#EB9698", "#87CCDA"] 
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['status'] = df['EmploymentStatus'].apply(lambda x: 1 if 'Terminated' in x else 0)
mapping = {1: 'Yes', 0: 'No'}
df['Attrition_Label'] = df['status'].map(mapping)
df['SpecialProjects'] = pd.to_numeric(df['SpecialProjectsCount'], errors='coerce')
df = df.dropna(subset=['SpecialProjects'])
df['SpecialProjects'] = df['SpecialProjects'].astype(int)

plt.figure(figsize=(16, 8))
ax = sns.countplot(data=df, 
                   x='SpecialProjects', 
                   hue='Attrition_Label', 
                   hue_order=['Yes', 'No'], 
                   palette=colors,
                   edgecolor="white")
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper right')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                    textcoords='offset points')
plt.title('Attrition Distribution by Special Projects (D2)', fontsize=16, pad=20)
plt.xlabel('Number of Special Projects', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "04_D2_SpecialProjects.png"), dpi=300)
table = pd.crosstab(df['SpecialProjects'], df['status'], dropna=False)
table.columns = ['No Attrition', 'Attrition']
table = table[['Attrition', 'No Attrition']]
table['Total'] = table['Attrition'] + table['No Attrition']
table['Attrition within group'] = (table['Attrition'] / table['Total']) * 100
total_exits = table['Attrition'].sum()
table['% of Total Attrition'] = (table['Attrition'] / total_exits) * 100
table = table.fillna(0)
table['Attrition within group'] = table['Attrition within group'].map('{:.1f}%'.format)
table['% of Total Attrition'] = table['% of Total Attrition'].map('{:.1f}%'.format)
table.index.name = 'Special Projects Count'
table.to_csv(os.path.join(TABLE_PATH, "Table_2_D2_SpecialProjects.csv"))