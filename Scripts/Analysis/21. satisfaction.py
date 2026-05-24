import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset2"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
palette = {"Yes": "#EB9698", "No": "#87CCDA"}
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['Outcome'] = df['EmploymentStatus'].apply(lambda x: 'Yes' if 'Terminated' in x else 'No')
df['EmpSatisfaction'] = pd.to_numeric(df['EmpSatisfaction'], errors='coerce').fillna(0).astype(int)
df = df[df['EmpSatisfaction'] > 0].copy() # Filter out any 0s

sat_stats = df.groupby(['EmpSatisfaction', 'Outcome'], observed=False).size().unstack(fill_value=0)
sat_stats = sat_stats[['Yes', 'No']]
sat_stats.columns = ['Attrition', 'No Attrition']
sat_stats['Total'] = sat_stats['Attrition'] + sat_stats['No Attrition']
sat_stats['Attrition within group'] = (sat_stats['Attrition'] / sat_stats['Total']) * 100
total_exits = sat_stats['Attrition'].sum()
sat_stats['% of Total Attrition'] = (sat_stats['Attrition'] / total_exits) * 100
sat_stats_fmt = sat_stats.copy()
sat_stats_fmt['Attrition within group'] = sat_stats_fmt['Attrition within group'].map('{:.1f}%'.format)
sat_stats_fmt['% of Total Attrition'] = sat_stats_fmt['% of Total Attrition'].map('{:.1f}%'.format)
sat_stats_fmt.index.name = 'Satisfaction Score'

sat_stats_fmt.to_csv(os.path.join(TABLE_PATH, "Table_6_D2_Satisfaction.csv"))

plt.figure(figsize=(12, 7))
ax = sns.countplot(data=df, x='EmpSatisfaction', hue='Outcome', 
                   hue_order=['Yes', 'No'], palette=palette)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                    textcoords='offset points')
plt.title('Attrition Distribution by Employee Satisfaction Score (D2)', fontsize=16, pad=20)
plt.xlabel('Satisfaction Score (1-5)', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.legend(title='Attrition', loc='upper right')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "09_D2_Satisfaction.png"), dpi=300)