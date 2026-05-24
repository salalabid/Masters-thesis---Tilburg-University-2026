import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
RAW_PATH = os.path.join(BASE_PATH, "Data", "raw", "kaggle5.csv")
SAVE_PATH = os.path.join(BASE_PATH, "Graphs", "dataset2")
TABLE_PATH = os.path.join(BASE_PATH, "Tables", "dataset2")
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
palette = {"Yes": "#EB9698", "No": "#87CCDA"}
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['Outcome'] = df['EmploymentStatus'].apply(lambda x: 'Yes' if 'Terminated' in x else 'No')
df['Lateness'] = df['DaysLateLast30'].apply(lambda x: 'Late' if x == 0 else 'On Time') #blank means no, filled (0) means yes
late_stats = df.groupby(['Lateness', 'Outcome'], observed=False).size().unstack(fill_value=0)
late_stats = late_stats[['Yes', 'No']]
late_stats.columns = ['Attrition', 'No Attrition']
late_stats['Total'] = late_stats['Attrition'] + late_stats['No Attrition']
late_stats['Attrition within group'] = (late_stats['Attrition'] / late_stats['Total'] * 100)
total_exits = late_stats['Attrition'].sum()
late_stats['% of Total Attrition'] = (late_stats['Attrition'] / total_exits * 100)
late_stats_fmt = late_stats.copy()
late_stats_fmt['Attrition within group'] = late_stats_fmt['Attrition within group'].map('{:.1f}%'.format)
late_stats_fmt['% of Total Attrition'] = late_stats_fmt['% of Total Attrition'].map('{:.1f}%'.format)
late_stats_fmt.index.name = 'Lateness Status'

late_stats_fmt.to_csv(os.path.join(TABLE_PATH, "Table_7_D2_Lateness.csv"))
plt.figure(figsize=(12, 8))
ax = sns.countplot(
    data=df, 
    x='Lateness', 
    order=['Late', 'On Time'],
    hue='Outcome', 
    hue_order=['Yes', 'No'], 
    palette=palette
)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                textcoords='offset points')
plt.title('Attrition Distribution by Binary Lateness (D2)', fontsize=16, pad=20)
plt.xlabel('Arrival Time', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.legend(title='Attrition', labels=['Yes', 'No'], loc='upper left')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "10_D2_Lateness.png"), dpi=300)