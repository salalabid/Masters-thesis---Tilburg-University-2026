import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset2"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
REF_DATE = pd.to_datetime('2021-04-29')
palette = {"Yes": "#EB9698", "No": "#87CCDA"}
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['Outcome'] = df['EmploymentStatus'].apply(lambda x: 'Yes' if 'Terminated' in x else 'No')

df['DOB'] = pd.to_datetime(df['DOB'], dayfirst=True, errors='coerce')
df['DOB'] = df['DOB'].apply(lambda x: x - pd.DateOffset(years=100) if pd.notnull(x) and x.year > 2021 else x)
df['Age'] = REF_DATE.year - df['DOB'].dt.year

age_bins = [29, 34, 39, 44, 49, 54, 59, 64, 71] 
age_labels = ['29-33', '34-38', '39-43', '44-48', '49-53', '54-58', '59-63', '64-70']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
age_stats = df.groupby(['AgeGroup', 'Outcome'], observed=False).size().unstack(fill_value=0)
age_stats = age_stats[['Yes', 'No']]
age_stats.columns = ['Attrition', 'No Attrition']
age_stats['Total'] = age_stats['Attrition'] + age_stats['No Attrition']
age_stats['Attrition within group'] = (age_stats['Attrition'] / age_stats['Total']) * 100
total_exits = age_stats['Attrition'].sum()
age_stats['% of Total Attrition'] = (age_stats['Attrition'] / total_exits) * 100

age_stats_fmt = age_stats[age_stats['Total'] > 0].copy()
age_stats_fmt['Attrition within group'] = age_stats_fmt['Attrition within group'].map('{:.1f}%'.format)
age_stats_fmt['% of Total Attrition'] = age_stats_fmt['% of Total Attrition'].map('{:.1f}%'.format)
age_stats_fmt.index.name = 'Age Group'
age_stats_fmt.to_csv(os.path.join(TABLE_PATH, "Table_4_D2_Age_Distribution.csv"))

plt.figure(figsize=(15, 8))
ax = sns.countplot(data=df, x='AgeGroup', hue='Outcome', 
                   hue_order=['Yes', 'No'], palette=palette)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                    textcoords='offset points')

plt.title('Attrition Distribution by Age Group (D2)', fontsize=16, pad=20)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.legend(title='Attrition', loc='upper right')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "07_D2_Age.png"), dpi=300)