import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset2"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
REF_DATE = pd.to_datetime("2021-04-29") 
palette = {"Yes": "#EB9698", "No": "#87CCDA"}
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['Outcome'] = df['EmploymentStatus'].apply(lambda x: 'Yes' if 'Terminated' in x else 'No')

df['DateofHire'] = pd.to_datetime(df['DateofHire'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['DateofHire']).copy()
df['Tenure_Float'] = (REF_DATE - df['DateofHire']).dt.days / 365.25
df = df[df['Tenure_Float'] >= 0].copy()

max_tenure = int(df['Tenure_Float'].max()) + 1
bins = list(range(0, max_tenure + 1))
labels = [f"{i}-{i+1} Year{'s' if i > 0 else ''}" for i in bins[:-1]]
df['Tenure Range'] = pd.cut(df['Tenure_Float'], bins=bins, labels=labels, right=False)

tenure_stats = df.groupby(['Tenure Range', 'Outcome'], observed=False).size().unstack(fill_value=0)
tenure_stats = tenure_stats[['Yes', 'No']]
tenure_stats.columns = ['Attrition', 'No Attrition']
tenure_stats['Total'] = tenure_stats['Attrition'] + tenure_stats['No Attrition']
tenure_stats['Attrition within group'] = (tenure_stats['Attrition'] / tenure_stats['Total']) * 100
total_exits = tenure_stats['Attrition'].sum()
tenure_stats['% of Total Attrition'] = (tenure_stats['Attrition'] / total_exits) * 100
tenure_stats_cleaned = tenure_stats[tenure_stats['Total'] > 0].copy()
tenure_stats_cleaned['Attrition within group'] = tenure_stats_cleaned['Attrition within group'].map('{:.1f}%'.format)
tenure_stats_cleaned['% of Total Attrition'] = tenure_stats_cleaned['% of Total Attrition'].map('{:.1f}%'.format)
tenure_stats_cleaned.index.name = 'Tenure'
tenure_stats_cleaned.to_csv(os.path.join(TABLE_PATH, "Table_3_D2_Tenure_Distribution.csv"))
df_plot = df[df['Tenure Range'].isin(tenure_stats_cleaned.index)].copy()
df_plot['Tenure Range'] = df_plot['Tenure Range'].astype(str)
plt.figure(figsize=(20, 9))
ax = sns.countplot(data=df_plot, 
                   x='Tenure Range', 
                   order=tenure_stats_cleaned.index.astype(str).tolist(),
                   hue='Outcome', 
                   hue_order=['Yes', 'No'], 
                   palette=palette)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                    textcoords='offset points')
plt.title('Attrition Distribution by Tenure (D2)', fontsize=16, pad=20)
plt.xlabel('Tenure', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.legend(title='Attrition', loc='upper right')
plt.xticks(rotation=45)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "05_D2_Tenure.png"), dpi=300)