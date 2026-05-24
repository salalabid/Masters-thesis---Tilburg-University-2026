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
df['EngagementSurvey'] = pd.to_numeric(df['EngagementSurvey'], errors='coerce')
df = df.dropna(subset=['EngagementSurvey']).copy()
df['EngagementGroup'] = pd.qcut(df['EngagementSurvey'], q=4, 
                               labels=['Low (Bottom 25%)', 'Below Avg', 'Above Avg', 'High (Top 25%)'])
eng_stats = df.groupby(['EngagementGroup', 'Outcome'], observed=False).size().unstack(fill_value=0)
eng_stats = eng_stats[['Yes', 'No']]
eng_stats.columns = ['Attrition', 'No Attrition']
eng_stats['Total'] = eng_stats['Attrition'] + eng_stats['No Attrition']
eng_stats['Attrition within group'] = (eng_stats['Attrition'] / eng_stats['Total']) * 100
total_exits = eng_stats['Attrition'].sum()
eng_stats['% of Total Attrition'] = (eng_stats['Attrition'] / total_exits) * 100
ranges = df.groupby('EngagementGroup', observed=False)['EngagementSurvey'].agg(['min', 'max'])
eng_stats = pd.concat([ranges, eng_stats], axis=1)
eng_stats_fmt = eng_stats.copy()
eng_stats_fmt['Attrition within group'] = eng_stats_fmt['Attrition within group'].map('{:.1f}%'.format)
eng_stats_fmt['% of Total Attrition'] = eng_stats_fmt['% of Total Attrition'].map('{:.1f}%'.format)
eng_stats_fmt.index.name = 'Engagement Category'
eng_stats_fmt.to_csv(os.path.join(TABLE_PATH, "Table_5_D2_Engagement.csv"))

plt.figure(figsize=(14, 8))
ax = sns.countplot(data=df, x='EngagementGroup', hue='Outcome', 
                   hue_order=['Yes', 'No'], palette=palette)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                    textcoords='offset points')
plt.title('Attrition Distribution by Engagement Survey Category (D2)', fontsize=16, pad=20)
plt.xlabel('Engagement Category', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.legend(title='Attrition', loc='upper right')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "08_D2_Engagement.png"), dpi=300)