import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
GRAPH_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset2"
REF_DATE = pd.to_datetime('2021-04-29')
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
D2_COLOR = "#87CCDA" 
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

df['DOB'] = pd.to_datetime(df['DOB'], dayfirst=True, errors='coerce')
df['DOB'] = df['DOB'].apply(lambda x: x - pd.DateOffset(years=100) if x.year > 2021 else x)
df['Age'] = REF_DATE.year - df['DOB'].dt.year
df['DateofHire'] = pd.to_datetime(df['DateofHire'], dayfirst=True, errors='coerce')
df['Tenure'] = (REF_DATE - df['DateofHire']).dt.days / 365.25

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['status'] = df['EmploymentStatus'].apply(lambda x: 1 if 'Terminated' in x else 0)

d2_features = [
    'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 
    'Age', 'Tenure'
]
df = df[(df['Age'] >= 18) & (df['Tenure'] >= 0)].copy()

correlations = df[d2_features + ['status']].corr()['status'].abs()
abs_corr = correlations.drop('status').sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('white')
ax.xaxis.grid(True, linestyle='--', color='grey', alpha=0.3, zorder=0)
sns.barplot(x=abs_corr.values, y=abs_corr.index, hue=abs_corr.index, 
            palette='dark:#87CCDA', edgecolor="white", ax=ax, legend=False, zorder=3)
plt.title('Top Numerical Features based on Absolute Correlation with Employee Attrition (D2)', fontsize=14, pad=20)
ax.set_xlabel('Absolute Correlation Coefficient', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_PATH, "03_D2_Absolute_Correlation.png"), dpi=300)

desc_stats = df[d2_features].describe().T
desc_stats = desc_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc_stats.columns = ['Count', 'Mean', 'Std', 'Min', '25th', '50th', '75th', 'Max']
pd.options.display.float_format = '{:,.2f}'.format
desc_stats.to_csv(os.path.join(TABLE_PATH, "Table_1_D2_Descriptive_Stats.csv"))