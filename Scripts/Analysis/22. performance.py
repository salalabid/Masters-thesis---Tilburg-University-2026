import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib.patches as mpatches

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
RAW_PATH = os.path.join(BASE_PATH, "Data", "raw", "kaggle5.csv")
TABLE_PATH = os.path.join(BASE_PATH, "Tables", "dataset2")
SAVE_PATH = os.path.join(BASE_PATH, "Graphs", "dataset2")
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#EB9698", "#008CA8"] 

df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()
valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['status_label'] = df['EmploymentStatus'].apply(lambda x: 'Attrition' if 'Terminated' in x else 'No Attrition')
status_order = ['Attrition', 'No Attrition']

perf_order = ['PIP', 'Needs Improvement', 'Fully Meets', 'Exceeds']
df['PerformanceScore'] = pd.Categorical(df['PerformanceScore'], categories=perf_order, ordered=True)
df = df.dropna(subset=['PerformanceScore']).copy()
perf_map = {'PIP': 0, 'Needs Improvement': 1, 'Fully Meets': 2, 'Exceeds': 3}
target_map = {'Attrition': 1, 'No Attrition': 0}
df['perf_rank'] = df['PerformanceScore'].map(perf_map)
df['attrition_binary'] = df['status_label'].map(target_map)
corr_val, p_val = stats.spearmanr(df['perf_rank'], df['attrition_binary'])

spearman_data = {
    "Variable": ["Performance Score"],
    "Spearman's Rank (rs)": [round(corr_val, 6)],
    "p-value": ["< 0.001" if p_val < 0.001 else round(p_val, 6)]
}
spearman_table = pd.DataFrame(spearman_data)

perf_table = df.groupby(['PerformanceScore', 'status_label'], observed=True).size().unstack(fill_value=0)
perf_table = perf_table.reindex(columns=status_order)
perf_table['Total'] = perf_table.sum(axis=1)
perf_table['Attrition within group'] = (perf_table['Attrition'] / perf_table['Total'] * 100).round(1).astype(str) + '%'
perf_table['% of Total Attrition'] = (perf_table['Attrition'] / perf_table['Attrition'].sum() * 100).round(1).astype(str) + '%'

perf_table.to_csv(os.path.join(TABLE_PATH, "Table_8_D2_Performance.csv"))
spearman_table.to_csv(os.path.join(TABLE_PATH, "Table_8-1_D2_Spearman_Results.csv"), index=False)

plt.figure(figsize=(12, 8))
ax = sns.countplot(data=df, x='PerformanceScore', hue='status_label', hue_order=status_order, palette=colors, edgecolor='white')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=11)
attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper right')
plt.title('Attrition by Performance Score (D2)', fontsize=15, pad=20)
plt.xlabel('Performance Score', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.text(0.5, -0.15, f"Spearman Correlation: {corr_val:.4f} (p-value: {p_val:.4f})", 
         transform=ax.transAxes, ha="center", fontsize=10, color='gray', style='italic')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "11_D2_Performance.png"), dpi=300)