import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
os.makedirs(SAVE_PATH, exist_ok=True)
d2_color = "#87CCDA" 
perf_order = ['Exceeds', 'Fully Meets', 'Needs Improvement', 'PIP']
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()
valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()

df['Attrition'] = df['EmploymentStatus'].apply(lambda x: 'Yes' if 'Terminated' in x else 'No')

df['SpecialProjectsCount'] = pd.to_numeric(df['SpecialProjectsCount'], errors='coerce').fillna(0).astype(int)
leavers_zero_df = df[(df['SpecialProjectsCount'] == 0) & (df['Attrition'] == 'Yes')].copy()

perf_counts = leavers_zero_df['PerformanceScore'].value_counts().reindex(perf_order).fillna(0)
total_leavers = perf_counts.sum()
plt.figure(figsize=(12, 7))
ax = perf_counts.plot(kind='bar', color=d2_color, edgecolor='white', width=0.7)
for i, count in enumerate(perf_counts):
    percentage = (count / total_leavers) * 100
    ax.annotate(f'{percentage:.1f}%', 
                (i, count), 
                ha='center', va='bottom', 
                fontsize=12, 
                xytext=(0, 8),
                textcoords='offset points', 
                color='#333333')
plt.title('Performance Rating of Leavers with 0 Special Projects (D2)', fontsize=16, pad=30)
plt.xlabel('Performance Score', fontsize=12)
plt.ylabel('Number of Leavers', fontsize=12)
plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Appendix_E_D2_Leavers_0_Projects_Perf.png"), dpi=300)