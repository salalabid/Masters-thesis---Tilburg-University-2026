import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset1"
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#EB9698", "#87CCDA"] 
df_raw = pd.read_csv(RAW_PATH)

if 'department' in df_raw.columns: #removing temp department
    df_diag = df_raw[df_raw['department'].str.strip().str.lower() != 'temp'].copy()

df_diag['Eval_Status'] = df_diag['last_evaluation'].isna().map({True: 'Missing', False: 'Present'})

mnar_plot_data = df_diag.groupby(['tenure', 'Eval_Status']).size().unstack(fill_value=0)
mnar_plot_data = mnar_plot_data[['Missing', 'Present']]

fig_mnar, ax_mnar = plt.subplots(figsize=(12, 7))
mnar_plot_data.plot(kind='bar', stacked=True, color=colors, ax=ax_mnar, edgecolor="white")

# legend
missing_patch = mpatches.Patch(color=colors[0], label='Yes')
present_patch = mpatches.Patch(color=colors[1], label='No')
ax_mnar.legend(handles=[missing_patch, present_patch], loc='upper right', title='Attrition')

# Add labels on the stacked portions
for p in ax_mnar.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    if height > 50:
        ax_mnar.annotate(f'{int(height)}', (x + width/2, y + height/2), 
                         ha='center', va='center', color='black')

plt.title('Missing Evaluation (D1)', fontsize=16, pad=20)
ax_mnar.set_xlabel('Tenure', fontsize=12)
ax_mnar.set_ylabel('Number of Employees', fontsize=12)
plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "01-1_D1_evaluation.png"), dpi=300)
plt.show()

df = df_diag.dropna(subset=['last_evaluation']).copy()

target = 'status'
mapping = {'Employed': 'No', 'Left': 'Yes'}
df[target] = df[target].map(mapping)

counts = df[target].value_counts().reindex(['Yes', 'No']).fillna(0)
total = len(df)
percentages = (counts / total) * 100

fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, 
            edgecolor="white", ax=ax2, legend=False)

attrition_patch = mpatches.Patch(color=colors[0], label='Yes')
no_attrition_patch = mpatches.Patch(color=colors[1], label='No')

plt.legend(handles=[attrition_patch, no_attrition_patch], title='Attrition', loc='upper left')

for i, (label, count) in enumerate(counts.items()):
    pct = percentages.iloc[i]
    ax2.text(i, count/2, f"{int(count)} ({pct:.1f}%)", 
            ha='center', va='center', fontsize=14, color='black')

plt.title('Employee Attrition: Count and Percentage (D1)', fontsize=16, pad=20)
ax2.set_xlabel('Attrition Outcome', fontsize=12)
ax2.set_ylabel('Number of Employees', fontsize=12)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "01_D1_Attrition.png"), dpi=300)