import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset2"
os.makedirs(SAVE_PATH, exist_ok=True)
colors = ["#EB9698", "#008CA8"] 
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()
valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['Department'] = df['Department'].str.strip().str.title().replace({'It/Is': 'IT', 'It': 'IT'})
df['RaceDesc'] = df['RaceDesc'].str.strip()
df['status_label'] = df['EmploymentStatus'].apply(lambda x: 'Yes' if 'Terminated' in x else 'No')
departments = df['Department'].unique()
n_depts = len(departments)

fig, axes = plt.subplots(nrows=(n_depts + 1) // 2, ncols=2, figsize=(16, 20))
axes = axes.flatten()
for i, dept in enumerate(departments):
    dept_df = df[df['Department'] == dept]
    ct = pd.crosstab(dept_df['RaceDesc'], dept_df['status_label'])
    for col in ['Yes', 'No']:
        if col not in ct.columns: ct[col] = 0
    ct = ct[['Yes', 'No']]
    ct.plot(kind='barh', stacked=True, color=colors, ax=axes[i], edgecolor='white', legend=False)
    axes[i].set_title(f"Department: {dept}", fontsize=14, weight='bold')
    axes[i].set_xlabel("Number of Employees")
    axes[i].set_ylabel("")
    for p in axes[i].patches:
        width = p.get_width()
        if width > 5:
            axes[i].annotate(f'{int(width)}', 
                             (p.get_x() + width/2, p.get_y() + 0.25),
                             ha='center', va='center', color='white', weight='bold')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

red_patch = mpatches.Patch(color=colors[0], label='Attrition (Yes)')
blue_patch = mpatches.Patch(color=colors[1], label='No Attrition (No)')
fig.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12)
plt.suptitle("Race-wise Attrition Volume Segmented by Department (D2)", fontsize=18, y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(SAVE_PATH, "15_D2_Race_Dept_Attrition_Faceted_Stacked.png"), dpi=300, bbox_inches='tight')