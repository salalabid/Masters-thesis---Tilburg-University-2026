import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
TABLE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Tables\dataset1"
GRAPH_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset1"
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
colors = ["#EB9698", "#008CA8"] 
df = pd.read_csv(RAW_PATH)

if 'department' in df.columns:
    df = df[df['department'].str.strip().str.lower() != 'temp'].copy()
df = df.dropna(subset=['last_evaluation']).copy()
df['status_label'] = df['status'].map({'Employed': 'Stayed', 'Left': 'Attrition'})

# High Evaluation (>0.8) AND High Workload (>240 hrs)
star_filter = (df['last_evaluation'] > 0.8) & (df['avg_monthly_hrs'] > 240)
stars_df = df[star_filter].copy()
total_workforce = len(df)
total_stars = len(stars_df)
star_attrition = stars_df['status_label'].value_counts()
attrition_rate = (star_attrition.get('Attrition', 0) / total_stars * 100)

summary_data = {
    "Metric": [
        "Total Workforce (N)", 
        "Total Star Employees (Eval > 0.8 & Hrs > 240)", 
        "Stars who Left (Attrition)", 
        "Stars who Stayed",
        "Attrition Rate within Star Group"
    ],
    "Value": [
        f"{total_workforce}",
        f"{total_stars}",
        f"{star_attrition.get('Attrition', 0)}",
        f"{star_attrition.get('Stayed', 0)}",
        f"{attrition_rate:.1f}%"
    ]
}
star_burnout_table = pd.DataFrame(summary_data)
plt.figure(figsize=(12, 8))
ax = sns.scatterplot(data=df, x='avg_monthly_hrs', y='last_evaluation', 
                hue='status_label', hue_order=['Attrition', 'Stayed'],
                palette={'Attrition': colors[0], 'Stayed': colors[1]},
                alpha=0.4, edgecolor=None)
plt.axvline(240, color='black', linestyle='--', alpha=0.3)
plt.axhline(0.8, color='black', linestyle='--', alpha=0.3)
plt.text(245, 0.82, f'Attrition Rate: {attrition_rate:.1f}%', 
         fontsize=11, fontweight='bold', color='#8B0000', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
yes_patch = mpatches.Patch(color=colors[0], label='Yes')
no_patch = mpatches.Patch(color=colors[1], label='No')
plt.legend(handles=[yes_patch, no_patch], title='Attrition', loc='upper left')
plt.title('Evaluation and Monthly Hours Worked)', fontsize=15, pad=20)
plt.xlabel('Average Monthly Hours', fontsize=12)
plt.ylabel('Last Evaluation Score', fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_PATH, "08.1_D1_Scatter(appendix).png"), dpi=300)
star_burnout_table.to_csv(os.path.join(TABLE_PATH, "Table_5_2_D1_Star_Burnout_DeepDive.csv"), index=False)