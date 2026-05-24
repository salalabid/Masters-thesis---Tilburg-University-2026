import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
SAVE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset1"
os.makedirs(SAVE_PATH, exist_ok=True)
df = pd.read_csv(RAW_PATH)

if 'department' in df.columns: # removing temp department
    df = df[df['department'].str.strip().str.lower() != 'temp'].copy()
df = df.dropna(subset=['last_evaluation']).copy() #remove missing evaluation
df['filed_complaint'] = df['filed_complaint'].fillna(0).astype(int) #swap blanks with 0
if df['status'].dtype == 'object':
    df['status'] = df['status'].map({'Employed': 0, 'Left': 1}) #target variable
if 'recently_promoted' in df.columns: # make recently promoted numeric
    df['recently_promoted'] = df['recently_promoted'].fillna(0).astype(int)

d1_features = [
    'avg_monthly_hrs', 'filed_complaint', 'last_evaluation', 
    'n_projects', 'recently_promoted', 'satisfaction', 'tenure', 'status'
]
corr_matrix = df[d1_features].corr() #Pearson correlation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt=".3f", 
            cmap='RdBu_r', 
            center=0, 
            linewidths=.5, 
            cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numerical Features Correlated with Attrition (D1)', 
          fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "02_D1_Correlation_Matrix(appendix).png"), dpi=300)