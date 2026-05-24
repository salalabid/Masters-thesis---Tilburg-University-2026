import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
RAW_PATH = os.path.join(BASE_PATH, "Data", "raw", "kaggle5.csv")
TABLE_PATH = os.path.join(BASE_PATH, "Tables", "dataset2")
os.makedirs(TABLE_PATH, exist_ok=True)
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()

valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['attrition_binary'] = df['EmploymentStatus'].apply(lambda x: 1 if 'Terminated' in x else 0)
def calculate_cramers_v(feat):
    contingency = pd.crosstab(df[feat], df['attrition_binary'])
    chi2, p, dof, ex = chi2_contingency(contingency)
    n = contingency.sum().sum()
    # Cramér's V formula: sqrt(chi2 / (n * (min(cols, rows) - 1)))
    v_val = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    return v_val, p
nominal_features = ['Department', 'MaritalDesc', 'RaceDesc']
results = []
for feat in nominal_features:
    v, p = calculate_cramers_v(feat)
    results.append({
        "Variable": feat,
        "Cramér's V": round(v, 6),
        "p-value": "< 0.001" if p < 0.001 else round(p, 6)
    })
cramers_df = pd.DataFrame(results)
cramers_df.to_csv(os.path.join(TABLE_PATH, "Table_22_D2_CramersV_Summary.csv"), index=False)