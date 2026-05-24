import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
INPUT_PATH = os.path.join(BASE_PATH, "Data", "raw", "kaggle5.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "Data", "processed", "dataset2")
REF_DATE = pd.to_datetime("2021-04-29") 
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()
valid_statuses = ['Active', 'Voluntarily Terminated', 'Terminated for Cause']
df = df[df['EmploymentStatus'].isin(valid_statuses)].copy()
df['Department'] = df['Department'].str.strip().str.title()
df['DaysLateLast30'] = df['DaysLateLast30'].fillna(-1)
df['DaysLateLast30'] = df['DaysLateLast30'].map({0: 1, -1: 0})
df['status'] = df['EmploymentStatus'].apply(lambda x: 1 if 'Terminated' in x else 0)
df['DOB'] = pd.to_datetime(df['DOB'], dayfirst=True, errors='coerce')
df['DateofHire'] = pd.to_datetime(df['DateofHire'], dayfirst=True, errors='coerce')
df['DOB'] = df['DOB'].apply(lambda x: x - pd.DateOffset(years=100) if not pd.isna(x) and x.year > 2021 else x)
df.dropna(subset=['DOB', 'DateofHire', 'Department', 'PerformanceScore'], inplace=True)
df['Age'] = (REF_DATE - df['DOB']).dt.days // 365
df['Tenure'] = (REF_DATE - df['DateofHire']).dt.days // 365
df = df[(df['Age'] >= 18) & (df['Tenure'] >= 0)].copy()

keep_features = [
    'Department', 'MaritalDesc', 'PerformanceScore', 'RaceDesc', 
    'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 
    'Age', 'Tenure', 'DaysLateLast30', 'status']
df_final = df[keep_features].copy()
X = df_final.drop('status', axis=1)
y = df_final['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train = X_train.copy()
X_test = X_test.copy()
perf_map = {
    'PIP': 0,
    'Needs Improvement': 1,
    'Fully Meets': 2,
    'Exceeds': 3}

for subset in [X_train, X_test]:
    subset['PerformanceScore'] = subset['PerformanceScore'].map(perf_map)

cat_cols = ['Department', 'MaritalDesc', 'RaceDesc']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X_train[cat_cols])

def apply_ohe(df_in, enc, cols):
    encoded = enc.transform(df_in[cols])
    feat_names = enc.get_feature_names_out(cols)
    encoded_df = pd.DataFrame(encoded, columns=feat_names, index=df_in.index).astype(int)
    return pd.concat([df_in.drop(cols, axis=1), encoded_df], axis=1)

X_train = apply_ohe(X_train, encoder, cat_cols)
X_test  = apply_ohe(X_test,  encoder, cat_cols)
num_cols = ['EngagementSurvey', 'Age', 'Tenure']

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(OUTPUT_DIR, 'D2_Train.csv'), index=False)
pd.concat([X_test, y_test],  axis=1).to_csv(os.path.join(OUTPUT_DIR, 'D2_Test.csv'),  index=False)

print(f"Final Train Shape: {X_train.shape}")
print(f"Final Test Shape: {X_test.shape}")
print(f"Status Distribution: {y_train.value_counts(normalize=True).to_dict()}")