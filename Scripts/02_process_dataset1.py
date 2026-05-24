import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

INPUT_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle1.csv"
OUTPUT_DIR = r"D:\Masters\Thesis\Block 3\Thesis work\Data\processed\dataset1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: Input file not found at {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()

df = df[df['department'] != 'temp']
df = df.dropna(subset=['last_evaluation'])
df['department'] = df['department'].replace('information_technology', 'IT')
df['filed_complaint'] = df['filed_complaint'].fillna(0)
df['recently_promoted'] = df['recently_promoted'].fillna(0)
df['status'] = df['status'].map({'Employed': 0, 'Left': 1})

X = df.drop('status', axis=1)
y = df['status']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp #10% of total is 12.5% of remaining 80% data
)

salary_map = {'low': 0, 'medium': 1, 'high': 2} #ordinal salary
for subset in [X_train, X_val, X_test]:
    subset['salary'] = subset['salary'].map(salary_map)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') #one-hot department
encoder.fit(X_train[['department']])
def apply_ohe(df_in, enc):
    encoded_cols = enc.transform(df_in[['department']])
    feature_names = enc.get_feature_names_out(['department'])
    encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df_in.index)
    return pd.concat([df_in.drop('department', axis=1), encoded_df], axis=1)
X_train = apply_ohe(X_train, encoder)
X_val   = apply_ohe(X_val, encoder)
X_test  = apply_ohe(X_test, encoder)

num_cols = ['avg_monthly_hrs', 'n_projects', 'last_evaluation', 'satisfaction', 'tenure'] #standard scaling
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols]   = scaler.transform(X_val[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(OUTPUT_DIR, 'D1_Train.csv'), index=False)
pd.concat([X_val, y_val],   axis=1).to_csv(os.path.join(OUTPUT_DIR, 'D1_Val.csv'),   index=False)
pd.concat([X_test, y_test],  axis=1).to_csv(os.path.join(OUTPUT_DIR, 'D1_Test.csv'),  index=False)