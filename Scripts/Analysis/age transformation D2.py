import pandas as pd

RAW_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
REF_DATE = pd.to_datetime('2021-04-29')
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip().str.lower()

df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['dob_fixed'] = df['dob'].apply(lambda x: x - pd.DateOffset(years=100) if x.year > 2021 else x)
df['age_2021'] = REF_DATE.year - df['dob_fixed'].dt.year
under_18 = df[df['age_2021'] < 18]

if not under_18.empty:
    print(under_18[['dob', 'age_2021']].head())
else:
    print("None found. All employees are 18 or older based on the 2021 reference.")