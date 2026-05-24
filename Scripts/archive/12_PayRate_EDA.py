import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load raw data
df = pd.read_csv('D:/Masters/Thesis/Block 3/Thesis work/Data/raw/kaggle5.csv')
pay_data = df['PayRate'].dropna().values.reshape(-1, 1)
target_data = df['Termd'].dropna().values

# Identify Ranges
hourly = df[df['PayRate'] < 100]['PayRate']
monthly = df[df['PayRate'] >= 100]['PayRate']

print(f"Hourly Range: ${hourly.min()} - ${hourly.max()}")
print(f"Monthly Range: ${monthly.min()} - ${monthly.max()}")

# Approximate the 'Decision Line' using a single-split tree
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(pay_data, target_data)
print(f"Approximate Algorithm Split Point: ${dt.tree_.threshold[0]:.2f}")