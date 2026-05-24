import pandas as pd
import numpy as np

# 1. Load the original training data to get the "Key" (Mean and Std Dev)
# We need this to turn Z-scores back into years
raw_train_path = r"D:\Masters\Thesis\Block 3\Thesis work\Data\processed\dataset2\D2_Train.csv"
augmented_path = r"D:\Masters\Thesis\Block 3\Thesis work\Data\augmented\D2_Train_TVAEGAN_900.csv"

# We need the original statistics from the training set
# (Note: In a real pipeline, you would save the 'scaler' object as a pickle file)
df_ref = pd.read_csv(raw_train_path)
mean_age = df_ref['Age'].mean()
std_age = df_ref['Age'].std()
mean_tenure = df_ref['Tenure'].mean()
std_tenure = df_ref['Tenure'].std()

# 2. Load the augmented data
df_aug = pd.read_csv(augmented_path)

# 3. Create a temporary 'Audit' dataframe with UNSCALED values
df_audit = df_aug.copy()
df_audit['Age_Raw'] = (df_aug['Age'] * std_age) + mean_age
df_audit['Tenure_Raw'] = (df_aug['Tenure'] * std_tenure) + mean_tenure

# 4. Run the Logic Check on RAW years
print(f"--- Logical Integrity Audit (UNSCALED): D2 TVAEGAN ---")
illogical = df_audit[df_audit['Age_Raw'] - df_audit['Tenure_Raw'] < 18]

print(f"Total Records: {len(df_aug)}")
print(f"Illogical Age-Tenure (Start <18yo): {len(illogical)} rows")

if len(illogical) == 0:
    print("✅ SUCCESS: All synthetic rows follow realistic HR logic.")
else:
    print(f"❌ WARNING: Found {len(illogical)} rows that violate logic.")
    print(illogical[['Age_Raw', 'Tenure_Raw']].head())

# 5. Show a sample of the 'reconstructed' employees
print("\nSAMPLE RECONSTRUCTED PROFILES:")
print(df_audit[['Age_Raw', 'Tenure_Raw']].head())