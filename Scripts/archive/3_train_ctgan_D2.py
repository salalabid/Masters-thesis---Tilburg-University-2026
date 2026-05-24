import pandas as pd
from ctgan import CTGAN
import os

# ==========================================
# 0. SETUP
# ==========================================
DATA_PATH = 'D:/Masters/Thesis/Block 3/Thesis work/Data/processed/D2_Train.csv'
OUTPUT_MODEL_PATH = 'D:/Masters/Thesis/Block 3/Thesis work/Models/'
OUTPUT_SYN_DATA_PATH = 'D:/Masters/Thesis/Block 3/Thesis work/Data/synthetic/'

os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_SYN_DATA_PATH, exist_ok=True)

# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================
print("--- Loading Dataset 2 (Training Set) ---")
df = pd.read_csv(DATA_PATH)
print(f"Data Shape: {df.shape}")

# ==========================================
# 2. DEFINE DISCRETE COLUMNS
# ==========================================
# For Dataset 2, we have many categorical features.
# We must list ALL columns that are NOT continuous numbers.
discrete_columns = [
    'Target',             # The label
    'Sex',
    'MaritalDesc',
    'CitizenDesc',
    'HispanicLatino',
    'RaceDesc',
    'Position',           # High cardinality
    'ManagerID',          # Encoded, but effectively categorical
    'RecruitmentSource',
    'State',
    'PerformanceScore'    # Ordinal/Categorical
]

# Note: We treat 'Absences' and 'SpecialProjectsCount' as continuous 
# for the GAN to learn the distribution, and we will round them later.

print(f"Discrete Columns identified: {len(discrete_columns)}")
print(discrete_columns)

# ==========================================
# 3. INITIALIZE AND TRAIN CTGAN
# ==========================================
print("\n--- Initializing CTGAN for Dataset 2 ---")
# Dataset 2 is smaller (2,300 rows), so 300 epochs is plenty.
ctgan = CTGAN(epochs=300, batch_size=500, verbose=True)

print("--- Starting Training (This may take a few minutes) ---")
ctgan.fit(df, discrete_columns)

print("--- Training Complete! ---")

# ==========================================
# 4. SAVE THE MODEL
# ==========================================
model_filename = OUTPUT_MODEL_PATH + 'ctgan_d2_model.pkl'
ctgan.save(model_filename)
print(f"Model saved to: {model_filename}")

# ==========================================
# 5. GENERATE SAMPLE DATA
# ==========================================
synthetic_data = ctgan.sample(1000)

print("\n--- Synthetic Data Sample ---")
print(synthetic_data.head())

# Sanity Check on Target
print("\n--- Target Distribution in Synthetic Data ---")
print(synthetic_data['Target'].value_counts(normalize=True))

synthetic_data.to_csv(OUTPUT_SYN_DATA_PATH + 'D2_Synthetic_Sample.csv', index=False)