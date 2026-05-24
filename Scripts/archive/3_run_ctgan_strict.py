import pandas as pd
import numpy as np
from ctgan import CTGAN
import os

# ==========================================
# 0. CONFIGURATION
# ==========================================
BASE_PATH = 'D:/Masters/Thesis/Block 3/Thesis work/'
DATA_PATH = BASE_PATH + 'Data/processed/'
MODEL_PATH = BASE_PATH + 'Models/'
OUTPUT_PATH = BASE_PATH + 'Data/augmented/'

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Strict Hyperparameters from Xu et al. (2019) paper
CTGAN_PARAMS = {
    'epochs': 300,
    'batch_size': 500,
    'generator_dim': (256, 256),
    'discriminator_dim': (256, 256),
    'generator_lr': 2e-4,
    'discriminator_lr': 2e-4,
    'verbose': True
}

def run_ctgan_pipeline(dataset_name, filename, target_col, discrete_columns):
    print(f"\n==================================================")
    print(f"STARTING STRICT CTGAN PIPELINE: {dataset_name}")
    print(f"==================================================")
    
    # 1. Load Data
    df_real = pd.read_csv(DATA_PATH + filename)
    print(f"Loaded {dataset_name}: {df_real.shape}")

    # 2. Train Model
    print(f"--- Training with params: {CTGAN_PARAMS} ---")
    model = CTGAN(**CTGAN_PARAMS)
    model.fit(df_real, discrete_columns)
    
    # Save Model
    model.save(MODEL_PATH + f'ctgan_{dataset_name}_strict.pkl')
    print("--- Training Complete & Model Saved ---")

    # 3. Generate Synthetic Leavers
    count_0 = df_real[df_real[target_col] == 0].shape[0]
    count_1 = df_real[df_real[target_col] == 1].shape[0]
    needed = count_0 - count_1
    
    if needed <= 0:
        print("Dataset balanced. Skipping generation.")
        return

    print(f"Generating {needed} synthetic leavers to balance classes...")
    synthetic_data = model.sample(needed * 4)
    
    # Filter for Target = 1
    synthetic_leavers = synthetic_data[synthetic_data[target_col] == 1].copy()
    
    if len(synthetic_leavers) < needed:
        print(f"Warning: Only generated {len(synthetic_leavers)} valid leavers.")
    else:
        synthetic_leavers = synthetic_leavers.head(needed)

    # 4. Post-Processing (Consistency Checks)
    print("--- Post-Processing ---")
    if "1" in dataset_name: # Dataset 1
        if 'n_projects' in synthetic_leavers.columns:
            synthetic_leavers['n_projects'] = synthetic_leavers['n_projects'].round().astype(int)
        if 'tenure' in synthetic_leavers.columns:
            synthetic_leavers['tenure'] = synthetic_leavers['tenure'].round().astype(int)
        if 'avg_monthly_hrs' in synthetic_leavers.columns:
             synthetic_leavers['avg_monthly_hrs'] = synthetic_leavers['avg_monthly_hrs'].clip(
                 df_real['avg_monthly_hrs'].min(), df_real['avg_monthly_hrs'].max()
             )

    else: # Dataset 2 (Aligned to 17-column structure)
        if 'SpecialProjectsCount' in synthetic_leavers.columns:
            synthetic_leavers['SpecialProjectsCount'] = synthetic_leavers['SpecialProjectsCount'].round().clip(lower=0).astype(int)
        if 'DaysLateLast30' in synthetic_leavers.columns:
            synthetic_leavers['DaysLateLast30'] = synthetic_leavers['DaysLateLast30'].round().clip(lower=0).astype(int)
        if 'EngagementSurvey' in synthetic_leavers.columns:
            synthetic_leavers['EngagementSurvey'] = synthetic_leavers['EngagementSurvey'].round().clip(1, 5)
        if 'EmpSatisfaction' in synthetic_leavers.columns:
            synthetic_leavers['EmpSatisfaction'] = synthetic_leavers['EmpSatisfaction'].round().clip(1, 5)

    # 5. Save Final Augmented File
    df_aug = pd.concat([df_real, synthetic_leavers], axis=0)
    
    final_name = "D1_Train_CTGAN.csv" if "1" in dataset_name else "D2_Train_CTGAN.csv"
    df_aug.to_csv(OUTPUT_PATH + final_name, index=False)
    print(f"SAVED: {OUTPUT_PATH + final_name} (Shape: {df_aug.shape})")

# ==========================================
# EXECUTION
# ==========================================

# Dataset 1
d1_discrete = ['department', 'salary', 'status', 'filed_complaint', 'recently_promoted', 'missing_last_evaluation']
run_ctgan_pipeline("Dataset 1", "D1_Train.csv", "status", d1_discrete)

# Dataset 2 (Target + 10 Categorical Predictors = 11 Discrete Columns)
d2_discrete = [
    'Target', 'Sex', 'MaritalDesc', 'CitizenDesc', 
    'HispanicLatino', 'RaceDesc', 'Position', 'ManagerID', 
    'RecruitmentSource', 'State', 'PerformanceScore'
]
run_ctgan_pipeline("Dataset 2", "D2_Train.csv", "Target", d2_discrete)

print("\n--- ALL STRICT CTGAN PROCESSING COMPLETE ---")