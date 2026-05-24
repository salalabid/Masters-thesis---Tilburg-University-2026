import pandas as pd
import numpy as np
from ctgan import CTGAN 
import os
import time
import torch

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
DATA_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
MODEL_PATH = os.path.join(BASE_PATH, "Models")
OUTPUT_PATH = os.path.join(BASE_PATH, "Data", "augmented")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

EPOCHS = 100

def run_ctgan_100_with_check(filename, target_col):
    print(f"--- Running D1 CTGAN 100-Epoch Test + Overfitting Check ---")
    
    file_path = os.path.join(DATA_PATH, filename)
    df_real = pd.read_csv(file_path)
    
    # Identify Discrete Columns
    base_discrete = ['salary', target_col, 'recently_promoted', 'filed_complaint']
    dept_cols = [c for c in df_real.columns if c.startswith('department_')]
    discrete_cols = base_discrete + dept_cols

    # 1. TRAINING
    print(f"Training for {EPOCHS} epochs...")
    model = CTGAN(epochs=EPOCHS, batch_size=500, verbose=True, cuda=torch.cuda.is_available())
    model.fit(df_real, discrete_cols)

    # 2. ITERATIVE SAMPLING (50% Rule)
    count_0 = len(df_real[df_real[target_col] == 0])
    count_1 = len(df_real[df_real[target_col] == 1])
    needed = count_0 - count_1
    
    collected_leavers = []
    total_found = 0
    while total_found < needed:
        chunk = model.sample(needed * 5) 
        leavers = chunk[chunk[target_col] == 1]
        collected_leavers.append(leavers)
        total_found += len(leavers)
    
    synthetic_leavers = pd.concat(collected_leavers).head(needed).copy()

    # 3. OVERFITTING CHECK (Novelty Test)
    # This finds if any synthetic row is an EXACT copy of a real row
    # We compare only against the original real training data
    duplicates = pd.merge(df_real, synthetic_leavers, how='inner').shape[0]
    novelty_rate = ((needed - duplicates) / needed) * 100

    print("\n" + "="*30)
    print("      OVERFITTING REPORT      ")
    print("="*30)
    print(f"Synthetic Rows Generated: {needed}")
    print(f"Exact Duplicates Found:   {duplicates}")
    print(f"Novelty Rate:            {novelty_rate:.2f}%")
    print("="*30 + "\n")

    # 4. ASSEMBLY & SAVE
    df_aug = pd.concat([df_real, synthetic_leavers], axis=0, ignore_index=True)
    save_name = f"D1_Train_CTGAN_{EPOCHS}.csv"
    df_aug.to_csv(os.path.join(OUTPUT_PATH, save_name), index=False)
    print(f"SUCCESS: {save_name} saved.")

if __name__ == "__main__":
    run_ctgan_100_with_check("D1_Train.csv", "status")