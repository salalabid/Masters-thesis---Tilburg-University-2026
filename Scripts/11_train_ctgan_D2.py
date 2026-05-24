import pandas as pd
import numpy as np
from ctgan import CTGAN 
import os

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
DATA_DIR = os.path.join(BASE_PATH, "Data", "processed", "dataset2")
OUTPUT_PATH = os.path.join(BASE_PATH, "Data", "augmented")
MODEL_PATH = os.path.join(BASE_PATH, "Models", "ctgan")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

def train_CTGAN(filename, target_col):
    df_real = pd.read_csv(os.path.join(DATA_DIR, filename))    
    base_discrete = [target_col, 'PerformanceScore', 'EmpSatisfaction', 'DaysLateLast30', 'SpecialProjectsCount']
    ohe_filter = ('Department_', 'MaritalDesc_', 'RaceDesc_')
    ohe_features = [c for c in df_real.columns if c.startswith(ohe_filter)]
    discrete_cols = base_discrete + ohe_features

    count_0 = len(df_real[df_real[target_col] == 0]) # No Attrition
    count_1 = len(df_real[df_real[target_col] == 1]) # Attrition

    needed = count_0 - count_1

    model = CTGAN(epochs=300, batch_size=100, verbose=True)
    model.fit(df_real, discrete_cols)
    model.save(os.path.join(MODEL_PATH, 'ctgan_d2.pkl'))
    all_synthetic_leavers = pd.DataFrame()

    while len(all_synthetic_leavers) < needed:
        batch = model.sample(2000)
        for col in discrete_cols:
            if col in batch.columns:
                if col == 'PerformanceScore':
                    batch[col] = batch[col].clip(0, 3).round().astype(int)
                elif col == 'EmpSatisfaction':
                    batch[col] = batch[col].clip(1, 5).round().astype(int)
                elif col == 'SpecialProjectsCount':
                    batch[col] = batch[col].clip(0, 8).round().astype(int)
                else:
                    batch[col] = batch[col].clip(0, 1).round().astype(int)
        leavers = batch[batch[target_col] == 1]
        all_synthetic_leavers = pd.concat([all_synthetic_leavers, leavers], axis=0)
        print(f"Collected: {len(all_synthetic_leavers)} / {needed}")

    synthetic_leavers = all_synthetic_leavers.head(needed)
    df_aug = pd.concat([df_real, synthetic_leavers], axis=0, ignore_index=True)

    df_aug.to_csv(os.path.join(OUTPUT_PATH, "D2_Train_CTGAN.csv"), index=False)

if __name__ == "__main__":
    train_CTGAN("D2_Train.csv", "status")