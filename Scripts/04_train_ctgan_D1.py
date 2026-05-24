import pandas as pd
import numpy as np
from ctgan import CTGAN 
import os

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
DATA_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
MODEL_PATH = os.path.join(BASE_PATH, "Models")
OUTPUT_PATH = os.path.join(BASE_PATH, "Data", "augmented")
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
epoch_milestones = [300, 500, 700]

def train_CTGAN(filename, target_col):
    df_real = pd.read_csv(os.path.join(DATA_PATH, filename))    
    base_discrete = ['salary', target_col, 'recently_promoted', 'filed_complaint']
    dept_cols = [c for c in df_real.columns if c.startswith('department_')]
    discrete_cols = base_discrete + [c for c in dept_cols if c in df_real.columns]

    count_0 = len(df_real[df_real[target_col] == 0]) #no attrition
    count_1 = len(df_real[df_real[target_col] == 1]) #attrition
    needed = count_0 - count_1

    for milestone in epoch_milestones:
        model = CTGAN(epochs=milestone, batch_size=500, verbose=True) #batch size 500 as dataset is 12000+
        model.fit(df_real, discrete_cols)
        model.save(os.path.join(MODEL_PATH, f'ctgan_d1_{milestone}.pkl'))

        all_synthetic_leavers = pd.DataFrame()
        while len(all_synthetic_leavers) < needed:
            batch = model.sample(2000)
            for col in discrete_cols:
                if col in batch.columns:
                    batch[col] = batch[col].clip(0,1).round().astype(int)
            leavers = batch[batch[target_col] == 1]
            all_synthetic_leavers = pd.concat([all_synthetic_leavers, leavers], axis=0)
        synthetic_leavers = all_synthetic_leavers.head(needed)

        df_aug = pd.concat([df_real, synthetic_leavers], axis=0, ignore_index=True)
        save_name = f"D1_Train_CTGAN_{milestone}.csv"
        df_aug.to_csv(os.path.join(OUTPUT_PATH, save_name), index=False)

if __name__ == "__main__":
    train_CTGAN("D1_Train.csv", "status")