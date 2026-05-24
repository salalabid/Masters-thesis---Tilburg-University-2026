import pandas as pd
import numpy as np
import os
import sys
import time
from src.tvaegan_synthesizer import TVAEGANSynthesizer

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
DATA_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
OUTPUT_PATH = os.path.join(BASE_PATH, "Data", "augmented")
os.makedirs(OUTPUT_PATH, exist_ok=True)
epoch_milestones = [300, 500, 700]

def train_tvaegan(filename, target_col):
    file_path = os.path.join(DATA_PATH, filename)
    df_real_train = pd.read_csv(file_path)
    count_0 = len(df_real_train[df_real_train[target_col] == 0]) #no attrition
    count_1 = len(df_real_train[df_real_train[target_col] == 1]) #attrition
    needed = count_0 - count_1
    for milestone in epoch_milestones:
        model = TVAEGANSynthesizer( #settings as per Anshelevich
            epochs=milestone,
            batch_size=500,
            w_reconstruct=10,
            w_regularize=1.0,
            s_generat=5,
            s_encoder=5,
            lr_encoder=0.00005,
            lr_critic=0.00005,
            lr_generat=0.00005,
            clip=0.01,
            dropout=0.1,
            ot_loss={"loss": "energy"})
        start_time = time.time()
        model.fit(df_real_train)
        print(f"Training Complete in {time.time() - start_time:.2f}s")
        all_synthetic_leavers = pd.DataFrame()
        while len(all_synthetic_leavers) < needed:
            batch = model.predict(samples=2000)
            leavers = batch[batch[target_col]==1]
            all_synthetic_leavers = pd.concat([all_synthetic_leavers, leavers], axis=0)
        synthetic_leavers = all_synthetic_leavers.head(needed).copy()
        discrete_flags = [target_col, 'recently_promoted', 'filed_complaint', 'salary']
        discrete_flags += [c for c in synthetic_leavers.columns if c.startswith('department_')]        
        for col in discrete_flags: #to convert leavers to integer 1 or 0
            if col in synthetic_leavers.columns:
                synthetic_leavers[col] = synthetic_leavers[col].clip(0,1).round().astype(int)
        df_aug = pd.concat([df_real_train, synthetic_leavers], axis=0, ignore_index=True)
        save_name = f"D1_Train_TVAEGAN{milestone}.csv"
        df_aug.to_csv(os.path.join(OUTPUT_PATH, save_name), index=False)

if __name__ == "__main__":
    train_tvaegan("D1_Train.csv", "status")