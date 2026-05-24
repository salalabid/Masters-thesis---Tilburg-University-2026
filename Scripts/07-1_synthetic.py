import pandas as pd
import numpy as np
import os
from ctgan import CTGAN
from src.tvaegan_synthesizer import TVAEGANSynthesizer

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
DATA_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
OUTPUT_PATH = os.path.join(BASE_PATH, "Data", "augmented")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def generate_synthetic_data(filename, target_col):
    df_real = pd.read_csv(os.path.join(DATA_PATH, filename))

    base_discrete = [target_col, 'salary', 'recently_promoted', 'filed_complaint']
    dept_cols = [c for c in df_real.columns if c.startswith('department_')]
    discrete_cols = base_discrete + [c for c in dept_cols if c in df_real.columns]

    ctgan = CTGAN(epochs=300, batch_size=500, discriminator_steps=5, verbose=True)
    ctgan.fit(df_real, discrete_cols)
    
    c_synth_0, c_synth_1 = pd.DataFrame(), pd.DataFrame()
    while len(c_synth_0) < 5000 or len(c_synth_1) < 5000:
        batch = ctgan.sample(2000)
        for col in discrete_cols:
            if col in batch.columns: batch[col] = batch[col].clip(0, 1).round().astype(int)
        
        s0, s1 = batch[batch[target_col] == 0], batch[batch[target_col] == 1]
        c_synth_0 = pd.concat([c_synth_0, s0]).head(5000)
        c_synth_1 = pd.concat([c_synth_1, s1]).head(5000)

    pd.concat([c_synth_0, c_synth_1]).to_csv(os.path.join(OUTPUT_PATH, "D1_Synthetic_CTGAN.csv"), index=False)

    tvae = TVAEGANSynthesizer(epochs=300, batch_size=500, w_reconstruct=10, w_regularize=1.0, s_generat=5, s_encoder=5, lr_encoder=0.00005, lr_critic=0.00005, lr_generat=0.00005, clip=0.01, dropout=0.1, ot_loss={"loss":"energy"})
    tvae.fit(df_real)
    
    t_synth_0, t_synth_1 = pd.DataFrame(), pd.DataFrame()
    while len(t_synth_0) < 5000 or len(t_synth_1) < 5000:
        batch = tvae.predict(samples=2000)
        for col in discrete_cols:
            if col in batch.columns: batch[col] = batch[col].clip(0, 1).round().astype(int)
        
        s0, s1 = batch[batch[target_col] == 0], batch[batch[target_col] == 1]
        t_synth_0 = pd.concat([t_synth_0, s0]).head(5000)
        t_synth_1 = pd.concat([t_synth_1, s1]).head(5000)

    pd.concat([t_synth_0, t_synth_1]).to_csv(os.path.join(OUTPUT_PATH, "D1_Synthetic_TVAEGAN.csv"), index=False)

if __name__ == "__main__":
    generate_synthetic_data("D1_Train.csv", "status")