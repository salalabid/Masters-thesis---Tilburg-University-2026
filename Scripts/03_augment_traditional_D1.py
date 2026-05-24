import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import os

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
DATA_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
OUTPUT_PATH = os.path.join(BASE_PATH, "Data", "augmented")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def run_traditional_augmentation(filename, target_col):
    file_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    try:
        smt = SMOTETomek(random_state=42)
        X_smt, y_smt = smt.fit_resample(X, y)
        df_smt = pd.concat([X_smt, y_smt], axis=1)
        
        save_name = "D1_Train_SMOTETomek.csv"
        df_smt.to_csv(os.path.join(OUTPUT_PATH, save_name), index=False)
    except Exception as e:
        print(f"   SMOTE Failed: {e}")

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    df_rus = pd.concat([X_rus, y_rus], axis=1)
    
    save_name = "D1_Train_UnderSampled.csv"
    df_rus.to_csv(os.path.join(OUTPUT_PATH, save_name), index=False)

if __name__ == "__main__":
    run_traditional_augmentation("D1_Train.csv", "status")