import pandas as pd
import numpy as np
import os
import torch
import warnings
import sys
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress Warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
PROCESSED_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
AUGMENTED_PATH = os.path.join(BASE_PATH, "Data", "augmented")
RESULTS_PATH = os.path.join(BASE_PATH, "Results")

os.makedirs(RESULTS_PATH, exist_ok=True)

# Files to Evaluate
FILES = {
    "Baseline (Imbalanced)": os.path.join(PROCESSED_PATH, "D1_Train.csv"),
    "Undersampling":         os.path.join(AUGMENTED_PATH, "D1_Train_UnderSampled.csv"),
    "SMOTE-Tomek":           os.path.join(AUGMENTED_PATH, "D1_Train_SMOTETomek.csv"),
    "CTGAN (Competitor)":    os.path.join(AUGMENTED_PATH, "D1_Train_CTGAN.csv"),
    "T-VAE-GAN (Ours)":      os.path.join(AUGMENTED_PATH, "D1_Train_TVAEGAN.csv")
}

# Fixed Validation/Test Sets (Always the same!)
VAL_FILE = os.path.join(PROCESSED_PATH, "D1_Val.csv")
TEST_FILE = os.path.join(PROCESSED_PATH, "D1_Test.csv")

TARGET = 'status'
CAT_COLS = [
    'salary', 'filed_complaint', 'recently_promoted', 
    'missing_last_evaluation', 'work_accident'
] 
# Note: 'department' is already OHE, so we don't list it here as categorical for LabelEncoding

def load_and_prep(train_path):
    print(f"   Loading: {os.path.basename(train_path)}...")
    if not os.path.exists(train_path):
        print("   [ERROR] File not found!")
        return None, None, None, None, None, None

    train = pd.read_csv(train_path)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].values
    X_val = val.drop(columns=[TARGET])
    y_val = val[TARGET].values
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].values

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_technique(technique_name, file_path):
    print(f"\n==================================================")
    print(f"EVALUATING: {technique_name}")
    print(f"==================================================")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prep(file_path)
    if X_train is None: return []

    # ----------------------------------------
    # 1. Preprocessing (Standard Scaling for SVM)
    # ----------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = []

    # ----------------------------------------
    # Model A: SVM (Distance Based)
    # ----------------------------------------
    print("   [1/3] Training SVM...")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        "Technique": technique_name, "Model": "SVM",
        "F1": f1_score(y_test, y_pred),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred)
    })

    # ----------------------------------------
    # Model B: XGBoost (Tree Based)
    # ----------------------------------------
    print("   [2/3] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        eval_metric='logloss', use_label_encoder=False, random_state=42
    )
    xgb.fit(X_train, y_train) # XGB handles unscaled data well
    
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    results.append({
        "Technique": technique_name, "Model": "XGBoost",
        "F1": f1_score(y_test, y_pred),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred)
    })

    # ----------------------------------------
    # Model C: TabNet (Deep Learning)
    # ----------------------------------------
    print("   [3/3] Training TabNet...")
    # TabNet needs numpy arrays
    X_train_np, X_val_np, X_test_np = X_train.values, X_val.values, X_test.values
    
    tabnet = TabNetClassifier(verbose=0, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2))
    
    try:
        tabnet.fit(
            X_train_np, y_train,
            eval_set=[(X_val_np, y_val)],
            eval_metric=['auc'],
            max_epochs=50, patience=15,
            batch_size=1024, virtual_batch_size=128
        )
        y_pred = tabnet.predict(X_test_np)
        y_prob = tabnet.predict_proba(X_test_np)[:, 1]
        
        results.append({
            "Technique": technique_name, "Model": "TabNet",
            "F1": f1_score(y_test, y_pred),
            "PR-AUC": average_precision_score(y_test, y_prob),
            "Recall": recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred)
        })
    except Exception as e:
        print(f"   [Warning] TabNet Failed: {e}")

    return results

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    final_results = []
    
    for name, path in FILES.items():
        final_results.extend(evaluate_technique(name, path))
        
    # Save Results
    df_res = pd.DataFrame(final_results)
    
    # Reorder columns for readability
    cols = ["Technique", "Model", "F1", "PR-AUC", "Recall", "Precision"]
    df_res = df_res[cols]
    
    save_path = os.path.join(RESULTS_PATH, "Final_Thesis_Results_D1.csv")
    df_res.to_csv(save_path, index=False)
    
    print("\n==================================================")
    print("RESULTS SUMMARY (Sorted by PR-AUC)")
    print("==================================================")
    print(df_res.sort_values(by="PR-AUC", ascending=False))
    print(f"\nSaved to: {save_path}")