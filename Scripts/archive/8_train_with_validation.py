import pandas as pd
import numpy as np
import os
import torch
import warnings

# Models
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# Metrics & Preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. CONFIGURATION
# ==========================================
BASE_PATH = 'D:/Masters/Thesis/Block 3/Thesis work/'
PROCESSED_PATH = BASE_PATH + 'Data/processed/'
AUGMENTED_PATH = BASE_PATH + 'Data/augmented/'
RESULTS_PATH = BASE_PATH + 'Results/'

os.makedirs(RESULTS_PATH, exist_ok=True)

# Dataset Configuration
DATASETS = {
    "D1": {
        "val_file": "D1_Val.csv",
        "test_file": "D1_Test.csv",
        "target": "status",
        "cat_cols": ['department', 'salary', 'filed_complaint', 'recently_promoted', 'missing_last_evaluation'],
        "train_files": {
            "Baseline": "D1_Train.csv",
            "CTGAN": "D1_Train_CTGAN.csv",
            "TVAEGAN": "D1_Train_TVAEGAN.csv",
            "SMOTE": "D1_Train_SMOTETomek.csv",
            "UnderSampled": "D1_Train_UnderSampled.csv"
        }
    },
    "D2": {
        "val_file": "D2_Val.csv",
        "test_file": "D2_Test.csv",
        "target": "Target",
        # Updated cat_cols for Dataset 2 (Total 11 categorical predictors)
        "cat_cols": ['Sex', 'MaritalDesc', 'CitizenDesc', 'HispanicLatino', 'RaceDesc', 
                     'Position', 'ManagerID', 'RecruitmentSource', 'State', 
                     'PerformanceScore', 'Department'],
        "train_files": {
            "Baseline": "D2_Train.csv",
            "CTGAN": "D2_Train_CTGAN.csv",
            "TVAEGAN": "D2_Train_TVAEGAN.csv",
            "SMOTE": "D2_Train_SMOTETomek.csv",
            "UnderSampled": "D2_Train_UnderSampled.csv"
        }
    }
}

def evaluate_predictions(y_true, y_pred, y_prob):
    res = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else 0,
        "PR-AUC": average_precision_score(y_true, y_prob) if y_prob is not None else 0
    }
    return res

def run_thesis_evaluation():
    print("==================================================")
    print("STARTING THESIS EVALUATION (Clean 17-Feature Set)")
    print("==================================================")
    
    all_results = []

    for ds_name, config in DATASETS.items():
        print(f"\n--- Processing Dataset: {ds_name} ---")
        
        val_df = pd.read_csv(PROCESSED_PATH + config["val_file"])
        test_df = pd.read_csv(PROCESSED_PATH + config["test_file"])
        
        for technique, filename in config["train_files"].items():
            print(f"  Technique: {technique}...", end=" ", flush=True)
            
            path = (AUGMENTED_PATH if technique != "Baseline" else PROCESSED_PATH) + filename
            if not os.path.exists(path):
                print(f"[Skipping] {filename}")
                continue
                
            train_df = pd.read_csv(path)
            
            # Preprocessing
            X_train = train_df.drop(columns=[config["target"]])
            y_train = train_df[config["target"]].values
            X_val = val_df.drop(columns=[config["target"]])
            y_val = val_df[config["target"]].values
            X_test = test_df.drop(columns=[config["target"]])
            y_test = test_df[config["target"]].values
            
            # Label Encoding & Alignment
            X_train_enc = X_train.copy()
            X_val_enc = X_val.copy()
            X_test_enc = X_test.copy()
            
            cat_idxs = []
            cat_dims = []
            
            for col in X_train.columns:
                if col in config["cat_cols"]:
                    # Rounding ensures GAN-generated floats are matched to their discrete labels
                    X_train_enc[col] = X_train_enc[col].round().astype(int)
                    X_val_enc[col] = X_val_enc[col].round().astype(int)
                    X_test_enc[col] = X_test_enc[col].round().astype(int)
                    
                    le = LabelEncoder()
                    combined = pd.concat([X_train_enc[col], X_val_enc[col], X_test_enc[col]], axis=0)
                    le.fit(combined)
                    
                    X_train_enc[col] = le.transform(X_train_enc[col])
                    X_val_enc[col] = le.transform(X_val_enc[col])
                    X_test_enc[col] = le.transform(X_test_enc[col])
                    
                    cat_idxs.append(X_train.columns.get_loc(col))
                    cat_dims.append(len(le.classes_))
                else:
                    X_train_enc[col] = X_train_enc[col].astype(float)
                    X_val_enc[col] = X_val_enc[col].astype(float)
                    X_test_enc[col] = X_test_enc[col].astype(float)

            # Scaling (Specifically for SVM)
            scaler = StandardScaler()
            num_cols = [c for c in X_train.columns if c not in config["cat_cols"]]
            X_train_scaled = X_train_enc.copy()
            X_val_scaled = X_val_enc.copy()
            X_test_scaled = X_test_enc.copy()
            
            if num_cols:
                scaler.fit(X_train_enc[num_cols])
                X_train_scaled[num_cols] = scaler.transform(X_train_enc[num_cols])
                X_val_scaled[num_cols] = scaler.transform(X_val_enc[num_cols])
                X_test_scaled[num_cols] = scaler.transform(X_test_enc[num_cols])

            # --- Model A: SVM ---
            clf_svm = SVC(probability=True, random_state=42) 
            clf_svm.fit(X_train_scaled, y_train)
            prob_svm = clf_svm.predict_proba(X_test_scaled)[:, 1]
            res_svm = evaluate_predictions(y_test, clf_svm.predict(X_test_scaled), prob_svm)
            res_svm.update({"Dataset": ds_name, "Technique": technique, "Model": "SVM"})
            all_results.append(res_svm)

            # --- Model B: XGBoost ---
            clf_xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=20, eval_metric='logloss', random_state=42)
            clf_xgb.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)
            prob_xgb = clf_xgb.predict_proba(X_test_enc)[:, 1]
            res_xgb = evaluate_predictions(y_test, clf_xgb.predict(X_test_enc), prob_xgb)
            res_xgb.update({"Dataset": ds_name, "Technique": technique, "Model": "XGBoost"})
            all_results.append(res_xgb)

            # --- Model C: TabNet ---
            clf_tab = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
                                        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                                        scheduler_params={"step_size":10, "gamma":0.9},
                                        scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax', verbose=0)
            try:
                clf_tab.fit(X_train=X_train_enc.values, y_train=y_train, eval_set=[(X_val_enc.values, y_val)], 
                            max_epochs=200, patience=20, batch_size=1024, virtual_batch_size=128)
                prob_tab = clf_tab.predict_proba(X_test_enc.values)[:, 1]
                res_tab = evaluate_predictions(y_test, clf_tab.predict(X_test_enc.values), prob_tab)
                res_tab.update({"Dataset": ds_name, "Technique": technique, "Model": "TabNet"})
                all_results.append(res_tab)
            except Exception as e:
                print(f"TabNet Error: {e}")

            print("Done.")

    # Save and Print Results
    results_df = pd.DataFrame(all_results)
    cols = ["Dataset", "Technique", "Model", "Accuracy", "F1", "PR-AUC", "ROC-AUC", "Precision", "Recall"]
    results_df = results_df[cols]
    results_df.to_csv(RESULTS_PATH + "final_thesis_results.csv", index=False)
    
    print("\n==================================================")
    print(f"RESULTS SAVED: {RESULTS_PATH}final_thesis_results.csv")
    print("==================================================")

if __name__ == "__main__":
    run_thesis_evaluation()