import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pytorch_tabnet.tab_model import TabNetClassifier

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
D1_PROCESSED_DIR = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
REPORTS_DIR = os.path.join(BASE_PATH, "Reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
TARGET = "status"
def tabnet_shap():
    df_train = pd.read_csv(os.path.join(D1_PROCESSED_DIR, "D1_Train.csv"))
    df_test = pd.read_csv(os.path.join(D1_PROCESSED_DIR, "D1_Test.csv"))  
    X_train = df_train.drop(columns=[TARGET]).values
    y_train = df_train[TARGET].values
    X_test = df_test.drop(columns=[TARGET])    
    feature_names = list(X_test.columns)
    dept_cols = [i for i, name in enumerate(feature_names) if "department" in name.lower()]
    other_cols = [i for i, name in enumerate(feature_names) if "department" not in name.lower()]
    model = TabNetClassifier(n_d=16, n_a=16, mask_type="entmax", lambda_sparse=0.001, momentum=0.02, verbose=0, seed=42)
    model.fit(
        X_train=X_train, y_train=y_train,
        max_epochs=50, patience=10,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0, drop_last=False)
    background_summary = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background_summary)    
    X_test_slice = X_test.head(100)
    shap_values_raw = explainer.shap_values(X_test_slice.values)
    if isinstance(shap_values_raw, list):
        shap_class_values = np.asarray(shap_values_raw[1])
    else:
        shap_class_values = np.asarray(shap_values_raw)   
    if len(shap_class_values.shape) == 3:
        shap_class_values = shap_class_values[:, :, 1]   
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    if dept_cols:
        dept_values = shap_class_values[:, dept_cols].sum(axis=1, keepdims=True)
        other_values = shap_class_values[:, other_cols]
        new_values = np.hstack([other_values, dept_values])
        dept_data = X_test_slice.values[:, dept_cols].sum(axis=1, keepdims=True)
        other_data = X_test_slice.values[:, other_cols]
        new_data = np.hstack([other_data, dept_data])
        new_feature_names = [feature_names[i] for i in other_cols] + ["Department"]
    else:
        new_values = shap_class_values
        new_data = X_test_slice.values
        new_feature_names = feature_names
    clean_shap_values = shap.Explanation(
        values=new_values,
        base_values=base_val,
        data=new_data,
        feature_names=new_feature_names)
    mean_abs_shap = np.abs(clean_shap_values.values).mean(axis=0).flatten()
    df_rank = pd.DataFrame({
        'Feature': new_feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    print(df_rank.to_string(index=False))
    plt.figure(figsize=(10, 8))
    shap.plots.bar(clean_shap_values, max_display=clean_shap_values.shape[1] + 1, show=False)
    ax = plt.gca()
    for text in list(ax.texts):
        text.remove()
    for patch in ax.patches:
        patch.set_facecolor("#008CA8")
        patch.set_edgecolor("#008CA8")   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(REPORTS_DIR, "D1_SHAP_TabNet.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    tabnet_shap()