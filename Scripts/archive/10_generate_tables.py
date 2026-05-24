import pandas as pd
import os

# ==========================================
# SETUP
# ==========================================
BASE_PATH = 'D:/Masters/Thesis/Block 3/Thesis work/'
RESULTS_PATH = BASE_PATH + 'Results/'
CSV_PATH = RESULTS_PATH + "final_thesis_results.csv"

if not os.path.exists(CSV_PATH):
    print(f"Error: File not found at {CSV_PATH}")
    exit()

# Load Data
df = pd.read_csv(CSV_PATH)

# ==========================================
# HELPER FUNCTION TO PRINT ACADEMIC TABLES
# ==========================================
def print_academic_table(dataset_name, title):
    print(f"\n\n### {title}")
    print("| Classifier | Augmentation Strategy | Accuracy | F1-Score | PR-AUC |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    # Filter for the specific dataset
    subset = df[df['Dataset'] == dataset_name].copy()
    
    # Define Sorting Order
    # 1. Models: SVM -> XGBoost -> TabNet
    # 2. Techniques: Baseline -> UnderSampled -> SMOTE -> CTGAN -> TVAEGAN
    model_order = ["SVM", "XGBoost", "TabNet"]
    tech_order = ["Baseline", "UnderSampled", "SMOTE", "CTGAN", "TVAEGAN"]
    
    # Sort
    subset['Model_Rank'] = subset['Model'].apply(lambda x: model_order.index(x) if x in model_order else 99)
    subset['Tech_Rank'] = subset['Technique'].apply(lambda x: tech_order.index(x) if x in tech_order else 99)
    subset = subset.sort_values(by=['Model_Rank', 'Tech_Rank'])
    
    # Print Rows
    current_model = ""
    for _, row in subset.iterrows():
        # Handle Model Grouping (only print Model name once per group)
        model_label = f"**{row['Model']}**" if row['Model'] != current_model else ""
        current_model = row['Model']
        
        # Format metrics to 3 decimal places (Standard Academic Format)
        acc = f"{row['Accuracy']:.3f}"
        f1 = f"{row['F1']:.3f}"
        prauc = f"{row['PR-AUC']:.3f}"
        
        print(f"| {model_label} | {row['Technique']} | {acc} | {f1} | {prauc} |")

# ==========================================
# EXECUTION
# ==========================================
print("==================================================")
print("GENERATING ACADEMIC TABLES FOR THESIS")
print("==================================================")

print_academic_table("D1", "Table 1: Detailed Classification Performance on Dataset 1 (Low-Dimensional)")
print_academic_table("D2", "Table 2: Detailed Classification Performance on Dataset 2 (High-Dimensional)")

print("\n\n==================================================")
print("INSTRUCTIONS:")
print("1. Copy the tables above.")
print("2. Paste them directly into your Thesis Word Doc or Markdown editor.")
print("==================================================")