import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# STRICT THESIS CONFIGURATION
# ==========================================
BASE_PATH = 'D:/Masters/Thesis/Block 3/Thesis work/'
RESULTS_PATH = BASE_PATH + 'Results/'
IMG_PATH = RESULTS_PATH + 'images/'
os.makedirs(IMG_PATH, exist_ok=True)

# 1. Load Results
csv_path = RESULTS_PATH + "final_thesis_results.csv"
if not os.path.exists(csv_path):
    print("Error: 'final_thesis_results.csv' not found. Run Script 8 first.")
    exit()

df = pd.read_csv(csv_path)

# 2. Strict Metrics Selection
# These match your base paper requirements exactly.
STRICT_METRICS = ["Accuracy", "F1", "PR-AUC"]

# 3. Setup Plotting Style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

def plot_strict_metric(dataset_name, metric):
    plt.figure(figsize=(10, 6))
    
    # Filter data for specific dataset
    data = df[df['Dataset'] == dataset_name]
    
    # Define exact order for consistent comparison
    # Baseline -> Traditional -> Generative
    technique_order = ["Baseline", "UnderSampled", "SMOTE", "CTGAN", "TVAEGAN"]
    
    # Create Chart
    ax = sns.barplot(
        data=data, 
        x='Model', 
        y=metric, 
        hue='Technique', 
        hue_order=technique_order,
        palette="viridis", # Professional academic color palette
        edgecolor="black",
        linewidth=1
    )
    
    # Formatting
    plt.title(f'{metric} - {dataset_name}', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel(metric, fontsize=12, fontweight='bold')
    plt.xlabel('Classifier', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.15) # Leave room for labels
    plt.legend(title='Augmentation', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add Score Labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=90)

    plt.tight_layout()
    
    # Save
    filename = f"{IMG_PATH}{dataset_name}_{metric}_Strict.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Generated Chart: {filename}")
    plt.close()

# --- Execution ---
print("Generating Strict Thesis Plots...")
for ds in ["D1", "D2"]:
    for metric in STRICT_METRICS:
        try:
            plot_strict_metric(ds, metric)
        except Exception as e:
            print(f"Skipping {metric} for {ds}: {e}")

print(f"\nDONE. Your charts are ready in: {IMG_PATH}")