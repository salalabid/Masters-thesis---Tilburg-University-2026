import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_PATH = r"D:\Masters\Thesis\Block 3\Thesis work\Data\raw\kaggle5.csv"
REF_DATE = pd.to_datetime("2021-04-29") 

print(f"--- Comprehensive Diagnosis: Dataset 2 ---")
df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()

# ==========================================
# 1. TARGET & ROW AUDIT
# ==========================================
print("\n[1] TARGET AUDIT: EmploymentStatus")
if 'EmploymentStatus' in df.columns:
    counts = df['EmploymentStatus'].value_counts()
    print(counts)
    print(f"Observation: 'Future Start' rows ({counts.get('Future Start', 0)}) should be removed in processing.")

# ==========================================
# 2. TEMPORAL FEATURE ENGINEERING AUDIT
# ==========================================
print("\n" + "="*50)
print("  SECTION 2: PERFORMANCE REVIEW RECENCY AUDIT  ")
print("="*50)

if 'LastPerformanceReview_Date' in df.columns:
    # Convert to datetime
    rev_date = pd.to_datetime(df['LastPerformanceReview_Date'], errors='coerce')
    
    # Calculate years since review for those who HAVE a date
    years_since = (REF_DATE - rev_date).dt.days / 365.25
    
    print(f"Valid Review Records: {years_since.notnull().sum()}")
    print(f"Missing Review Records: {years_since.isnull().sum()}")
    
    if years_since.notnull().sum() > 0:
        print(f"Min Years Since Review: {years_since.min():.2f}")
        print(f"Max Years Since Review: {years_since.max():.2f}")
        print(f"Mean Years Since Review: {years_since.mean():.2f}")
        
        # Recommendation Logic
        suggested_penalty = np.ceil(years_since.max() + 2)
        print(f"\nSTRATEGIC RECOMMENDATION:")
        print(f"Use {suggested_penalty} years as the 'Never Reviewed' penalty value.")
else:
    print("LastPerformanceReview_Date not found.")

# ==========================================
# 3. BEHAVIORAL FEATURE AUDIT
# ==========================================
print("\n[3] BEHAVIORAL AUDIT: Attendance")
if 'DaysLateLast30' in df.columns:
    missing_late = df['DaysLateLast30'].isnull().sum()
    print(f"DaysLateLast30 Missing: {missing_late} rows.")
    print(f"Action: These will be filled with 0 (No incidents) during processing.")

# ==========================================
# 4. CARDINALITY & LEAKAGE (Standard Checks)
# ==========================================
print("\n[4] NOISE & LEAKAGE CHECK")
flagged = ['ManagerID', 'State', 'RecruitmentSource', 'Position']
for col in flagged:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values. (To be dropped to reduce noise)")

leakage = ['TermReason', 'DateofTermination', 'Termd']
print(f"Leakage columns identified for removal: {leakage}")

# ==========================================
# 5. STRATEGIC DEPARTMENT AUDIT
# ==========================================
if 'Department' in df.columns:
    print("\n" + "="*50)
    print("      SECTION 5: DEPARTMENT DISTRIBUTION      ")
    print("="*50)
    dept_series = df['Department'].astype(str).str.strip()
    counts = dept_series.value_counts()
    percents = dept_series.value_counts(normalize=True) * 100
    
    dept_audit = pd.concat([counts, percents], axis=1, keys=['Count', 'Percentage'])
    print(dept_audit)
    
    small_groups = counts[counts < 10].index.tolist()
    if small_groups:
        print(f"\nWARNING: Small groups {small_groups} may need merging.")
    print("="*50)