#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
from scipy.stats import gaussian_kde
import glob
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
import sys
# Configuration
PHENOTYPE = sys.argv[1]  # e.g., "Phenotype_2"  
RESULTS_OUTPUT_DIR = "PRS_AUC_Analysis"
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# Define all methods to check
ALL_METHODS = ['Plink3', 'GCTA3', 'LDAK-GWAS3', 'PRSice-2-3', 'LDpred-gibbs3']

# Set publication-quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2

IMPORTANT_COLUMNS = [
    'clump_p1', 'clump_r2', 'clump_kb', 'p_window_size', 'p_slide_size',
    'p_LD_threshold', 'pvalue', 'referencepanel', 'PRSice-2_Model',
    'effectsizes', 'h2model', 'model', 'numberofpca', 'tempalpha', 'l1weight',
    'ldaksubmodel', 'ldakmodel', 'ldakpower', 'ldradius', 'ldfilename',
    'colname', 'gibsfraction', 'gibsburn', 'gibsiterations',
    'LDpred-funct-bins', 'heritability_model', 'unique_h2', 'grid_pvalue',
    'burn_in', 'num_iter', 'sparse', 'temp_pvalue', 'allow_jump_sign',
    'shrink_corr', 'use_MLE', 'lasso_parameters_count',
]

def find_common_rows(allfoldsframe):
    """Find common rows across all available folds."""
    if len(allfoldsframe) == 0:
        return []
    
    performance_columns = ['Train_pure_prs', 'Test_pure_prs']
    important_columns = [col for col in IMPORTANT_COLUMNS if col in allfoldsframe[0].columns]
    
    def drop_performance_columns(df):
        return df.drop(columns=performance_columns, errors='ignore')
    
    def get_important_columns(df):
        existing_columns = [col for col in important_columns if col in df.columns]
        return df[existing_columns].copy() if existing_columns else pd.DataFrame()
    
    allfoldsframe_dropped = [get_important_columns(drop_performance_columns(df)) 
                             for df in allfoldsframe]
    
    if any(df.empty for df in allfoldsframe_dropped):
        return []
    
    common_columns = set(allfoldsframe_dropped[0].columns)
    for df in allfoldsframe_dropped[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    common_columns = list(common_columns)
    if len(common_columns) == 0:
        return []
    
    allfoldsframe_dropped = [df[common_columns] for df in allfoldsframe_dropped]
    
    common_rows = allfoldsframe_dropped[0]
    for df in allfoldsframe_dropped[1:]:
        common_rows = pd.merge(common_rows, df, how='inner', on=common_columns)
    
    if common_rows.empty:
        return []
    
    extracted_common_rows_frames = []
    for original_df in allfoldsframe:
        extracted = pd.merge(common_rows, original_df, how='inner', on=common_columns)
        extracted_common_rows_frames.append(extracted)
    
    return extracted_common_rows_frames

def sum_and_average_columns(data_frames):
    """Average numerical columns across multiple DataFrames."""
    summed_df = pd.DataFrame()
    non_numerical_df = pd.DataFrame()
    
    for df in data_frames:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        non_numerical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if summed_df.empty:
            summed_df = pd.DataFrame(0, index=range(len(df)), columns=numerical_cols)
        
        summed_df[numerical_cols] = summed_df[numerical_cols].add(df[numerical_cols], fill_value=0)
        
        if non_numerical_df.empty:
            non_numerical_df = df[non_numerical_cols].copy()
    
    averaged_df = summed_df / len(data_frames)
    result_df = pd.concat([averaged_df, non_numerical_df], axis=1)
    
    return result_df

def check_prs_files_exist_all_folds(phenotype, method, folds, row):
    """Check if PRS files exist in ALL folds for a given row."""
    
    pvalue = row.get('pvalue', None)
    if pvalue is None:
        return False
    
    for fold_num in folds:
        method_dir = os.path.join(phenotype, f"Fold_{fold_num}", method)
        
        if not os.path.exists(method_dir):
            return False
        
        if 'PRSice' in method:
            # Check for all_score files
            train_files = glob.glob(os.path.join(method_dir, "*Train*.all_score"))
            test_files = glob.glob(os.path.join(method_dir, "*Test*.all_score"))
            val_files = glob.glob(os.path.join(method_dir, "*Val*.all_score"))
            
            if not (train_files and test_files and val_files):
                return False
        else:
            # Check for profile files with this pvalue
            pvalue_str = str(pvalue).replace('Pt_', '').replace('X', '')
            all_profiles = glob.glob(os.path.join(method_dir, "*.profile"))
            
            # Check train, test, val files exist
            has_train = False
            has_test = False
            has_val = False
            
            for profile in all_profiles:
                basename = os.path.basename(profile).lower()
                if pvalue_str in os.path.basename(profile):
                    if 'train' in basename:
                        has_train = True
                    elif 'test' in basename:
                        has_test = True
                    elif 'val' in basename:
                        has_val = True
            
            if not (has_train and has_test and has_val):
                return False
    
    return True

def find_best_method_and_parameters(phenotype, methods):
    """Find the best performing method with files existing in all folds."""
    
    print(f"\n{'='*80}")
    print("SCANNING ALL METHODS TO FIND BEST PERFORMER")
    print("="*80)
    
    all_candidates = []
    
    for method in methods:
        print(f"\nChecking {method}...")
        
        allfoldsframe = []
        available_folds = []
        
        for fold in range(5):
            file_path = os.path.join(phenotype, f"Fold_{fold}", method, "Results.csv")
            if os.path.exists(file_path):
                try:
                    temp = pd.read_csv(file_path)
                    allfoldsframe.append(temp)
                    available_folds.append(fold)
                except:
                    pass
        
        if len(allfoldsframe) == 0:
            print(f"  ✗ No data found")
            continue
        
        print(f"  ✓ Found {len(available_folds)} fold(s)")
        
        extracted_common_rows_list = find_common_rows(allfoldsframe)
        
        if len(extracted_common_rows_list) == 0:
            print(f"  ✗ No common rows")
            continue
        
        averaged_df = sum_and_average_columns(extracted_common_rows_list)
        
        if averaged_df.empty:
            print(f"  ✗ Empty results")
            continue
        
        # Determine sorting based on method
        use_lowest = 'LDpred-gibbs' in method
        averaged_df_sorted = averaged_df.sort_values(by='Test_pure_prs', ascending=use_lowest)
        
        print(f"  Checking which rows have files in all folds...")
        
        # Check each row (sorted by test performance) to find ones with files in all folds
        for idx, (_, row) in enumerate(averaged_df_sorted.iterrows()):
            if idx >= 10:  # Check top 10 rows max
                break
            
            if check_prs_files_exist_all_folds(phenotype, method, available_folds, row):
                test_auc = row['Test_pure_prs']
                print(f"    ✓ Rank {idx+1}: Test AUC = {test_auc:.6f} - Files exist in all folds")
                
                all_candidates.append({
                    'method': method,
                    'row': row,
                    'folds': available_folds,
                    'extracted': extracted_common_rows_list,
                    'test_auc': test_auc,
                    'rank': idx + 1
                })
                break  # Take the first valid one for this method
            else:
                print(f"    ✗ Rank {idx+1}: Files missing in some folds")
        
        if not any(c['method'] == method for c in all_candidates):
            print(f"  ✗ No valid rows with files in all folds")
    
    if len(all_candidates) == 0:
        print("\n❌ No valid methods found with files in all folds!")
        return None, None, None, None
    
    # Sort candidates by test AUC and pick the best
    all_candidates.sort(key=lambda x: x['test_auc'], reverse=True)
    best = all_candidates[0]
    
    print(f"\n{'='*80}")
    print(f"BEST METHOD WITH COMPLETE FILES: {best['method']}")
    print("="*80)
    print(f"Rank in method: {best['rank']}")
    print(f"Train AUC: {best['row']['Train_pure_prs']:.6f}")
    print(f"Test AUC:  {best['row']['Test_pure_prs']:.6f}")
    if 'pvalue' in best['row']:
        print(f"P-value:   {best['row']['pvalue']}")
    print(f"Available folds: {best['folds']}")
    print("="*80)
    
    # Show other candidates
    if len(all_candidates) > 1:
        print(f"\nOther candidates (with complete files):")
        for i, cand in enumerate(all_candidates[1:6], 2):  # Show top 5 others
            print(f"  {i}. {cand['method']} - Test AUC: {cand['test_auc']:.6f}")
    
    return best['row'], best['method'], best['folds'], best['extracted']

def read_phenotype_file(pheno_file):
    """Read phenotype file with proper handling."""
    if not os.path.exists(pheno_file):
        print(f"    ⚠️  File not found: {pheno_file}")
        return None
    
    print(f"    Reading: {pheno_file}")
    
    # Read FAM file (no header, space/tab separated)
    df = pd.read_csv(pheno_file, sep=r'\s+', header=None)
    
    print(f"    Raw data shape: {df.shape}")
    print(f"    First few rows:\n{df.head()}")
    
    # FAM file format: FID IID PID MID SEX PHENO
    if len(df.columns) >= 6:
        df.columns = ['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO']
        df = df[['FID', 'IID', 'PHENO']]
    elif len(df.columns) == 3:
        df.columns = ['FID', 'IID', 'PHENO']
    elif len(df.columns) == 2:
        df.columns = ['IID', 'PHENO']
        df['FID'] = df['IID']
    else:
        print(f"    ⚠️  Unexpected number of columns: {len(df.columns)}")
        return None
    
    # Convert PHENO to numeric and handle 1/2 coding
    df['PHENO'] = pd.to_numeric(df['PHENO'], errors='coerce')
    
    print(f"    PHENO values before conversion: {df['PHENO'].unique()}")
    
    # If coded as 1/2, convert to 0/1
    if df['PHENO'].min() == 1 and df['PHENO'].max() == 2:
        df['PHENO'] = df['PHENO'] - 1
        print(f"    Converted 1/2 coding to 0/1")
    
    print(f"    PHENO values after conversion: {df['PHENO'].unique()}")
    print(f"    Final shape: {df.shape}")
    
    return df[['FID', 'IID', 'PHENO']]

def find_closest_pvalue_column(df, target_pvalue):
    """Find the column that most closely matches the target p-value."""
    pvalue_cols = [col for col in df.columns if col not in ['FID', 'IID']]
    
    if not pvalue_cols:
        return None
    
    col_pvalues = []
    for col in pvalue_cols:
        try:
            # Try to extract numeric value from column name
            clean_col = col.replace('Pt_', '').replace('X', '')
            pval_float = float(clean_col)
            col_pvalues.append((col, pval_float))
        except:
            continue
    
    if not col_pvalues:
        return pvalue_cols[-1]  # Return last column as fallback
    
    # Clean target pvalue
    target_str = str(target_pvalue).replace('Pt_', '').replace('X', '')
    try:
        target = float(target_str)
    except:
        return pvalue_cols[-1]
    
    closest_col = min(col_pvalues, key=lambda x: abs(x[1] - target))
    
    return closest_col[0]

def read_prs_file_prsice(prs_file, pvalue):
    """Read PRSice PRS file and extract specific p-value column."""
    if not os.path.exists(prs_file):
        print(f"    ⚠️  File not found: {prs_file}")
        return None
    
    print(f"    Reading: {prs_file}")
    
    df = pd.read_csv(prs_file, sep=r'\s+')
    
    print(f"    File shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    
    # Clean pvalue string
    target_col = str(pvalue).replace('Pt_', '').replace('X', '')
    
    # Try exact match first
    matching_col = None
    for col in df.columns:
        clean_col = col.replace('Pt_', '').replace('X', '')
        if clean_col == target_col:
            matching_col = col
            break
    
    # If no exact match, find closest
    if matching_col is None:
        matching_col = find_closest_pvalue_column(df, pvalue)
    
    if matching_col is None:
        print(f"    ⚠️  Could not find p-value column for {pvalue}")
        print(f"    Available columns: {list(df.columns)}")
        return None
    
    # Return only FID, IID, and PRS
    result = df[['FID', 'IID']].copy()
    result['PRS'] = pd.to_numeric(df[matching_col], errors='coerce')
    
    print(f"    Using column: {matching_col}")
    print(f"    Result shape: {result.shape}")
    
    return result

def read_prs_file_plink_gcta_ldak(source_dir, pvalue, data_type):
    """Read Plink/GCTA/LDAK PRS file by finding matching profile."""
    pvalue_str = str(pvalue).replace('Pt_', '').replace('X', '')
    
    # Search for profile files
    all_profiles = glob.glob(os.path.join(source_dir, "*.profile"))
    
    print(f"    Found {len(all_profiles)} profile files")
    
    matching_files = []
    for profile in all_profiles:
        basename = os.path.basename(profile).lower()
        if pvalue_str in os.path.basename(profile) and data_type.lower() in basename:
            matching_files.append(profile)
    
    # Filter out Model_1 if multiple exist
    if len(matching_files) > 1:
        filtered = [f for f in matching_files if "Model_1" not in os.path.basename(f)]
        if filtered:
            matching_files = filtered
    
    if not matching_files:
        print(f"    ⚠️  No profile file found for pvalue={pvalue}, type={data_type}")
        print(f"    All profiles: {[os.path.basename(p) for p in all_profiles]}")
        return None
    
    prs_file = matching_files[0]
    print(f"    Using file: {os.path.basename(prs_file)}")
    
    df = pd.read_csv(prs_file, sep=r'\s+')
    
    print(f"    File shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    
    # Find the SCORE column
    score_col = None
    for col in df.columns:
        if 'SCORE' in col.upper():
            score_col = col
            break
    
    if score_col is None:
        print(f"    ⚠️  No SCORE column found")
        return None
    
    result = df[['FID', 'IID']].copy()
    result['PRS'] = pd.to_numeric(df[score_col], errors='coerce')
    
    print(f"    Result shape: {result.shape}")
    
    return result

def load_prs_and_pheno(phenotype, fold, method, best_row):
    """Load PRS scores and phenotypes for train, test, validation."""
    
    print(f"\nLoading data for Fold_{fold}...")
    
    base_dir = os.path.join(phenotype, f"Fold_{fold}")
    method_dir = os.path.join(base_dir, method)
    
    pvalue = best_row.get('pvalue', None)
    
    data = {}
    
    # Load train data
    print("  Loading train data...")
    if 'PRSice' in method:
        train_prs_file = glob.glob(os.path.join(method_dir, "*Train*.all_score"))
        if train_prs_file:
            print(f"    Found train PRS file: {os.path.basename(train_prs_file[0])}")
            data['train_prs'] = read_prs_file_prsice(train_prs_file[0], pvalue)
        else:
            data['train_prs'] = None
            print("    ✗ No train PRS file found")
    else:
        data['train_prs'] = read_prs_file_plink_gcta_ldak(method_dir, pvalue, "train")
    
    train_pheno_path = os.path.join(base_dir, "train_data.fam")
    print(f"    Looking for FAM file: {train_pheno_path}")
    train_pheno = read_phenotype_file(train_pheno_path)
    data['train_pheno'] = train_pheno
    
    if data['train_prs'] is not None and train_pheno is not None:
        print(f"    ✓ Train PRS: {len(data['train_prs'])} samples")
        print(f"    ✓ Train Pheno: {len(train_pheno)} samples")
        print(f"    PRS columns: {list(data['train_prs'].columns)}")
        print(f"    Pheno columns: {list(train_pheno.columns)}")
    else:
        print(f"    ✗ Train PRS is None: {data['train_prs'] is None}")
        print(f"    ✗ Train Pheno is None: {train_pheno is None}")
    
    # Load test data
    print("  Loading test data...")
    if 'PRSice' in method:
        test_prs_file = glob.glob(os.path.join(method_dir, "*Test*.all_score"))
        if test_prs_file:
            print(f"    Found test PRS file: {os.path.basename(test_prs_file[0])}")
            data['test_prs'] = read_prs_file_prsice(test_prs_file[0], pvalue)
        else:
            data['test_prs'] = None
            print("    ✗ No test PRS file found")
    else:
        data['test_prs'] = read_prs_file_plink_gcta_ldak(method_dir, pvalue, "test")
    
    test_pheno_path = os.path.join(base_dir, "test_data.fam")
    print(f"    Looking for FAM file: {test_pheno_path}")
    test_pheno = read_phenotype_file(test_pheno_path)
    data['test_pheno'] = test_pheno
    
    if data['test_prs'] is not None and test_pheno is not None:
        print(f"    ✓ Test PRS: {len(data['test_prs'])} samples")
        print(f"    ✓ Test Pheno: {len(test_pheno)} samples")
        print(f"    PRS columns: {list(data['test_prs'].columns)}")
        print(f"    Pheno columns: {list(test_pheno.columns)}")
    else:
        print(f"    ✗ Test PRS is None: {data['test_prs'] is None}")
        print(f"    ✗ Test Pheno is None: {test_pheno is None}")
    
    # Load validation data
    print("  Loading validation data...")
    if 'PRSice' in method:
        val_prs_file = glob.glob(os.path.join(method_dir, "*Val*.all_score"))
        if val_prs_file:
            print(f"    Found validation PRS file: {os.path.basename(val_prs_file[0])}")
            data['val_prs'] = read_prs_file_prsice(val_prs_file[0], pvalue)
        else:
            data['val_prs'] = None
            print("    ✗ No validation PRS file found")
    else:
        data['val_prs'] = read_prs_file_plink_gcta_ldak(method_dir, pvalue, "val")
    
    if data['val_prs'] is not None:
        print(f"    ✓ Validation PRS: {len(data['val_prs'])} samples")
        print(f"    PRS columns: {list(data['val_prs'].columns)}")
    
    return data

def calculate_auc_metrics(prs_df, pheno_df):
    """Calculate AUC and related metrics."""
    if prs_df is None or pheno_df is None:
        print("    ⚠️  Missing PRS or phenotype data")
        print(f"       PRS is None: {prs_df is None}")
        print(f"       Pheno is None: {pheno_df is None}")
        return None
    
    print(f"    Input PRS shape: {prs_df.shape}")
    print(f"    Input Pheno shape: {pheno_df.shape}")
    print(f"    PRS columns: {list(prs_df.columns)}")
    print(f"    Pheno columns: {list(pheno_df.columns)}")
    
    # Ensure column names are strings
    prs_df = prs_df.copy()
    pheno_df = pheno_df.copy()
    prs_df.columns = prs_df.columns.astype(str)
    pheno_df.columns = pheno_df.columns.astype(str)
    
    # Convert IID to string for merging
    prs_df['IID'] = prs_df['IID'].astype(str)
    pheno_df['IID'] = pheno_df['IID'].astype(str)
    
    print(f"    PRS sample IDs (first 5): {list(prs_df['IID'].head())}")
    print(f"    Pheno sample IDs (first 5): {list(pheno_df['IID'].head())}")
    
    # Merge PRS with phenotypes
    merged = pd.merge(prs_df, pheno_df, on='IID', how='inner', suffixes=('', '_pheno'))
    
    if len(merged) == 0:
        print("    ⚠️  No samples after merging PRS and phenotype")
        print(f"    PRS unique IDs: {len(prs_df['IID'].unique())}")
        print(f"    Pheno unique IDs: {len(pheno_df['IID'].unique())}")
        # Check for common IDs
        common_ids = set(prs_df['IID'].values) & set(pheno_df['IID'].values)
        print(f"    Common IDs: {len(common_ids)}")
        if len(common_ids) > 0:
            print(f"    Example common IDs: {list(common_ids)[:5]}")
        return None
    
    print(f"    Merged: {len(merged)} samples")
    print(f"    Merged columns: {list(merged.columns)}")
    
    y_true = merged['PHENO'].values
    y_scores = merged['PRS'].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(y_scores) & ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    y_scores = y_scores[valid_mask]
    
    if len(y_true) == 0:
        print("    ⚠️  No valid data after removing NaNs")
        return None
    
    if len(np.unique(y_true)) < 2:
        print(f"    ⚠️  Only one class found: {np.unique(y_true)}")
        return None
    
    # Calculate AUC
    auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    cases = y_scores[y_true == 1]
    controls = y_scores[y_true == 0]
    
    print(f"    ✓ AUC: {auc_score:.6f} (Cases: {len(cases)}, Controls: {len(controls)})")
    
    return {
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'cases': cases,
        'controls': controls,
        'y_true': y_true,
        'y_scores': y_scores,
        'n_cases': len(cases),
        'n_controls': len(controls)
    }

def estimate_validation_auc_range(val_prs_scores, n_cases=2500, n_total=50000,
                                   train_auc=None, test_auc=None, n_simulations=10000):
    """Estimate possible AUC range for validation set."""
    
    print("\nEstimating validation AUC range...")
    print(f"  Validation samples: {n_total}")
    print(f"  Expected cases: {n_cases}")
    print(f"  Running {n_simulations} Monte Carlo simulations...")
    
    n_controls = n_total - n_cases
    
    # Remove NaN values
    val_prs_scores = val_prs_scores[~np.isnan(val_prs_scores)]
    
    if len(val_prs_scores) != n_total:
        print(f"  ⚠️  Warning: Expected {n_total} samples, got {len(val_prs_scores)}")
        n_total = len(val_prs_scores)
        n_cases = min(n_cases, n_total)
        n_controls = n_total - n_cases
    
    # Rank PRS scores (high to low)
    sorted_indices = np.argsort(val_prs_scores)[::-1]
    
    # Best case: top scores are all cases
    best_case_labels = np.zeros(n_total)
    best_case_labels[sorted_indices[:n_cases]] = 1
    best_case_auc = roc_auc_score(best_case_labels, val_prs_scores)
    
    # Worst case: bottom scores are cases
    worst_case_labels = np.zeros(n_total)
    worst_case_labels[sorted_indices[-n_cases:]] = 1
    worst_case_auc = roc_auc_score(worst_case_labels, val_prs_scores)
    
    # Random case: Monte Carlo simulation
    print("  Running simulations...")
    random_aucs = []
    for i in range(n_simulations):
        if i % 2000 == 0 and i > 0:
            print(f"    Progress: {i}/{n_simulations}")
        
        random_labels = np.zeros(n_total)
        random_case_indices = np.random.choice(n_total, n_cases, replace=False)
        random_labels[random_case_indices] = 1
        
        try:
            random_auc = roc_auc_score(random_labels, val_prs_scores)
            random_aucs.append(random_auc)
        except:
            continue
    
    random_auc_mean = np.mean(random_aucs)
    random_auc_std = np.std(random_aucs)
    
    # Expected AUC based on test performance
    if test_auc is not None:
        expected_auc = test_auc * 0.95
        expected_range = (test_auc * 0.85, min(test_auc * 1.0, 1.0))
    else:
        expected_auc = random_auc_mean
        expected_range = (random_auc_mean - 2*random_auc_std,
                         random_auc_mean + 2*random_auc_std)
    
    print("  ✓ Simulations complete")
    
    return {
        'best_case_auc': best_case_auc,
        'worst_case_auc': worst_case_auc,
        'random_auc_mean': random_auc_mean,
        'random_auc_std': random_auc_std,
        'random_aucs': random_aucs,
        'expected_auc': expected_auc,
        'expected_range': expected_range,
        'theoretical_range': (worst_case_auc, best_case_auc)
    }

def plot_roc_curves(train_metrics, test_metrics, phenotype, method, fold):
    """Plot ROC curves for train and test sets."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(train_metrics['fpr'], train_metrics['tpr'],
            label=f'Train (AUC = {train_metrics["auc"]:.4f})',
            color='#1f77b4', linewidth=2.5, alpha=0.8)
    
    ax.plot(test_metrics['fpr'], test_metrics['tpr'],
            label=f'Test (AUC = {test_metrics["auc"]:.4f})',
            color='#ff7f0e', linewidth=2.5, alpha=0.8)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
            label='Random (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    ax.set_title(f'ROC Curves - {phenotype} (Fold {fold})\nMethod: {method}',
                fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_ROC_Curves.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_ROC_Curves.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Generated ROC curves plot")

def plot_prs_distributions(train_metrics, test_metrics, val_prs, phenotype, method, fold):
    """Plot PRS score distributions."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    val_scores = val_prs['PRS'].values
    val_scores = val_scores[~np.isnan(val_scores)]
    
    case_color = '#d62728'
    control_color = '#2ca02c'
    val_color = '#9467bd'
    
    # 1. Training distribution
    ax1 = axes[0]
    ax1.hist(train_metrics['cases'], bins=50, alpha=0.6, color=case_color,
            label=f'Cases (n={train_metrics["n_cases"]})',
            density=True, edgecolor='black', linewidth=0.5)
    ax1.hist(train_metrics['controls'], bins=50, alpha=0.6, color=control_color,
            label=f'Controls (n={train_metrics["n_controls"]})',
            density=True, edgecolor='black', linewidth=0.5)
    
    try:
        kde_cases = gaussian_kde(train_metrics['cases'])
        kde_controls = gaussian_kde(train_metrics['controls'])
        x_range = np.linspace(min(train_metrics['y_scores']),
                             max(train_metrics['y_scores']), 200)
        ax1.plot(x_range, kde_cases(x_range), color=case_color,
                linewidth=2.5, linestyle='--', alpha=0.8)
        ax1.plot(x_range, kde_controls(x_range), color=control_color,
                linewidth=2.5, linestyle='--', alpha=0.8)
    except:
        pass
    
    ax1.set_xlabel('PRS Score', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax1.set_title(f'Training Set PRS Distribution (AUC = {train_metrics["auc"]:.4f})',
                 fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Test distribution
    ax2 = axes[1]
    ax2.hist(test_metrics['cases'], bins=50, alpha=0.6, color=case_color,
            label=f'Cases (n={test_metrics["n_cases"]})',
            density=True, edgecolor='black', linewidth=0.5)
    ax2.hist(test_metrics['controls'], bins=50, alpha=0.6, color=control_color,
            label=f'Controls (n={test_metrics["n_controls"]})',
            density=True, edgecolor='black', linewidth=0.5)
    
    try:
        kde_cases = gaussian_kde(test_metrics['cases'])
        kde_controls = gaussian_kde(test_metrics['controls'])
        x_range = np.linspace(min(test_metrics['y_scores']),
                             max(test_metrics['y_scores']), 200)
        ax2.plot(x_range, kde_cases(x_range), color=case_color,
                linewidth=2.5, linestyle='--', alpha=0.8)
        ax2.plot(x_range, kde_controls(x_range), color=control_color,
                linewidth=2.5, linestyle='--', alpha=0.8)
    except:
        pass
    
    ax2.set_xlabel('PRS Score', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax2.set_title(f'Test Set PRS Distribution (AUC = {test_metrics["auc"]:.4f})',
                 fontweight='bold', fontsize=13)
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Validation distribution
    ax3 = axes[2]
    ax3.hist(val_scores, bins=60, alpha=0.7, color=val_color,
            label=f'All Samples (n={len(val_scores)})\n2500 unknown cases',
            density=True, edgecolor='black', linewidth=0.5)
    
    try:
        kde_val = gaussian_kde(val_scores)
        x_range = np.linspace(min(val_scores), max(val_scores), 200)
        ax3.plot(x_range, kde_val(x_range), color=val_color,
                linewidth=2.5, linestyle='--', alpha=0.8)
    except:
        pass
    
    ax3.set_xlabel('PRS Score', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax3.set_title('Validation Set PRS Distribution (Labels Unknown)',
                 fontweight='bold', fontsize=13)
    ax3.legend(fontsize=11, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'PRS Score Distributions - {phenotype} (Fold {fold})',
                fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_PRS_Distributions.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_PRS_Distributions.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Generated PRS distributions plot")

def plot_auc_points_distribution(train_metrics, test_metrics, phenotype, fold):
    """Plot AUC distribution with decision boundaries."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Find optimal threshold (Youden's index)
    train_j_scores = train_metrics['tpr'] - train_metrics['fpr']
    train_optimal_idx = np.argmax(train_j_scores)
    train_optimal_threshold = train_metrics['thresholds'][train_optimal_idx]
    
    test_j_scores = test_metrics['tpr'] - test_metrics['fpr']
    test_optimal_idx = np.argmax(test_j_scores)
    test_optimal_threshold = test_metrics['thresholds'][test_optimal_idx]
    
    # Plot 1: Distributions with thresholds
    ax1.hist(train_metrics['controls'], bins=40, alpha=0.4, color='#2ca02c',
            label='Train Controls', density=True)
    ax1.hist(train_metrics['cases'], bins=40, alpha=0.4, color='#d62728',
            label='Train Cases', density=True)
    
    ax1.hist(test_metrics['controls'], bins=40, alpha=0.3, color='#1f77b4',
            label='Test Controls', density=True, histtype='step', linewidth=2)
    ax1.hist(test_metrics['cases'], bins=40, alpha=0.3, color='#ff7f0e',
            label='Test Cases', density=True, histtype='step', linewidth=2)
    
    ax1.axvline(train_optimal_threshold, color='green', linestyle='--',
               linewidth=2.5, label=f'Train Threshold: {train_optimal_threshold:.3f}')
    ax1.axvline(test_optimal_threshold, color='blue', linestyle='--',
               linewidth=2.5, label=f'Test Threshold: {test_optimal_threshold:.3f}')
    
    ax1.set_xlabel('PRS Score', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=13)
    ax1.set_title('PRS Distribution with Decision Boundaries',
                 fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left', fontsize=9, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: ROC space with operating points
    ax2.plot(train_metrics['fpr'], train_metrics['tpr'],
            label=f'Train ROC (AUC={train_metrics["auc"]:.4f})',
            color='#1f77b4', linewidth=2.5, alpha=0.8)
    ax2.plot(test_metrics['fpr'], test_metrics['tpr'],
            label=f'Test ROC (AUC={test_metrics["auc"]:.4f})',
            color='#ff7f0e', linewidth=2.5, alpha=0.8)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    train_opt_fpr = train_metrics['fpr'][train_optimal_idx]
    train_opt_tpr = train_metrics['tpr'][train_optimal_idx]
    test_opt_fpr = test_metrics['fpr'][test_optimal_idx]
    test_opt_tpr = test_metrics['tpr'][test_optimal_idx]
    
    ax2.scatter(train_opt_fpr, train_opt_tpr, s=200, color='#1f77b4',
               marker='o', edgecolor='black', linewidth=2, zorder=5,
               label=f'Train Optimal (FPR={train_opt_fpr:.3f}, TPR={train_opt_tpr:.3f})')
    ax2.scatter(test_opt_fpr, test_opt_tpr, s=200, color='#ff7f0e',
               marker='s', edgecolor='black', linewidth=2, zorder=5,
               label=f'Test Optimal (FPR={test_opt_fpr:.3f}, TPR={test_opt_tpr:.3f})')
    
    ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    ax2.set_title('ROC Space with Optimal Operating Points',
                 fontweight='bold', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    
    plt.suptitle(f'AUC Points Distribution - {phenotype} (Fold {fold})',
                fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_AUC_Points_Distribution.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_AUC_Points_Distribution.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Generated AUC points distribution plot")

def estimate_validation_performance_by_threshold(train_metrics, test_metrics, val_prs, n_cases=2500, n_total=50000):
    """Estimate validation AUC based on decision thresholds from train/test."""
    
    print("\n" + "="*80)
    print("VALIDATION PERFORMANCE ESTIMATION BY THRESHOLD ANALYSIS")
    print("="*80)
    
    val_scores = val_prs['PRS'].values
    val_scores = val_scores[~np.isnan(val_scores)]
    
    n_controls = n_total - n_cases
    
    # Find optimal thresholds using Youden's index
    train_j_scores = train_metrics['tpr'] - train_metrics['fpr']
    train_optimal_idx = np.argmax(train_j_scores)
    train_threshold = train_metrics['thresholds'][train_optimal_idx]
    train_tpr = train_metrics['tpr'][train_optimal_idx]
    train_fpr = train_metrics['fpr'][train_optimal_idx]
    
    test_j_scores = test_metrics['tpr'] - test_metrics['fpr']
    test_optimal_idx = np.argmax(test_j_scores)
    test_threshold = test_metrics['thresholds'][test_optimal_idx]
    test_tpr = test_metrics['tpr'][test_optimal_idx]
    test_fpr = test_metrics['fpr'][test_optimal_idx]
    
    print(f"\nTrain Optimal Threshold: {train_threshold:.6f}")
    print(f"  Sensitivity (TPR): {train_tpr:.4f}")
    print(f"  1-Specificity (FPR): {train_fpr:.4f}")
    print(f"  Specificity: {1-train_fpr:.4f}")
    
    print(f"\nTest Optimal Threshold: {test_threshold:.6f}")
    print(f"  Sensitivity (TPR): {test_tpr:.4f}")
    print(f"  1-Specificity (FPR): {test_fpr:.4f}")
    print(f"  Specificity: {1-test_fpr:.4f}")
    
    # Use average threshold
    avg_threshold = (train_threshold + test_threshold) / 2
    print(f"\nAverage Threshold: {avg_threshold:.6f}")
    
    # Count validation samples above/below thresholds
    results = {}
    
    for threshold_name, threshold, tpr, fpr in [
        ('Train', train_threshold, train_tpr, train_fpr),
        ('Test', test_threshold, test_tpr, test_fpr),
        ('Average', avg_threshold, (train_tpr+test_tpr)/2, (train_fpr+test_fpr)/2)
    ]:
        n_above = np.sum(val_scores >= threshold)
        n_below = np.sum(val_scores < threshold)
        percent_above = (n_above / n_total) * 100
        
        print(f"\n{'-'*80}")
        print(f"Using {threshold_name} Threshold ({threshold:.6f}):")
        print(f"{'-'*80}")
        print(f"  Validation samples above threshold: {n_above:,} ({percent_above:.2f}%)")
        print(f"  Validation samples below threshold: {n_below:,} ({100-percent_above:.2f}%)")
        
        # Scenario 1: Optimistic - Assume threshold perfectly separates cases/controls
        # Cases should be above threshold, controls below
        expected_cases_above = int(n_cases * tpr)
        expected_controls_above = int(n_controls * fpr)
        expected_cases_below = n_cases - expected_cases_above
        expected_controls_below = n_controls - expected_controls_above
        
        print(f"\n  Expected distribution (based on {threshold_name} performance):")
        print(f"    True Positives (cases above):   {expected_cases_above:,} / {n_cases:,} ({tpr*100:.2f}%)")
        print(f"    False Positives (controls above): {expected_controls_above:,} / {n_controls:,} ({fpr*100:.2f}%)")
        print(f"    True Negatives (controls below):  {expected_controls_below:,} / {n_controls:,} ({(1-fpr)*100:.2f}%)")
        print(f"    False Negatives (cases below):    {expected_cases_below:,} / {n_cases:,} ({(1-tpr)*100:.2f}%)")
        
        # Calculate multiple scenarios
        print(f"\n  Possible Scenarios:")
        
        # Scenario A: Best case - samples above threshold are enriched for cases
        if n_above <= n_cases:
            # All above are cases
            scenario_a_tp = n_above
            scenario_a_fn = n_cases - n_above
            scenario_a_fp = 0
            scenario_a_tn = n_controls
        else:
            # All cases are above, plus some controls
            scenario_a_tp = n_cases
            scenario_a_fn = 0
            scenario_a_fp = n_above - n_cases
            scenario_a_tn = n_controls - scenario_a_fp
        
        scenario_a_tpr = scenario_a_tp / n_cases
        scenario_a_fpr = scenario_a_fp / n_controls
        # Approximate AUC using TPR and FPR
        scenario_a_auc = 0.5 + (scenario_a_tpr - scenario_a_fpr) / 2
        
        print(f"    A) Best Case (cases enriched above threshold):")
        print(f"       TP={scenario_a_tp:,}, FN={scenario_a_fn:,}, FP={scenario_a_fp:,}, TN={scenario_a_tn:,}")
        print(f"       Sensitivity: {scenario_a_tpr:.4f}, Specificity: {1-scenario_a_fpr:.4f}")
        print(f"       Estimated AUC: ~{scenario_a_auc:.4f}")
        
        # Scenario B: Expected case - use train/test performance
        scenario_b_tp = min(expected_cases_above, n_above)
        scenario_b_fp = n_above - scenario_b_tp
        scenario_b_fn = n_cases - scenario_b_tp
        scenario_b_tn = n_controls - scenario_b_fp
        
        scenario_b_tpr = scenario_b_tp / n_cases
        scenario_b_fpr = scenario_b_fp / n_controls
        scenario_b_auc = 0.5 + (scenario_b_tpr - scenario_b_fpr) / 2
        
        print(f"    B) Expected Case (based on {threshold_name} TPR/FPR):")
        print(f"       TP={scenario_b_tp:,}, FN={scenario_b_fn:,}, FP={scenario_b_fp:,}, TN={scenario_b_tn:,}")
        print(f"       Sensitivity: {scenario_b_tpr:.4f}, Specificity: {1-scenario_b_fpr:.4f}")
        print(f"       Estimated AUC: ~{scenario_b_auc:.4f}")
        
        # Scenario C: Random distribution
        expected_cases_by_random = int((n_above / n_total) * n_cases)
        scenario_c_tp = expected_cases_by_random
        scenario_c_fp = n_above - scenario_c_tp
        scenario_c_fn = n_cases - scenario_c_tp
        scenario_c_tn = n_controls - scenario_c_fp
        
        scenario_c_tpr = scenario_c_tp / n_cases
        scenario_c_fpr = scenario_c_fp / n_controls
        scenario_c_auc = 0.5 + (scenario_c_tpr - scenario_c_fpr) / 2
        
        print(f"    C) Random Distribution (null model):")
        print(f"       TP={scenario_c_tp:,}, FN={scenario_c_fn:,}, FP={scenario_c_fp:,}, TN={scenario_c_tn:,}")
        print(f"       Sensitivity: {scenario_c_tpr:.4f}, Specificity: {1-scenario_c_fpr:.4f}")
        print(f"       Estimated AUC: ~{scenario_c_auc:.4f}")
        
        results[threshold_name] = {
            'threshold': threshold,
            'n_above': n_above,
            'n_below': n_below,
            'best_case_auc': scenario_a_auc,
            'expected_auc': scenario_b_auc,
            'random_auc': scenario_c_auc,
            'train_test_tpr': tpr,
            'train_test_fpr': fpr
        }
    
    print("\n" + "="*80)
    print("SUMMARY OF THRESHOLD-BASED PREDICTIONS")
    print("="*80)
    
    print(f"\n{'Threshold':<12} {'Samples':<15} {'Best AUC':<12} {'Expected AUC':<15} {'Random AUC':<12}")
    print(f"{' '*12} {'Above/Below':<15} {'':<12} {'':<15} {'':<12}")
    print("-"*80)
    for name, res in results.items():
        print(f"{name:<12} {res['n_above']:>6,} / {res['n_below']:<6,}  "
              f"{res['best_case_auc']:>10.4f}  {res['expected_auc']:>13.4f}  {res['random_auc']:>10.4f}")
    
    print("="*80)
    
    return results

def calculate_confusion_matrix_metrics(tp, fp, fn, tn):
    """Calculate comprehensive metrics from confusion matrix."""
    total = tp + fp + fn + tn
    
    # Basic metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    accuracy = (tp + tn) / total if total > 0 else 0
    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) \
          if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'mcc': mcc
    }

def print_confusion_matrix_box(name, tp, fp, fn, tn, show_metrics=True):
    """Print confusion matrix in a simple box format."""
    
    print(f"\n{'='*70}")
    print(f"CONFUSION MATRIX - {name}")
    print("="*70)
    
    # Print the box
    total = tp + fp + fn + tn
    print("\n┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│                     │   Predicted  │   Predicted  │              │")
    print("│                     │   Positive   │   Negative   │    Total     │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Actual Positive     │   {tp:>8,}   │   {fn:>8,}   │   {tp+fn:>8,}   │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Actual Negative     │   {fp:>8,}   │   {tn:>8,}   │   {fp+tn:>8,}   │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Total               │   {tp+fp:>8,}   │   {fn+tn:>8,}   │   {total:>8,}   │")
    print("└─────────────────────┴──────────────┴──────────────┴──────────────┘")
    
    if show_metrics:
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        
        print(f"\nMetrics:")
        print(f"  Sensitivity (TPR):  {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"  Specificity (TNR):  {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"  Precision (PPV):    {ppv:.4f} ({ppv*100:.2f}%)")
        print(f"  NPV:                {npv:.4f} ({npv*100:.2f}%)")
        print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("="*70)

def print_confusion_matrix_detailed(name, tp, fp, fn, tn, n_cases=None, n_controls=None):
    """Print detailed confusion matrix with all metrics."""
    
    # Use simple box format
    print_confusion_matrix_box(name, tp, fp, fn, tn, show_metrics=True)
    
    # Calculate metrics for return
    metrics = calculate_confusion_matrix_metrics(tp, fp, fn, tn)
    return metrics

def print_train_test_confusion_matrices(train_metrics, test_metrics):
    """Print confusion matrices for train and test sets."""
    
    print("\n" + "="*80)
    print("CONFUSION MATRICES FOR TRAIN AND TEST SETS")
    print("="*80)
    
    # Find optimal thresholds
    train_j_scores = train_metrics['tpr'] - train_metrics['fpr']
    train_optimal_idx = np.argmax(train_j_scores)
    train_threshold = train_metrics['thresholds'][train_optimal_idx]
    
    test_j_scores = test_metrics['tpr'] - test_metrics['fpr']
    test_optimal_idx = np.argmax(test_j_scores)
    test_threshold = test_metrics['thresholds'][test_optimal_idx]
    
    # Calculate confusion matrices
    # Train
    train_pred = (train_metrics['y_scores'] >= train_threshold).astype(int)
    train_tp = np.sum((train_metrics['y_true'] == 1) & (train_pred == 1))
    train_fp = np.sum((train_metrics['y_true'] == 0) & (train_pred == 1))
    train_fn = np.sum((train_metrics['y_true'] == 1) & (train_pred == 0))
    train_tn = np.sum((train_metrics['y_true'] == 0) & (train_pred == 0))
    
    # Test
    test_pred = (test_metrics['y_scores'] >= test_threshold).astype(int)
    test_tp = np.sum((test_metrics['y_true'] == 1) & (test_pred == 1))
    test_fp = np.sum((test_metrics['y_true'] == 0) & (test_pred == 1))
    test_fn = np.sum((test_metrics['y_true'] == 1) & (test_pred == 0))
    test_tn = np.sum((test_metrics['y_true'] == 0) & (test_pred == 0))
    
    # Print matrices
    train_metrics_dict = print_confusion_matrix_box(
        f"TRAINING SET (Threshold: {train_threshold:.6f})",
        train_tp, train_fp, train_fn, train_tn
    )
    
    test_metrics_dict = print_confusion_matrix_box(
        f"TEST SET (Threshold: {test_threshold:.6f})",
        test_tp, test_fp, test_fn, test_tn
    )
    
    return {
        'train': {'confusion': (train_tp, train_fp, train_fn, train_tn), 
                  'metrics': calculate_confusion_matrix_metrics(train_tp, train_fp, train_fn, train_tn), 
                  'threshold': train_threshold},
        'test': {'confusion': (test_tp, test_fp, test_fn, test_tn), 
                 'metrics': calculate_confusion_matrix_metrics(test_tp, test_fp, test_fn, test_tn), 
                 'threshold': test_threshold}
    }

def print_validation_estimated_confusion_matrices(threshold_results, n_cases=2500, n_total=50000):
    """Print estimated confusion matrices for validation set based on different scenarios."""
    
    print("\n" + "="*80)
    print("ESTIMATED CONFUSION MATRICES FOR VALIDATION SET")
    print("="*80)
    print("\nNote: Validation labels are unknown. These are estimates based on threshold analysis.")
    print("="*80)
    
    n_controls = n_total - n_cases
    
    for threshold_name, results in threshold_results.items():
        n_above = results['n_above']
        n_below = results['n_below']
        threshold = results['threshold']
        tpr = results['train_test_tpr']
        fpr = results['train_test_fpr']
        
        print(f"\n{'#'*80}")
        print(f"THRESHOLD: {threshold_name} ({threshold:.6f})")
        print(f"Samples above threshold: {n_above:,} | Samples below threshold: {n_below:,}")
        print("#"*80)
        
        # Scenario A: Best Case
        print(f"\n--- SCENARIO A: Best Case (Maximum Performance) ---")
        if n_above <= n_cases:
            a_tp, a_fn = n_above, n_cases - n_above
            a_fp, a_tn = 0, n_controls
        else:
            a_tp, a_fn = n_cases, 0
            a_fp, a_tn = n_above - n_cases, n_controls - (n_above - n_cases)
        
        print_confusion_matrix_box(f"{threshold_name} - Best Case", a_tp, a_fp, a_fn, a_tn)
        
        # Scenario B: Expected Case
        print(f"\n--- SCENARIO B: Expected Case (Based on Train/Test Performance) ---")
        expected_cases_above = int(n_cases * tpr)
        
        b_tp = min(expected_cases_above, n_above)
        b_fp = n_above - b_tp
        b_fn = n_cases - b_tp
        b_tn = n_controls - b_fp
        
        print_confusion_matrix_box(f"{threshold_name} - Expected Case", b_tp, b_fp, b_fn, b_tn)
        
        # Scenario C: Random Distribution
        print(f"\n--- SCENARIO C: Random Distribution (Null Model) ---")
        expected_cases_by_random = int((n_above / n_total) * n_cases)
        scenario_c_tp = expected_cases_by_random
        scenario_c_fp = n_above - scenario_c_tp
        scenario_c_fn = n_cases - scenario_c_tp
        scenario_c_tn = n_controls - scenario_c_fp
        
        print_confusion_matrix_box(f"{threshold_name} - Random Distribution", scenario_c_tp, scenario_c_fp, scenario_c_fn, scenario_c_tn)

def print_validation_sample_distribution_detailed(threshold_results, val_prs, n_cases=2500, n_total=50000):
    """Print detailed validation sample distribution analysis."""
    
    print("\n" + "="*80)
    print("DETAILED VALIDATION SAMPLE DISTRIBUTION ANALYSIS")
    print("="*80)
    print("\nNote: Cases are expected to have HIGHER PRS scores.")
    print("="*80)
    
    val_scores = val_prs['PRS'].values
    val_scores = val_scores[~np.isnan(val_scores)]
    
    # Sort scores and get rankings
    sorted_indices = np.argsort(val_scores)[::-1]  # High to low
    sorted_scores = val_scores[sorted_indices]
    
    n_controls = n_total - n_cases
    
    for threshold_name, results in threshold_results.items():
        threshold = results['threshold']
        n_above = results['n_above']
        n_below = results['n_below']
        tpr = results['train_test_tpr']
        fpr = results['train_test_fpr']
        
        print(f"\n{'#'*80}")
        print(f"THRESHOLD: {threshold_name} ({threshold:.6f})")
        print(f"Samples above threshold: {n_above:,} | Samples below threshold: {n_below:,}")
        print("#"*80)
        
        # Calculate scores above and below threshold
        above_threshold_indices = val_scores >= threshold
        above_scores = val_scores[above_threshold_indices]
        
        below_threshold_indices = val_scores < threshold
        below_scores = val_scores[below_threshold_indices]
        
        # Scenario A: Best Case
        print(f"\n--- SCENARIO A: Best Case (Maximum Performance) ---")
        if n_above <= n_cases:
            a_tp, a_fn = n_above, n_cases - n_above
            a_fp, a_tn = 0, n_controls
        else:
            a_tp, a_fn = n_cases, 0
            a_fp, a_tn = n_above - n_cases, n_controls - (n_above - n_cases)
        
        print_confusion_matrix_box(f"{threshold_name} - Best Case", a_tp, a_fp, a_fn, a_tn)
        
        # Scenario B: Expected Case
        print(f"\n--- SCENARIO B: Expected Case (Based on Train/Test Performance) ---")
        expected_cases_above = int(n_cases * tpr)
        
        b_tp = min(expected_cases_above, n_above)
        b_fp = n_above - b_tp
        b_fn = n_cases - b_tp
        b_tn = n_controls - b_fp
        
        print_confusion_matrix_box(f"{threshold_name} - Expected Case", b_tp, b_fp, b_fn, b_tn)
        
        # Scenario C: Random Distribution
        print(f"\n--- SCENARIO C: Random Distribution (Null Model) ---")
        expected_cases_by_random = int((n_above / n_total) * n_cases)
        scenario_c_tp = expected_cases_by_random
        scenario_c_fp = n_above - scenario_c_tp
        scenario_c_fn = n_cases - scenario_c_tp
        scenario_c_tn = n_controls - scenario_c_fp
        
        print_confusion_matrix_box(f"{threshold_name} - Random Distribution", scenario_c_tp, scenario_c_fp, scenario_c_fn, scenario_c_tn)
        
        # PRS statistics for above/below groups
        print(f"\n📊 PRS SCORE STATISTICS:")
        print(f"\n  Samples ABOVE threshold (n={n_above:,}):")
        if len(above_scores) > 0:
            print(f"    Mean PRS:   {np.mean(above_scores):.6f}")
            print(f"    Median PRS: {np.median(above_scores):.6f}")
            print(f"    Std PRS:    {np.std(above_scores):.6f}")
            print(f"    Min PRS:    {np.min(above_scores):.6f}")
            print(f"    Max PRS:    {np.max(above_scores):.6f}")
        else:
            print(f"    No samples above threshold")
        
        print(f"\n  Samples BELOW threshold (n={n_below:,}):")
        if len(below_scores) > 0:
            print(f"    Mean PRS:   {np.mean(below_scores):.6f}")
            print(f"    Median PRS: {np.median(below_scores):.6f}")
            print(f"    Std PRS:    {np.std(below_scores):.6f}")
            print(f"    Min PRS:    {np.min(below_scores):.6f}")
            print(f"    Max PRS:    {np.max(below_scores):.6f}")
        else:
            print(f"    No samples below threshold")
        
        # Risk stratification
        print(f"\n🎯 RISK STRATIFICATION OF VALIDATION SET:")
        
        # Top samples (highest risk)
        top_percentiles = [1, 5, 10, 25]
        for pct in top_percentiles:
            n_top = int(n_total * pct / 100)
            top_threshold_val = sorted_scores[n_top-1] if n_top > 0 else sorted_scores[0]
            n_above_pct = np.sum(val_scores >= top_threshold_val)
            
            # Expected cases in top percentile
            # Assuming cases are enriched at top
            expected_cases_in_top = min(int(n_cases * tpr * (100/pct)), n_cases)
            
            print(f"  Top {pct:>3}% (n={n_top:>6,}): PRS >= {top_threshold_val:.6f}")
            print(f"    Expected cases: ~{expected_cases_in_top:,} ({expected_cases_in_top/n_top*100:.1f}% of this group)")
        
        # Rank analysis - where do we expect the 2500 cases to be?
        print(f"\n📍 EXPECTED CASE DISTRIBUTION BY RANK:")
        print(f"  If cases have higher PRS (as expected):")
        
        # Best case: all cases in top 2500
        print(f"    Best case: All {n_cases:,} cases in top {n_cases:,} samples")
        print(f"      - Rank 1 to {n_cases:,}")
        print(f"      - PRS threshold: >= {sorted_scores[n_cases-1]:.6f}")
        
        # Expected based on TPR
        expected_cases_in_top_2500 = int(n_cases * tpr)
        print(f"\n    Expected (based on TPR={tpr:.4f}):")
        print(f"      - ~{expected_cases_in_top_2500:,} cases in top {n_cases:,} samples")
        print(f"      - ~{n_cases - expected_cases_in_top_2500:,} cases scattered below")
        
        # Worst case
        print(f"\n    Worst case: Cases evenly distributed")
        print(f"      - Cases would be scattered throughout all {n_total:,} samples")
        print(f"      - Case prevalence: {n_cases/n_total*100:.2f}%")
        
        # Percentile breakdown
        print(f"\n📊 VALIDATION SAMPLES BY SCORE PERCENTILES:")
        percentile_ranges = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100)]
        
        for p_low, p_high in percentile_ranges:
            p_low_val = np.percentile(val_scores, p_low)
            p_high_val = np.percentile(val_scores, p_high)
            n_in_range = np.sum((val_scores >= p_low_val) & (val_scores < p_high_val))
            
            # Expected cases in this range (rough estimate)
            if p_high >= 75:  # High risk
                expected_case_fraction = tpr * ((100 - p_low) / 100)
            elif p_low <= 25:  # Low risk
                expected_case_fraction = (1 - tpr) * (p_high / 100)
            else:  # Medium risk
                expected_case_fraction = 0.05  # Assume some cases
            
            expected_cases_in_range = int(n_cases * expected_case_fraction)
            
            print(f"  Percentile {p_low:>3}-{p_high:<3}%: n={n_in_range:>6,}, "
                  f"PRS=[{p_low_val:.4f}, {p_high_val:.4f}), "
                  f"Expected cases: ~{expected_cases_in_range:,}")
        
        print("\n" + "="*80)

def generate_submission_file(val_prs, phenotype, fold, method, output_dir="Submissions"):
    """Generate CAGI submission file in the required format."""
    
    print("\n" + "="*80)
    print("GENERATING CAGI SUBMISSION FILE")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if val_prs is None:
        print("❌ No validation PRS data available!")
        return None
    
    # Prepare submission dataframe
    submission_df = val_prs[['FID', 'IID', 'PRS']].copy()
    
    # Remove any NaN values
    submission_df = submission_df.dropna()
    
    # Sort by IID for consistency
    submission_df = submission_df.sort_values(by='IID')
    
    # Generate filename
    submission_filename = f"{phenotype}_Fold{fold}_{method}_submission.txt"
    submission_path = os.path.join(output_dir, submission_filename)
    
    # Save as tab-delimited file
    submission_df.to_csv(submission_path, sep='\t', index=False, 
                         float_format='%.10f')
    
    # Print summary
    print(f"\n✓ Submission file created successfully!")
    print(f"  File: {submission_path}")
    print(f"  Format: Tab-delimited text file")
    print(f"  Samples: {len(submission_df):,}")
    print(f"\nFile preview (first 10 rows):")
    print(submission_df.head(10).to_string(index=False))
    
    # Print statistics
    print(f"\n📊 PRS Score Statistics:")
    print(f"  Mean:   {submission_df['PRS'].mean():.6f}")
    print(f"  Median: {submission_df['PRS'].median():.6f}")
    print(f"  Std:    {submission_df['PRS'].std():.6f}")
    print(f"  Min:    {submission_df['PRS'].min():.6f}")
    print(f"  Max:    {submission_df['PRS'].max():.6f}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n📈 PRS Score Percentiles:")
    for p in percentiles:
        val = np.percentile(submission_df['PRS'].values, p)
        print(f"  {p:>3}th: {val:.6f}")
    
    print("\n" + "="*80)
    print("SUBMISSION FILE FORMAT VALIDATION")
    print("="*80)
    
    # Validate format
    required_columns = ['FID', 'IID', 'PRS']
    missing_cols = [col for col in required_columns if col not in submission_df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
    else:
        print("✓ All required columns present: FID, IID, PRS")
    
    # Check data types
    print(f"\n✓ Column types:")
    print(f"  FID: {submission_df['FID'].dtype}")
    print(f"  IID: {submission_df['IID'].dtype}")
    print(f"  PRS: {submission_df['PRS'].dtype}")
    
    # Check for duplicates
    n_duplicates = submission_df['IID'].duplicated().sum()
    if n_duplicates > 0:
        print(f"\n⚠️  Warning: {n_duplicates} duplicate IIDs found!")
    else:
        print(f"\n✓ No duplicate IIDs found")
    
    # Check for missing values
    n_missing = submission_df.isnull().sum().sum()
    if n_missing > 0:
        print(f"⚠️  Warning: {n_missing} missing values found!")
    else:
        print(f"✓ No missing values")
    
    print("\n✓ Submission file ready for CAGI submission!")
    print("="*80)
    
    # Create a README file
    readme_path = os.path.join(output_dir, f"{phenotype}_README.txt")
    with open(readme_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CAGI7 PRS CHALLENGE - SUBMISSION FILE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Phenotype: {phenotype}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Submission File: {submission_filename}\n\n")
        f.write(f"File Format:\n")
        f.write(f"  - Tab-delimited text file\n")
        f.write(f"  - Columns: FID, IID, PRS\n")
        f.write(f"     - Number of samples: {len(submission_df):,}\n\n")
        f.write(f"PRS Statistics:\n")
        f.write(f"  - Mean:   {submission_df['PRS'].mean():.6f}\n")
        f.write(f"  - Median: {submission_df['PRS'].median():.6f}\n")
        f.write(f"  - Std:    {submission_df['PRS'].std():.6f}\n")
        f.write(f"  - Range:  [{submission_df['PRS'].min():.6f}, {submission_df['PRS'].max():.6f}]\n\n")
        f.write(f"CAGI Challenge Requirements:\n")
        f.write(f"  ✓ FID: Sample FID from validation cohort\n")
        f.write(f"  ✓ IID: Sample IID from validation cohort\n")
        f.write(f"  ✓ PRS: Predicted risk score (higher = higher risk)\n\n")
        f.write(f"Notes:\n")
        f.write(f"  - PRS values are real-valued numbers\n")
        f.write(f"  - Higher PRS indicates higher predicted disease risk\n")
        f.write(f"  - Based on {method} method\n")
        f.write(f"  - Generated from Fold {fold} analysis\n\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ README file created: {readme_path}")
    
    return submission_path

def plot_pca_analysis(train_metrics, test_metrics, val_prs, phenotype, fold, method):
    """Generate PCA plots for train, test, and validation sets."""
    
    print("\n" + "="*80)
    print("GENERATING PCA ANALYSIS")
    print("="*80)
    
    # Prepare data
    train_data = {
        'scores': train_metrics['y_scores'],
        'labels': train_metrics['y_true'],
        'name': 'Training'
    }
    
    test_data = {
        'scores': test_metrics['y_scores'],
        'labels': test_metrics['y_true'],
        'name': 'Test'
    }
    
    val_scores = val_prs['PRS'].values
    val_scores = val_scores[~np.isnan(val_scores)]
    val_data = {
        'scores': val_scores,
        'labels': np.full(len(val_scores), -1),
        'name': 'Validation'
    }
    
    # Create features matrix
    def create_features(scores):
        features = []
        features.append(scores)
        features.append(scores ** 2)
        features.append((scores - np.mean(scores)) / np.std(scores))
        ranks = np.argsort(np.argsort(scores))
        features.append(ranks)
        features.append(ranks / len(ranks) * 100)
        return np.column_stack(features)
    
    train_features = create_features(train_data['scores'])
    test_features = create_features(test_data['scores'])
    val_features = create_features(val_data['scores'])
    
    # Fit PCA
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    val_features_scaled = scaler.transform(val_features)
    
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_features_scaled)
    test_pca = pca.transform(test_features_scaled)
    val_pca = pca.transform(val_features_scaled)
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Create PCA plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Training
    ax1 = axes[0, 0]
    train_cases = train_pca[train_data['labels'] == 1]
    train_controls = train_pca[train_data['labels'] == 0]
    
    ax1.scatter(train_controls[:, 0], train_controls[:, 1], 
               c='#2ca02c', alpha=0.5, s=30, label=f'Controls (n={len(train_controls)})', 
               edgecolors='black', linewidth=0.5)
    ax1.scatter(train_cases[:, 0], train_cases[:, 1], 
               c='#d62728', alpha=0.7, s=30, label=f'Cases (n={len(train_cases)})', 
               edgecolors='black', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
    ax1.set_title('Training Set PCA', fontweight='bold', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test
    ax2 = axes[0, 1]
    test_cases = test_pca[test_data['labels'] == 1]
    test_controls = test_pca[test_data['labels'] == 0]
    
    ax2.scatter(test_controls[:, 0], test_controls[:, 1], 
               c='#2ca02c', alpha=0.5, s=30, label=f'Controls (n={len(test_controls)})')
    ax2.scatter(test_cases[:, 0], test_cases[:, 1], 
               c='#d62728', alpha=0.7, s=30, label=f'Cases (n={len(test_cases)})')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
    ax2.set_title('Test Set PCA', fontweight='bold', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation
    ax3 = axes[1, 0]
    scatter = ax3.scatter(val_pca[:, 0], val_pca[:, 1], 
                         c=val_data['scores'], alpha=0.6, s=20, 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=0.3)
    plt.colorbar(scatter, ax=ax3, label='PRS Score')
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
    ax3.set_title('Validation Set PCA (colored by PRS)', fontweight='bold', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All combined
    ax4 = axes[1, 1]
    ax4.scatter(train_pca[:, 0], train_pca[:, 1], 
               c='#1f77b4', alpha=0.3, s=15, label='Train')
    ax4.scatter(test_pca[:, 0], test_pca[:, 1], 
               c='#ff7f0e', alpha=0.4, s=20, label='Test')
    ax4.scatter(val_pca[:, 0], val_pca[:, 1], 
               c='#9467bd', alpha=0.5, s=10, label='Validation')
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
    ax4.set_title('All Datasets Combined', fontweight='bold', fontsize=13)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'PCA Analysis - {phenotype} (Fold {fold})\nMethod: {method}', 
                fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_PCA_Analysis.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_PCA_Analysis.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Generated PCA analysis plots")
    
    return {'train_pca': train_pca, 'test_pca': test_pca, 'val_pca': val_pca}

def plot_validation_auc_estimates(val_auc_est, train_auc, test_auc, phenotype, fold):
    """Plot validation AUC estimates."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: AUC comparison bars
    categories = ['Train', 'Test', 'Expected\nValidation',
                 'Random', 'Best Case', 'Worst Case']
    values = [
        train_auc,
        test_auc,
        val_auc_est['expected_auc'],
        val_auc_est['random_auc_mean'],
        val_auc_est['best_case_auc'],
        val_auc_est['worst_case_auc']
    ]
    errors = [
        0, 0,
        (val_auc_est['expected_range'][1] - val_auc_est['expected_range'][0])/2,
        val_auc_est['random_auc_std'],
        0, 0
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5,
                   yerr=errors, capsize=5, error_kw={'linewidth': 2})
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('AUC', fontweight='bold', fontsize=13)
    ax1.set_title('AUC Comparison: Observed and Expected',
                 fontweight='bold', fontsize=14)
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    # Plot 2: Random AUC distribution
    ax2.hist(val_auc_est['random_aucs'], bins=50, alpha=0.7, color='#9467bd',
            edgecolor='black', density=True, label='Random Label Assignment')
    
    ax2.axvline(val_auc_est['expected_auc'], color='#2ca02c', linewidth=3,
               linestyle='--', label=f'Expected AUC: {val_auc_est["expected_auc"]:.4f}')
    
    ax2.axvline(test_auc, color='#ff7f0e', linewidth=3,
               linestyle='--', label=f'Test AUC: {test_auc:.4f}')
    
    ax2.axvspan(val_auc_est['expected_range'][0],
               val_auc_est['expected_range'][1],
               alpha=0.2, color='green', label='Expected Range')
    
    ax2.set_xlabel('AUC', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Density', fontweight='bold', fontsize=13)
    ax2.set_title('Validation AUC: Random vs Expected Distribution',
                 fontweight='bold', fontsize=14)
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    
    plt.suptitle(f'Validation AUC Estimates - {phenotype} (Fold {fold})',
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_Validation_AUC_Estimates.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_Validation_AUC_Estimates.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Generated validation AUC estimates plot")

def save_summary_table(phenotype, fold, method, best_row, train_metrics, test_metrics, val_auc_est, threshold_results, output_dir="Submissions"):
    """Save a summary table with all key metrics."""
    
    print("\n" + "="*80)
    print("SAVING SUMMARY TABLE")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary dictionary
    summary_data = {
        'Phenotype': phenotype,
        'Fold': fold,
        'Method': method,
        'P-value': best_row.get('pvalue', 'N/A'),
        'Train_AUC': train_metrics['auc'],
        'Train_Cases': train_metrics['n_cases'],
        'Train_Controls': train_metrics['n_controls'],
        'Test_AUC': test_metrics['auc'],
        'Test_Cases': test_metrics['n_cases'],
        'Test_Controls': test_metrics['n_controls'],
    }
    
    if val_auc_est is not None:
        summary_data.update({
            'Expected_Validation_AUC': val_auc_est['expected_auc'],
            'Expected_Range_Lower': val_auc_est['expected_range'][0],
            'Expected_Range_Upper': val_auc_est['expected_range'][1],
            'Monte_Carlo_Mean_AUC': val_auc_est['random_auc_mean'],
            'Monte_Carlo_Std_AUC': val_auc_est['random_auc_std'],
            'Best_Case_AUC': val_auc_est['best_case_auc'],
            'Worst_Case_AUC': val_auc_est['worst_case_auc'],
        })
    
    if threshold_results is not None and 'Average' in threshold_results:
        avg_res = threshold_results['Average']
        summary_data.update({
            'Threshold_Samples_Above': avg_res['n_above'],
            'Threshold_Samples_Below': avg_res['n_below'],
            'Threshold_Expected_AUC': avg_res['expected_auc'],
            'Threshold_Best_Case_AUC': avg_res['best_case_auc'],
            'Threshold_Random_AUC': avg_res['random_auc'],
        })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame([summary_data])
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{phenotype}_Summary_Table.csv")
    summary_df.to_csv(csv_path, index=False, float_format='%.6f')
    
    # Save as formatted text table
    txt_path = os.path.join(output_dir, f"{phenotype}_Summary_Table.txt")
    with open(txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"PRS ANALYSIS SUMMARY - {phenotype}\n")
        f.write("="*100 + "\n\n")
        
        f.write("STUDY DESIGN\n")
        f.write("-"*100 + "\n")
        f.write(f"  Phenotype:                 {phenotype}\n")
        f.write(f"  Fold:                      {fold}\n")
        f.write(f"  Best Method:               {method}\n")
        f.write(f"  P-value Threshold:         {best_row.get('pvalue', 'N/A')}\n")
        f.write("\n")
        
        f.write("TRAINING SET PERFORMANCE\n")
        f.write("-"*100 + "\n")
        f.write(f"  AUC:                       {train_metrics['auc']:.6f}\n")
        f.write(f"  Cases:                     {train_metrics['n_cases']:,}\n")
        f.write(f"  Controls:                  {train_metrics['n_controls']:,}\n")
        f.write("\n")
        
        f.write("TEST SET PERFORMANCE\n")
        f.write("-"*100 + "\n")
        f.write(f"  AUC:                       {test_metrics['auc']:.6f}\n")
        f.write(f"  Cases:                     {test_metrics['n_cases']:,}\n")
        f.write(f"  Controls:                  {test_metrics['n_controls']:,}\n")
        f.write("\n")
        
        if val_auc_est is not None:
            f.write("VALIDATION SET ESTIMATES (Monte Carlo Simulation)\n")
            f.write("-"*100 + "\n")
            f.write(f"  Expected Validation AUC:   {val_auc_est['expected_auc']:.6f}\n")
            f.write(f"  Expected Range:            [{val_auc_est['expected_range'][0]:.6f}, {val_auc_est['expected_range'][1]:.6f}]\n")
            f.write(f"  Monte Carlo Mean AUC:      {val_auc_est['random_auc_mean']:.6f}\n")
            f.write(f"  Monte Carlo Std AUC:       {val_auc_est['random_auc_std']:.6f}\n")
            f.write(f"  Best Case AUC:             {val_auc_est['best_case_auc']:.6f}\n")
            f.write(f"  Worst Case AUC:            {val_auc_est['worst_case_auc']:.6f}\n")
            f.write("\n")
        
        if threshold_results is not None and 'Average' in threshold_results:
            avg_res = threshold_results['Average']
            f.write("THRESHOLD-BASED VALIDATION ESTIMATES\n")
            f.write("-"*100 + "\n")
            f.write(f"  Samples Above Threshold:   {avg_res['n_above']:,}\n")
            f.write(f"  Samples Below Threshold:   {avg_res['n_below']:,}\n")
            f.write(f"  Expected AUC:              {avg_res['expected_auc']:.6f}\n")
            f.write(f"  Best Case AUC:             {avg_res['best_case_auc']:.6f}\n")
            f.write(f"  Random AUC:                {avg_res['random_auc']:.6f}\n")
            f.write("\n")
        
        f.write("="*100 + "\n")
    
    print(f"\n✓ Summary table saved:")
    print(f"  CSV:  {csv_path}")
    print(f"  TXT:  {txt_path}")
    
    # Display table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)
    
    return csv_path, txt_path

def main():
    """Main execution function."""
    
    print("="*80)
    print("COMPLETE PRS AUC ANALYSIS - ALL METHODS")
    print("="*80)
    print(f"Phenotype: {PHENOTYPE}")
    print(f"Methods to check: {', '.join(ALL_METHODS)}")
    print("="*80)
    
    # Find best performing method across all
    best_row, best_method, available_folds, extracted_common_rows = find_best_method_and_parameters(
        PHENOTYPE, ALL_METHODS
    )
    
    if best_row is None:
        print("\n❌ Could not find any valid methods!")
        return
    
    # Use the first available fold for detailed analysis
    best_fold = available_folds[0]
    print(f"\nUsing Fold_{best_fold} for detailed analysis")
    
    # Load data for the best fold
    data = load_prs_and_pheno(PHENOTYPE, best_fold, best_method, best_row)
    
    if data['train_prs'] is None or data['test_prs'] is None:
        print("\n❌ Could not load required PRS data!")
        return
    
    # Calculate metrics
    print("\n" + "="*80)
    print("CALCULATING AUC METRICS")
    print("="*80)
    
    train_metrics = calculate_auc_metrics(data['train_prs'], data['train_pheno'])
    test_metrics = calculate_auc_metrics(data['test_prs'], data['test_pheno'])
    
    if train_metrics is None or test_metrics is None:
        print("\n❌ Could not calculate AUC metrics!")
        return
    
    # Print results
    print("\n" + "="*80)
    print("AUC RESULTS")
    print("="*80)
    print(f"Method:         {best_method}")
    print(f"Fold:           {best_fold}")
    if 'pvalue' in best_row:
        print(f"P-value:        {best_row['pvalue']}")
    print(f"\nTraining AUC:   {train_metrics['auc']:.6f}")
    print(f"  Cases:        {train_metrics['n_cases']}")
    print(f"  Controls:     {train_metrics['n_controls']}")
    print(f"\nTest AUC:       {test_metrics['auc']:.6f}")
    print(f"  Cases:        {test_metrics['n_cases']}")
    print(f"  Controls:     {test_metrics['n_controls']}")
    print(f"\nAUC Difference: {abs(train_metrics['auc'] - test_metrics['auc']):.6f}")
    print("="*80)
    
    # Print confusion matrices for train and test
    confusion_results = print_train_test_confusion_matrices(train_metrics, test_metrics)
    
    # Threshold-based validation prediction
    threshold_results = None
    if data['val_prs'] is not None:
        threshold_results = estimate_validation_performance_by_threshold(
            train_metrics, test_metrics, data['val_prs'],
            n_cases=2500, n_total=50000
        )
        
        # Print detailed validation sample distribution
        print_validation_sample_distribution_detailed(
            threshold_results, data['val_prs'], n_cases=2500, n_total=50000
        )
        
        # Print estimated validation confusion matrices
        print_validation_estimated_confusion_matrices(
            threshold_results, n_cases=2500, n_total=50000
        )
    
    # Generate CAGI submission file
    submission_file = None
    if data['val_prs'] is not None:
        submission_file = generate_submission_file(
            data['val_prs'], PHENOTYPE, best_fold, best_method
        )
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\n1. ROC Curves...")
    plot_roc_curves(train_metrics, test_metrics, PHENOTYPE, best_method, best_fold)
    
    print("\n2. PRS Distributions...")
    if data['val_prs'] is not None:
        plot_prs_distributions(train_metrics, test_metrics, data['val_prs'],
                              PHENOTYPE, best_method, best_fold)
    
    print("\n3. AUC Points Distribution...")
    plot_auc_points_distribution(train_metrics, test_metrics, PHENOTYPE, best_fold)
    
    # PCA Analysis
    if data['val_prs'] is not None:
        print("\n4. PCA Analysis...")
        pca_results = plot_pca_analysis(train_metrics, test_metrics, data['val_prs'],
                                       PHENOTYPE, best_fold, best_method)
    
    # Validation AUC estimation
    val_auc_est = None
    if data['val_prs'] is not None:
        print("\n" + "="*80)
        print("VALIDATION AUC ESTIMATION")
        print("="*80)
        
        val_scores = data['val_prs']['PRS'].values
        val_scores = val_scores[~np.isnan(val_scores)]
        
        val_auc_est = estimate_validation_auc_range(
            val_scores,
            n_cases=2500,
            n_total=50000,
            train_auc=train_metrics['auc'],
            test_auc=test_metrics['auc'],
            n_simulations=10000
        )
        
        # Print validation estimates
        print("\n" + "="*80)
        print("VALIDATION AUC ESTIMATES")
        print("="*80)
        print(f"Expected Validation AUC:  {val_auc_est['expected_auc']:.6f}")
        print(f"Expected Range:           [{val_auc_est['expected_range'][0]:.6f}, "
              f"{val_auc_est['expected_range'][1]:.6f}]")
        print(f"\nTheoretical Range:        [{val_auc_est['theoretical_range'][0]:.6f}, "
              f"{val_auc_est['theoretical_range'][1]:.6f}]")
        print(f"  Best Case AUC:          {val_auc_est['best_case_auc']:.6f} "
              f"(top 2500 scores are cases)")
        print(f"  Worst Case AUC:         {val_auc_est['worst_case_auc']:.6f} "
              f"(bottom 2500 scores are cases)")
        print(f"\nRandom Assignment:        {val_auc_est['random_auc_mean']:.6f} "
              f"± {val_auc_est['random_auc_std']:.6f}")
        
        # Add threshold-based estimates
        if threshold_results:
            print(f"\nThreshold-Based Estimates:")
            print(f"  Using Average Threshold:  {threshold_results['Average']['expected_auc']:.6f}")
            print(f"  Best case (if enriched):  {threshold_results['Average']['best_case_auc']:.6f}")
            print(f"  Random distribution:      {threshold_results['Average']['random_auc']:.6f}")
        
        print("="*80)
        
        print("\n5. Validation AUC Estimates...")
        plot_validation_auc_estimates(val_auc_est, train_metrics['auc'],
                                     test_metrics['auc'], PHENOTYPE, best_fold)
    
    # Save summary table (NEW)
    csv_summary, txt_summary = save_summary_table(
        PHENOTYPE, best_fold, best_method, best_row,
        train_metrics, test_metrics, val_auc_est, threshold_results
    )
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   Phenotype:      {PHENOTYPE}")
    print(f"   Best Method:    {best_method}")
    print(f"   Fold:           Fold_{best_fold}")
    if 'pvalue' in best_row:
        print(f"   P-value:        {best_row['pvalue']}")
    print(f"\n📈 Performance Metrics:")
    print(f"   Training AUC:   {train_metrics['auc']:.6f}")
    print(f"   Test AUC:       {test_metrics['auc']:.6f}")
    
    if confusion_results:
        print(f"\n📋 Confusion Matrix Metrics:")
        print(f"   Train Sensitivity: {confusion_results['train']['metrics']['sensitivity']:.4f}")
        print(f"   Train Specificity: {confusion_results['train']['metrics']['specificity']:.4f}")
        print(f"   Test Sensitivity:  {confusion_results['test']['metrics']['sensitivity']:.4f}")
        print(f"   Test Specificity:  {confusion_results['test']['metrics']['specificity']:.4f}")
    
    if val_auc_est is not None:
        print(f"\n   Expected Val AUC: {val_auc_est['expected_auc']:.6f}")
        print(f"   Possible Range:   [{val_auc_est['expected_range'][0]:.6f}, "
              f"{val_auc_est['expected_range'][1]:.6f}]")
        
        if threshold_results:
            avg_res = threshold_results['Average']
            print(f"\n📍 Threshold-Based Predictions:")
            print(f"   Samples above threshold: {avg_res['n_above']:,} ({avg_res['n_above']/50000*100:.1f}%)")
            print(f"   Expected AUC (threshold): {avg_res['expected_auc']:.6f}")
            print(f"   Best case AUC:            {avg_res['best_case_auc']:.6f}")
    
    print(f"\n📁 Generated files in {RESULTS_OUTPUT_DIR}/:")
    print(f"   - {PHENOTYPE}_Fold{best_fold}_ROC_Curves.png/.pdf")
    print(f"   - {PHENOTYPE}_Fold{best_fold}_PRS_Distributions.png/.pdf")
    print(f"   - {PHENOTYPE}_Fold{best_fold}_AUC_Points_Distribution.png/.pdf")
    print(f"   - {PHENOTYPE}_Fold{best_fold}_PCA_Analysis.png/.pdf")
    if val_auc_est is not None:
        print(f"   - {PHENOTYPE}_Fold{best_fold}_Validation_AUC_Estimates.png/.pdf")
    
    if submission_file:
        print(f"\n📤 CAGI Submission Files:")
        print(f"   - {submission_file}")
        print(f"   - Submissions/{PHENOTYPE}_README.txt")
        print(f"   - {csv_summary}")
        print(f"   - {txt_summary}")
    
    print("="*80)

if __name__ == "__main__":
    main()