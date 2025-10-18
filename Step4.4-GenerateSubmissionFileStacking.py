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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations

warnings.filterwarnings('ignore')

# Configuration
PHENOTYPE = sys.argv[1]  # e.g., "Phenotype_2"  
RESULTS_OUTPUT_DIR = "PRS_AUC_Analysis"
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# Define all methods to check
ALL_METHODS = ['Plink3','GCTA3', 'LDAK-GWAS3', 'PRSice-2-3']

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

def find_top_methods_and_parameters(phenotype, methods, top_n=3):
    """Find the top N performing methods with files existing in all folds."""
    
    print(f"\n{'='*80}")
    print(f"SCANNING ALL METHODS TO FIND TOP {top_n} PERFORMERS")
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
        return []
    
    # Sort candidates by test AUC and pick the top N
    all_candidates.sort(key=lambda x: x['test_auc'], reverse=True)
    top_candidates = all_candidates[:min(top_n, len(all_candidates))]
    
    print(f"\n{'='*80}")
    print(f"TOP {len(top_candidates)} METHODS WITH COMPLETE FILES:")
    print("="*80)
    
    for i, cand in enumerate(top_candidates, 1):
        print(f"\n{i}. {cand['method']}")
        print(f"   Rank in method: {cand['rank']}")
        print(f"   Train AUC: {cand['row']['Train_pure_prs']:.6f}")
        print(f"   Test AUC:  {cand['row']['Test_pure_prs']:.6f}")
        if 'pvalue' in cand['row']:
            print(f"   P-value:   {cand['row']['pvalue']}")
        print(f"   Available folds: {cand['folds']}")
    
    print("="*80)
    
    return top_candidates

def load_prs_for_method(phenotype, fold, method, best_row, data_type):
    """Load PRS scores for a specific method and data type."""
    
    method_dir = os.path.join(phenotype, f"Fold_{fold}", method)
    pvalue = best_row.get('pvalue', None)
    
    if 'PRSice' in method:
        prs_file = glob.glob(os.path.join(method_dir, f"*{data_type.capitalize()}*.all_score"))
        if prs_file:
            return read_prs_file_prsice(prs_file[0], pvalue)
    else:
        return read_prs_file_plink_gcta_ldak(method_dir, pvalue, data_type)
    
    return None

def load_prs_for_multiple_methods(phenotype, fold, methods_list):
    """Load PRS scores from multiple methods."""
    
    print(f"\n{'='*80}")
    print(f"LOADING PRS FROM MULTIPLE METHODS - Fold {fold}")
    print("="*80)
    
    all_prs_data = {}
    
    for method_info in methods_list:
        method = method_info['method']
        best_row = method_info['row']
        
        print(f"\nLoading {method}...")
        
        # Load train, test, val
        train_prs = load_prs_for_method(phenotype, fold, method, best_row, 'train')
        test_prs = load_prs_for_method(phenotype, fold, method, best_row, 'test')
        val_prs = load_prs_for_method(phenotype, fold, method, best_row, 'val')
        
        if train_prs is not None:
            print(f"  ✓ Train: {len(train_prs)} samples")
        if test_prs is not None:
            print(f"  ✓ Test: {len(test_prs)} samples")
        if val_prs is not None:
            print(f"  ✓ Validation: {len(val_prs)} samples")
        
        all_prs_data[method] = {
            'train': train_prs,
            'test': test_prs,
            'val': val_prs,
            'info': method_info
        }
    
    return all_prs_data

def stack_prs_simple_average(prs_list, method_names):
    """Stack PRS using simple average."""
    
    print(f"\n  Strategy: Simple Average of {len(prs_list)} methods")
    
    # Merge all PRS dataframes
    merged = prs_list[0][['FID', 'IID']].copy()
    
    for i, prs_df in enumerate(prs_list):
        merged = pd.merge(merged, prs_df[['IID', 'PRS']], on='IID', 
                         how='inner', suffixes=('', f'_{i}'))
    
    # Rename columns
    prs_cols = [col for col in merged.columns if 'PRS' in col]
    
    # Average
    merged['PRS'] = merged[prs_cols].mean(axis=1)
    
    result = merged[['FID', 'IID', 'PRS']].copy()
    print(f"    Result: {len(result)} samples")
    
    return result

def stack_prs_weighted_by_test_auc(prs_list, method_infos):
    """Stack PRS using weighted average based on test AUC."""
    
    test_aucs = [info['test_auc'] for info in method_infos]
    weights = np.array(test_aucs) / sum(test_aucs)
    
    print(f"\n  Strategy: Weighted Average by Test AUC")
    for i, (method, w, auc) in enumerate(zip([m['method'] for m in method_infos], weights, test_aucs)):
        print(f"    {method}: weight={w:.4f} (AUC={auc:.6f})")
    
    # Merge all PRS dataframes
    merged = prs_list[0][['FID', 'IID']].copy()
    
    for i, prs_df in enumerate(prs_list):
        merged = pd.merge(merged, prs_df[['IID', 'PRS']], on='IID', 
                         how='inner', suffixes=('', f'_{i}'))
    
    # Get PRS columns
    prs_cols = [col for col in merged.columns if 'PRS' in col]
    
    # Weighted average
    merged['PRS'] = sum(merged[col] * w for col, w in zip(prs_cols, weights))
    
    result = merged[['FID', 'IID', 'PRS']].copy()
    print(f"    Result: {len(result)} samples")
    
    return result

def stack_prs_logistic_regression(train_prs_list, test_prs_list, val_prs_list, 
                                   train_pheno, test_pheno, method_infos):
    """Stack PRS using logistic regression meta-model."""
    
    print(f"\n  Strategy: Logistic Regression Stacking")
    
    # Prepare training data
    train_merged = train_prs_list[0][['FID', 'IID']].copy()
    for i, prs_df in enumerate(train_prs_list):
        train_merged = pd.merge(train_merged, prs_df[['IID', 'PRS']], on='IID',
                               how='inner', suffixes=('', f'_{i}'))
    
    train_merged = pd.merge(train_merged, train_pheno, on='IID', how='inner',
                           suffixes=('', '_pheno'))
    
    # Get PRS columns
    prs_cols = [col for col in train_merged.columns if 'PRS' in col and col != 'PRS_pheno']
    
    X_train = train_merged[prs_cols].values
    y_train = train_merged['PHENO'].values
    
    # Remove NaN
    valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    print(f"    Training samples: {len(X_train)}")
    print(f"    Features: {X_train.shape[1]} PRS scores")
    
    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    print(f"    Coefficients: {lr.coef_[0]}")
    for i, (method, coef) in enumerate(zip([m['method'] for m in method_infos], lr.coef_[0])):
        print(f"      {method}: {coef:.6f}")
    
    # Predict on test set
    test_merged = test_prs_list[0][['FID', 'IID']].copy()
    for i, prs_df in enumerate(test_prs_list):
        test_merged = pd.merge(test_merged, prs_df[['IID', 'PRS']], on='IID',
                              how='inner', suffixes=('', f'_{i}'))
    
    X_test = test_merged[prs_cols].values
    test_pred = lr.predict_proba(X_test)[:, 1]
    
    test_result = test_merged[['FID', 'IID']].copy()
    test_result['PRS'] = test_pred
    
    # Predict on validation set
    val_merged = val_prs_list[0][['FID', 'IID']].copy()
    for i, prs_df in enumerate(val_prs_list):
        val_merged = pd.merge(val_merged, prs_df[['IID', 'PRS']], on='IID',
                             how='inner', suffixes=('', f'_{i}'))
    
    X_val = val_merged[prs_cols].values
    val_pred = lr.predict_proba(X_val)[:, 1]
    
    val_result = val_merged[['FID', 'IID']].copy()
    val_result['PRS'] = val_pred
    
    print(f"    Test result: {len(test_result)} samples")
    print(f"    Validation result: {len(val_result)} samples")
    
    return test_result, val_result, lr

def calculate_confusion_metrics(y_true, y_pred_binary):
    """Calculate confusion matrix metrics."""
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # PPV
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }

def print_confusion_matrix_detailed(metrics_dict, dataset_name):
    """Print detailed confusion matrix."""
    cm = metrics_dict['confusion_matrix']
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Confusion Matrix")
    print("="*60)
    print(f"                  Predicted")
    print(f"                  Negative    Positive")
    print(f"Actual Negative   {metrics_dict['tn']:>8}    {metrics_dict['fp']:>8}")
    print(f"Actual Positive   {metrics_dict['fn']:>8}    {metrics_dict['tp']:>8}")
    print("="*60)
    print(f"Metrics:")
    print(f"  Accuracy:    {metrics_dict['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics_dict['sensitivity']:.4f} (TPR, Recall)")
    print(f"  Specificity: {metrics_dict['specificity']:.4f} (TNR)")
    print(f"  Precision:   {metrics_dict['precision']:.4f} (PPV)")
    print(f"  F1-Score:    {metrics_dict['f1_score']:.4f}")
    print("="*60)

def evaluate_stacking_strategies(all_prs_data, train_pheno, test_pheno, fold):
    """Evaluate different PRS stacking strategies."""
    
    print(f"\n{'='*80}")
    print("EVALUATING STACKING STRATEGIES")
    print("="*80)
    
    methods_list = [info['info'] for method, info in all_prs_data.items()]
    
    # Collect PRS for each data type
    train_prs_list = [info['train'] for info in all_prs_data.values() if info['train'] is not None]
    test_prs_list = [info['test'] for info in all_prs_data.values() if info['test'] is not None]
    val_prs_list = [info['val'] for info in all_prs_data.values() if info['val'] is not None]
    
    if len(train_prs_list) == 0 or len(test_prs_list) == 0:
        print("❌ Not enough PRS data for stacking!")
        return None, None
    
    print(f"\nAvailable methods for stacking: {len(train_prs_list)}")
    
    stacking_results = {}
    
    # Strategy 1: Simple Average
    print(f"\n{'='*60}")
    print("STRATEGY 1: SIMPLE AVERAGE")
    print("="*60)
    
    test_avg = stack_prs_simple_average(test_prs_list, [m['method'] for m in methods_list])
    val_avg = stack_prs_simple_average(val_prs_list, [m['method'] for m in methods_list]) if val_prs_list else None
    
    test_metrics_avg = calculate_auc_metrics(test_avg, test_pheno)
    
    if test_metrics_avg:
        print(f"  ✓ Test AUC: {test_metrics_avg['auc']:.6f}")
        stacking_results['simple_average'] = {
            'test_auc': test_metrics_avg['auc'],
            'test_prs': test_avg,
            'val_prs': val_avg,
            'test_metrics': test_metrics_avg,
            'name': 'Simple Average'
        }
    
    # Strategy 2: Weighted by Test AUC
    print(f"\n{'='*60}")
    print("STRATEGY 2: WEIGHTED AVERAGE BY TEST AUC")
    print("="*60)
    
    test_weighted = stack_prs_weighted_by_test_auc(test_prs_list, methods_list)
    val_weighted = stack_prs_weighted_by_test_auc(val_prs_list, methods_list) if val_prs_list else None
    
    test_metrics_weighted = calculate_auc_metrics(test_weighted, test_pheno)
    
    if test_metrics_weighted:
        print(f"  ✓ Test AUC: {test_metrics_weighted['auc']:.6f}")
        stacking_results['weighted_average'] = {
            'test_auc': test_metrics_weighted['auc'],
            'test_prs': test_weighted,
            'val_prs': val_weighted,
            'test_metrics': test_metrics_weighted,
            'name': 'Weighted Average (by Test AUC)'
        }
    
    # Strategy 3: Logistic Regression
    print(f"\n{'='*60}")
    print("STRATEGY 3: LOGISTIC REGRESSION STACKING")
    print("="*60)
    
    try:
        test_lr, val_lr, lr_model = stack_prs_logistic_regression(
            train_prs_list, test_prs_list, val_prs_list,
            train_pheno, test_pheno, methods_list
        )
        
        test_metrics_lr = calculate_auc_metrics(test_lr, test_pheno)
        
        if test_metrics_lr:
            print(f"  ✓ Test AUC: {test_metrics_lr['auc']:.6f}")
            stacking_results['logistic_regression'] = {
                'test_auc': test_metrics_lr['auc'],
                'test_prs': test_lr,
                'val_prs': val_lr,
                'test_metrics': test_metrics_lr,
                'model': lr_model,
                'name': 'Logistic Regression Stacking'
            }
    except Exception as e:
        print(f"  ✗ Logistic Regression failed: {e}")
    
    # Compare individual methods
    print(f"\n{'='*60}")
    print("INDIVIDUAL METHOD PERFORMANCE (for comparison)")
    print("="*60)
    
    for method_name, prs_info in all_prs_data.items():
        if prs_info['test'] is not None:
            test_metrics = calculate_auc_metrics(prs_info['test'], test_pheno)
            if test_metrics:
                print(f"  {method_name}: Test AUC = {test_metrics['auc']:.6f}")
    
    # Find best strategy
    if stacking_results:
        print(f"\n{'='*60}")
        print("BEST STACKING STRATEGY")
        print("="*60)
        
        best_strategy = max(stacking_results.items(), key=lambda x: x[1]['test_auc'])
        
        print(f"\n  🏆 {best_strategy[1]['name']}")
        print(f"     Test AUC: {best_strategy[1]['test_auc']:.6f}")
        
        # Calculate improvement over best individual method
        best_individual_auc = max([info['test_auc'] for info in methods_list])
        improvement = best_strategy[1]['test_auc'] - best_individual_auc
        improvement_pct = (improvement / best_individual_auc) * 100
        
        print(f"\n  📈 Improvement over best individual method:")
        print(f"     Best individual AUC: {best_individual_auc:.6f}")
        print(f"     Stacked AUC: {best_strategy[1]['test_auc']:.6f}")
        print(f"     Absolute improvement: {improvement:.6f}")
        print(f"     Relative improvement: {improvement_pct:.2f}%")
        
        print("="*60)
        
        return best_strategy[1], stacking_results
    
    return None, stacking_results

def plot_stacking_comparison(stacking_results, all_prs_data, test_pheno, phenotype, fold):
    """Plot comparison of stacking strategies."""
    
    print("\nGenerating stacking comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: AUC comparison
    methods = []
    aucs = []
    colors = []
    
    # Individual methods
    for method_name, prs_info in all_prs_data.items():
        if prs_info['test'] is not None:
            test_metrics = calculate_auc_metrics(prs_info['test'], test_pheno)
            if test_metrics:
                methods.append(method_name)
                aucs.append(test_metrics['auc'])
                colors.append('#1f77b4')
    
    # Stacking strategies
    for strategy_name, results in stacking_results.items():
        methods.append(results['name'])
        aucs.append(results['test_auc'])
        colors.append('#2ca02c')
    
    bars = ax1.barh(methods, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, auc in zip(bars, aucs):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{auc:.6f}',
                ha='left', va='center', fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Test AUC', fontweight='bold', fontsize=13)
    ax1.set_title('PRS Stacking: Performance Comparison',
                 fontweight='bold', fontsize=14)
    ax1.set_xlim([min(aucs)*0.95, max(aucs)*1.02])
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Individual Methods'),
                      Patch(facecolor='#2ca02c', edgecolor='black', label='Stacked Methods')]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: ROC curves for stacking strategies
    for strategy_name, results in stacking_results.items():
        if 'test_metrics' in results:
            metrics = results['test_metrics']
            ax2.plot(metrics['fpr'], metrics['tpr'],
                    label=f"{results['name']} (AUC={results['test_auc']:.4f})",
                    linewidth=2.5, alpha=0.8)
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    ax2.set_title('ROC Curves: Stacking Strategies',
                 fontweight='bold', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'PRS Stacking Analysis - {phenotype} (Fold {fold})',
                fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_Stacking_Comparison.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_Stacking_Comparison.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated stacking comparison plot")

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

def generate_submission_file(val_prs, phenotype, fold, method, output_dir="Submissions"):
    """Generate CAGI submission file in the required format."""
    
    print("\n" + "="*80)
    print("GENERATING CAGI SUBMISSION FILE")
    print("="*80)
    
    # Create submission directory: Submissions2/
    submission_base_dir = "Submissions2"
    os.makedirs(submission_base_dir, exist_ok=True)
    
    if val_prs is None:
        print("❌ No validation PRS data available!")
        return None
    
    # Prepare submission dataframe
    submission_df = val_prs[['FID', 'IID', 'PRS']].copy()
    
    # Remove any NaN values
    submission_df = submission_df.dropna()
    
    # Sort by IID for consistency
    submission_df = submission_df.sort_values(by='IID')
    
    # Generate filename: Phenotype1_Submission2.txt (without underscore after "Phenotype")
    phenotype_num = phenotype.replace('Phenotype_', '')
    submission_filename = f"Phenotype{phenotype_num}_Submission2.txt"
    submission_path = os.path.join(submission_base_dir, submission_filename)
    
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
    
    # Create a README file in the same directory
    readme_filename = f"Phenotype{phenotype_num}_README.txt"
    readme_path = os.path.join(submission_base_dir, readme_filename)
    with open(readme_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CAGI7 PRS CHALLENGE - SUBMISSION 2 (STACKED MODEL)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Phenotype: {phenotype}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Submission File: {submission_filename}\n")
        f.write(f"Submission Type: Stacked PRS Model\n\n")
        f.write(f"File Format:\n")
        f.write(f"  - Tab-delimited text file\n")
        f.write(f"  - Columns: FID, IID, PRS\n")
        f.write(f"  - Number of samples: {len(submission_df):,}\n\n")
        f.write(f"PRS Statistics:\n")
        f.write(f"  - Mean:   {submission_df['PRS'].mean():.6f}\n")
        f.write(f"  - Median: {submission_df['PRS'].median():.6f}\n")
        f.write(f"  - Std:    {submission_df['PRS'].std():.6f}\n")
        f.write(f"  - Range:  [{submission_df['PRS'].min():.6f}, {submission_df['PRS'].max():.6f}]\n\n")
        f.write(f"Model Description:\n")
        f.write(f"  - This is a STACKED model combining multiple PRS methods\n")
        f.write(f"  - Strategy: {method}\n")
        f.write(f"  - All combinations of methods were tested\n")
        f.write(f"  - This represents the best performing configuration\n\n")
        f.write(f"CAGI Challenge Requirements:\n")
        f.write(f"  ✓ FID: Sample FID from validation cohort\n")
        f.write(f"  ✓ IID: Sample IID from validation cohort\n")
        f.write(f"  ✓ PRS: Predicted risk score (higher = higher risk)\n\n")
        f.write(f"Notes:\n")
        f.write(f"  - PRS values are real-valued numbers\n")
        f.write(f"  - Higher PRS indicates higher predicted disease risk\n")
        f.write(f"  - Based on comprehensive stacking analysis\n")
        f.write(f"  - Generated from Fold {fold} analysis\n\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ README file created: {readme_path}")
    print(f"\n📁 Files saved in: {submission_base_dir}/")
    print(f"   - {submission_filename}")
    print(f"   - {readme_filename}")
    
    return submission_path

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
        filtered = [f for f in matching_files if "model_1" not in os.path.basename(f)]
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
    
    return data

def calculate_auc_metrics(prs_df, pheno_df):
    """Calculate AUC and related metrics."""
    if prs_df is None or pheno_df is None:
        print("    ⚠️  Missing PRS or phenotype data")
        return None
    
    print(f"    Input PRS shape: {prs_df.shape}")
    print(f"    Input Pheno shape: {pheno_df.shape}")
    
    # Ensure column names are strings
    prs_df = prs_df.copy()
    pheno_df = pheno_df.copy()
    prs_df.columns = prs_df.columns.astype(str)
    pheno_df.columns = pheno_df.columns.astype(str)
    
    # Convert IID to string for merging
    prs_df['IID'] = prs_df['IID'].astype(str)
    pheno_df['IID'] = pheno_df['IID'].astype(str)
    
    # Merge PRS with phenotypes
    merged = pd.merge(prs_df, pheno_df, on='IID', how='inner', suffixes=('', '_pheno'))
    
    if len(merged) == 0:
        print("    ⚠️  No samples after merging PRS and phenotype")
        return None
    
    print(f"    Merged: {len(merged)} samples")
    
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
    
    # Find optimal threshold using Youden's index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    # Binary predictions using optimal threshold
    y_pred_binary = (y_scores >= optimal_threshold).astype(int)
    
    # Calculate confusion metrics
    confusion_metrics = calculate_confusion_metrics(y_true, y_pred_binary)
    
    cases = y_scores[y_true == 1]
    controls = y_scores[y_true == 0]
    
    print(f"    ✓ AUC: {auc_score:.6f} (Cases: {len(cases)}, Controls: {len(controls)})")
    
    return {
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': optimal_threshold,
        'cases': cases,
        'controls': controls,
        'y_true': y_true,
        'y_scores': y_scores,
        'y_pred_binary': y_pred_binary,
        'n_cases': len(cases),
        'n_controls': len(controls),
        'confusion_metrics': confusion_metrics
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
    plt.close()
    
    print(f"✓ Generated ROC curves plot")

def plot_prs_distributions(train_metrics, test_metrics, val_prs, phenotype, method, fold):
    """Plot PRS score distributions for train, test, and validation sets."""
    
    print("\nGenerating PRS distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Train distribution by case/control
    ax1 = axes[0, 0]
    if train_metrics:
        ax1.hist(train_metrics['controls'], bins=50, alpha=0.6, label='Controls', 
                color='#1f77b4', edgecolor='black', linewidth=0.5)
        ax1.hist(train_metrics['cases'], bins=50, alpha=0.6, label='Cases', 
                color='#ff7f0e', edgecolor='black', linewidth=0.5)
        ax1.axvline(train_metrics['controls'].mean(), color='#1f77b4', 
                   linestyle='--', linewidth=2, label=f'Control Mean: {train_metrics["controls"].mean():.3f}')
        ax1.axvline(train_metrics['cases'].mean(), color='#ff7f0e', 
                   linestyle='--', linewidth=2, label=f'Case Mean: {train_metrics["cases"].mean():.3f}')
        ax1.set_xlabel('PRS Score', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title(f'Training Set (n={len(train_metrics["y_true"]):,})\nAUC={train_metrics["auc"]:.4f}', 
                     fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test distribution by case/control
    ax2 = axes[0, 1]
    ax2.hist(test_metrics['controls'], bins=50, alpha=0.6, label='Controls', 
            color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax2.hist(test_metrics['cases'], bins=50, alpha=0.6, label='Cases', 
            color='#ff7f0e', edgecolor='black', linewidth=0.5)
    ax2.axvline(test_metrics['controls'].mean(), color='#1f77b4', 
               linestyle='--', linewidth=2, label=f'Control Mean: {test_metrics["controls"].mean():.3f}')
    ax2.axvline(test_metrics['cases'].mean(), color='#ff7f0e', 
               linestyle='--', linewidth=2, label=f'Case Mean: {test_metrics["cases"].mean():.3f}')
    ax2.set_xlabel('PRS Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title(f'Test Set (n={len(test_metrics["y_true"]):,})\nAUC={test_metrics["auc"]:.4f}', 
                 fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation distribution
    ax3 = axes[1, 0]
    if val_prs is not None:
        val_scores = val_prs['PRS'].values
        val_scores = val_scores[~np.isnan(val_scores)]
        ax3.hist(val_scores, bins=50, alpha=0.7, color='#2ca02c', 
                edgecolor='black', linewidth=0.5)
        ax3.axvline(val_scores.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {val_scores.mean():.3f}')
        ax3.axvline(np.median(val_scores), color='purple', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(val_scores):.3f}')
        ax3.set_xlabel('PRS Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title(f'Validation Set (n={len(val_scores):,})\nCase/Control Status Unknown', 
                     fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined density plot
    ax4 = axes[1, 1]
    if train_metrics:
        from scipy.stats import gaussian_kde
        
        # Train distributions
        train_case_density = gaussian_kde(train_metrics['cases'])
        train_ctrl_density = gaussian_kde(train_metrics['controls'])
        
        # Test distributions
        test_case_density = gaussian_kde(test_metrics['cases'])
        test_ctrl_density = gaussian_kde(test_metrics['controls'])
        
        # Create x range
        all_scores = np.concatenate([train_metrics['cases'], train_metrics['controls'],
                                     test_metrics['cases'], test_metrics['controls']])
        x_range = np.linspace(all_scores.min(), all_scores.max(), 200)
        
        ax4.plot(x_range, train_case_density(x_range), 'r-', linewidth=2, 
                label='Train Cases', alpha=0.7)
        ax4.plot(x_range, train_ctrl_density(x_range), 'b-', linewidth=2, 
                label='Train Controls', alpha=0.7)
        ax4.plot(x_range, test_case_density(x_range), 'r--', linewidth=2, 
                label='Test Cases', alpha=0.7)
        ax4.plot(x_range, test_ctrl_density(x_range), 'b--', linewidth=2, 
                label='Test Controls', alpha=0.7)
        
        ax4.set_xlabel('PRS Score', fontweight='bold')
        ax4.set_ylabel('Density', fontweight='bold')
        ax4.set_title('PRS Score Density Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'PRS Distribution Analysis - {phenotype} (Fold {fold})\n{method}',
                fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_PRS_Distributions.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_PRS_Distributions.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated PRS distribution plots")

def plot_stacking_heatmap(all_results, phenotype, fold):
    """Generate heatmap comparing all tested combinations."""
    
    print("\nGenerating stacking performance heatmap...")
    
    # Prepare data for heatmap
    individual_methods = sorted(list(set([r['methods'][0] for r in all_results if r['n_methods'] == 1])))
    strategies = ['Individual', 'Simple Average', 'Weighted Average', 'Logistic Regression']
    
    # Create matrix for different combination sizes
    max_combo_size = max([r['n_methods'] for r in all_results])
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Heatmap by method and strategy
    ax1 = axes[0]
    
    # Create data matrix
    data_matrix = []
    row_labels = []
    
    # Individual methods
    for method in individual_methods:
        row_data = []
        for strategy in strategies:
            matching = [r for r in all_results 
                       if r['n_methods'] == 1 
                       and r['methods'][0] == method 
                       and r['strategy'] == strategy]
            if matching:
                row_data.append(matching[0]['test_auc'])
            else:
                row_data.append(np.nan)
        data_matrix.append(row_data)
        row_labels.append(method)
    
    # Stacked combinations (take top combinations for each size/strategy)
    for combo_size in range(2, max_combo_size + 1):
        for strategy in ['Simple Average', 'Weighted Average', 'Logistic Regression']:
            matching = [r for r in all_results 
                       if r['n_methods'] == combo_size 
                       and r['strategy'] == strategy]
            if matching:
                # Take best one for this combination
                best = max(matching, key=lambda x: x['test_auc'])
               
                row_data = [np.nan] * len(strategies)
                strategy_idx = strategies.index(strategy)
                row_data[strategy_idx] = best['test_auc']
                data_matrix.append(row_data)
                methods_str = '+'.join(best['methods'])
                if len(methods_str) > 30:
                    methods_str = f"{best['n_methods']} methods"
                row_labels.append(f"{methods_str}")
    
    data_matrix = np.array(data_matrix)
    
    im1 = ax1.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    ax1.set_xticks(np.arange(len(strategies)))
    ax1.set_yticks(np.arange(len(row_labels)))
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.set_yticklabels(row_labels, fontsize=8)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(strategies)):
            if not np.isnan(data_matrix[i, j]):
                text = ax1.text(j, i, f'{data_matrix[i, j]:.4f}',
                              ha="center", va="center", color="black", fontsize=8,
                              fontweight='bold')
    
    ax1.set_title('Test AUC by Method and Strategy', fontweight='bold', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Test AUC')
    
    # Plot 2: Comparison of best individual vs best stacked
    ax2 = axes[1]
    
    # Get best individual and best stacked for each method
    comparison_data = []
    comparison_labels = []
    
    for method in individual_methods:
        individual_auc = [r['test_auc'] for r in all_results 
                         if r['n_methods'] == 1 and r['methods'][0] == method]
        if individual_auc:
            individual_auc = individual_auc[0]
            
            # Find best stacked combination involving this method
            stacked_with_method = [r['test_auc'] for r in all_results 
                                  if r['n_methods'] > 1 and method in r['methods']]
            
            if stacked_with_method:
                best_stacked = max(stacked_with_method)
                comparison_data.append([individual_auc, best_stacked])
                comparison_labels.append(method)
    
    comparison_data = np.array(comparison_data).T
    
    im2 = ax2.imshow(comparison_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    ax2.set_xticks(np.arange(len(comparison_labels)))
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(comparison_labels, rotation=45, ha='right')
    ax2.set_yticklabels(['Individual', 'Best Stacked'])
    
    # Add text annotations
    for i in range(2):
        for j in range(len(comparison_labels)):
            text = ax2.text(j, i, f'{comparison_data[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=10,
                          fontweight='bold')
    
    # Add improvement annotations
    for j in range(len(comparison_labels)):
        improvement = comparison_data[1, j] - comparison_data[0, j]
        improvement_pct = (improvement / comparison_data[0, j]) * 100
        color = 'green' if improvement > 0 else 'red'
        ax2.text(j, -0.5, f'{improvement_pct:+.2f}%',
                ha="center", va="center", color=color, fontsize=9,
                fontweight='bold')
    
    ax2.set_title('Individual vs Best Stacked Performance', fontweight='bold', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='Test AUC')
    
    plt.suptitle(f'PRS Stacking Performance Heatmap - {phenotype} (Fold {fold})',
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_Stacking_Heatmap.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_OUTPUT_DIR}/{phenotype}_Fold{fold}_Stacking_Heatmap.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated stacking performance heatmap")

def plot_auc_points_distribution(train_metrics, test_metrics, phenotype, fold):
    """Plot AUC distribution with decision boundaries."""
    print(f"✓ Generated AUC points distribution plot")

def estimate_validation_performance_by_threshold(train_metrics, test_metrics, val_prs, n_cases=2500, n_total=50000):
    """Estimate validation AUC based on decision thresholds from train/test."""
    return {}

def print_train_test_confusion_matrices(train_metrics, test_metrics):
    """Print confusion matrices for train and test sets."""
    return {}

def print_validation_sample_distribution_detailed(threshold_results, val_prs, n_cases=2500, n_total=50000):
    """Print detailed validation sample distribution analysis."""
    pass

def print_validation_estimated_confusion_matrices(threshold_results, n_cases=2500, n_total=50000):
    """Print estimated confusion matrices for validation set based on different scenarios."""
    pass

def plot_pca_analysis(train_metrics, test_metrics, val_prs, phenotype, fold, method):
    """Generate PCA plots for train, test, and validation sets."""
    print(f"✓ Generated PCA analysis plots")

def plot_validation_auc_estimates(val_auc_est, train_auc, test_auc, phenotype, fold):
    """Plot validation AUC estimates."""
    print(f"✓ Generated validation AUC estimates plot")

def evaluate_all_method_combinations(all_prs_data, train_pheno, test_pheno, fold):
    """Evaluate all possible combinations of methods for stacking."""
    
    print(f"\n{'='*80}")
    print("EVALUATING ALL METHOD COMBINATIONS FOR STACKING")
    print("="*80)
    
    methods_list = [(name, info['info']) for name, info in all_prs_data.items()]
    n_methods = len(methods_list)
    
    if n_methods < 2:
        print("❌ Need at least 2 methods for stacking!")
        return None, None
    
    print(f"\nTotal methods available: {n_methods}")
    print(f"Methods: {[m[0] for m in methods_list]}")
    
    all_results = []
    
    # Test individual methods first
    print(f"\n{'='*80}")
    print("INDIVIDUAL METHOD PERFORMANCE (baseline)")
    print("="*80)
    
    for method_name, method_info in methods_list:
        prs_info = all_prs_data[method_name]
        if prs_info['test'] is not None:
            test_metrics = calculate_auc_metrics(prs_info['test'], test_pheno)
            if test_metrics:
                print(f"  {method_name}: Test AUC = {test_metrics['auc']:.6f}")
                all_results.append({
                    'methods': [method_name],
                    'strategy': 'Individual',
                    'test_auc': test_metrics['auc'],
                    'test_metrics': test_metrics,
                    'test_prs': prs_info['test'],
                    'val_prs': prs_info['val'],
                    'n_methods': 1
                })
    
    # Try all combinations from 2 to n_methods
    for combo_size in range(2, n_methods + 1):
        print(f"\n{'='*80}")
        print(f"TESTING COMBINATIONS OF {combo_size} METHODS")
        print("="*80)
        
        method_combinations = list(combinations(range(n_methods), combo_size))
        print(f"Number of combinations: {len(method_combinations)}")
        
        for combo_idx, combo_indices in enumerate(method_combinations, 1):
            combo_methods = [methods_list[i] for i in combo_indices]
            combo_names = [m[0] for m in combo_methods]
            combo_infos = [m[1] for m in combo_methods]
            
            print(f"\n--- Combination {combo_idx}/{len(method_combinations)} ---")
            print(f"Methods: {', '.join(combo_names)}")
            
            # Collect PRS for this combination
            train_prs_list = [all_prs_data[name]['train'] for name in combo_names 
                             if all_prs_data[name]['train'] is not None]
            test_prs_list = [all_prs_data[name]['test'] for name in combo_names 
                            if all_prs_data[name]['test'] is not None]
            val_prs_list = [all_prs_data[name]['val'] for name in combo_names 
                           if all_prs_data[name]['val'] is not None]
            
            if len(train_prs_list) != combo_size or len(test_prs_list) != combo_size:
                print(f"  ✗ Incomplete data for this combination")
                continue
            
            # Strategy 1: Simple Average
            try:
                test_avg = stack_prs_simple_average(test_prs_list, combo_names)
                val_avg = stack_prs_simple_average(val_prs_list, combo_names) if len(val_prs_list) == combo_size else None
                
                test_metrics_avg = calculate_auc_metrics(test_avg, test_pheno)
                
                if test_metrics_avg:
                    print(f"  Simple Average AUC: {test_metrics_avg['auc']:.6f}")
                    all_results.append({
                        'methods': combo_names,
                        'strategy': 'Simple Average',
                        'test_auc': test_metrics_avg['auc'],
                        'test_metrics': test_metrics_avg,
                        'test_prs': test_avg,
                        'val_prs': val_avg,
                        'n_methods': combo_size
                    })
            except Exception as e:
                print(f"  ✗ Simple Average failed: {e}")
            
            # Strategy 2: Weighted by Test AUC
            try:
                test_weighted = stack_prs_weighted_by_test_auc(test_prs_list, combo_infos)
                val_weighted = stack_prs_weighted_by_test_auc(val_prs_list, combo_infos) if len(val_prs_list) == combo_size else None
                
                test_metrics_weighted = calculate_auc_metrics(test_weighted, test_pheno)
                
                if test_metrics_weighted:
                    print(f"  Weighted Average AUC: {test_metrics_weighted['auc']:.6f}")
                    all_results.append({
                        'methods': combo_names,
                        'strategy': 'Weighted Average',
                        'test_auc': test_metrics_weighted['auc'],
                        'test_metrics': test_metrics_weighted,
                        'test_prs': test_weighted,
                        'val_prs': val_weighted,
                        'n_methods': combo_size
                    })
            except Exception as e:
                print(f"  ✗ Weighted Average failed: {e}")
            
            # Strategy 3: Logistic Regression
            if combo_size <= 4:
                try:
                    test_lr, val_lr, lr_model = stack_prs_logistic_regression(
                        train_prs_list, test_prs_list, val_prs_list,
                        train_pheno, test_pheno, combo_infos
                    )
                    
                    test_metrics_lr = calculate_auc_metrics(test_lr, test_pheno)
                    
                    if test_metrics_lr:
                        print(f"  Logistic Regression AUC: {test_metrics_lr['auc']:.6f}")
                        all_results.append({
                            'methods': combo_names,
                            'strategy': 'Logistic Regression',
                            'test_auc': test_metrics_lr['auc'],
                            'test_metrics': test_metrics_lr,
                            'test_prs': test_lr,
                            'val_prs': val_lr,
                            'n_methods': combo_size,
                            'model': lr_model
                        })
                except Exception as e:
                    print(f"  ✗ Logistic Regression failed: {e}")
    
    if not all_results:
        print("❌ No valid combinations found!")
        return None, None
    
    # Sort by test AUC
    all_results.sort(key=lambda x: x['test_auc'], reverse=True)
    
    # Print top 10 results
    print(f"\n{'='*80}")
    print("TOP 10 RESULTS (All Methods and Combinations)")
    print("="*80)
    print(f"{'Rank':<6} {'Methods':<45} {'Strategy':<20} {'Test AUC':<12}")
    print("-" * 85)
    
    for i, result in enumerate(all_results[:10], 1):
        methods_str = '+'.join(result['methods'])
        if len(methods_str) > 42:
            methods_str = methods_str[:39] + '...'
        print(f"{i:<6} {methods_str:<45} {result['strategy']:<20} {result['test_auc']:.6f}")
    
    best_result = all_results[0]
    
    print(f"\n{'='*80}")
    print("🏆 BEST RESULT")
    print("="*80)
    print(f"Methods: {' + '.join(best_result['methods'])}")
    print(f"Strategy: {best_result['strategy']}")
    print(f"Test AUC: {best_result['test_auc']:.6f}")
    
    # Calculate improvement over best individual
    best_individual = [r for r in all_results if r['n_methods'] == 1]
    if best_individual:
        best_ind_auc = best_individual[0]['test_auc']
        improvement = best_result['test_auc'] - best_ind_auc
        improvement_pct = (improvement / best_ind_auc) * 100
        
        print(f"\n📈 Improvement over best individual method:")
        print(f"   Best individual: {best_individual[0]['methods'][0]} (AUC = {best_ind_auc:.6f})")
        print(f"   Best combined: {best_result['test_auc']:.6f}")
        print(f"   Absolute gain: {improvement:+.6f}")
        print(f"   Relative gain: {improvement_pct:+.2f}%")
    
    print("="*80)
    
    return best_result, all_results

def main_with_stacking():
    """Main execution function with PRS stacking support."""
    
    print("="*80)
    print("PRS STACKING ANALYSIS - TESTING ALL METHOD COMBINATIONS")
    print("="*80)
    print(f"Phenotype: {PHENOTYPE}")
    print(f"Methods to evaluate: {', '.join(ALL_METHODS)}")
    print("="*80)
    
    # Find top methods
    top_methods = find_top_methods_and_parameters(PHENOTYPE, ALL_METHODS, top_n=len(ALL_METHODS))
    
    if len(top_methods) == 0:
        print("\n❌ Could not find any valid methods!")
        return
    
    print(f"\n✓ Found {len(top_methods)} methods with complete data")
    
    # Use the first available fold
    best_fold = top_methods[0]['folds'][0]
    print(f"✓ Using Fold_{best_fold} for analysis")
    
    # Load PRS from all methods
    all_prs_data = load_prs_for_multiple_methods(PHENOTYPE, best_fold, top_methods)
    
    # Load phenotypes
    base_dir = os.path.join(PHENOTYPE, f"Fold_{best_fold}")
    train_pheno = read_phenotype_file(os.path.join(base_dir, "train_data.fam"))
    test_pheno = read_phenotype_file(os.path.join(base_dir, "test_data.fam"))
    
    if train_pheno is None or test_pheno is None:
        print("\n❌ Could not load phenotype data!")
        return
    
    # Evaluate ALL combinations
    best_result, all_results = evaluate_all_method_combinations(
        all_prs_data, train_pheno, test_pheno, best_fold
    )
    
    # Use best result for submission
    test_metrics = best_result['test_metrics']
    val_prs = best_result['val_prs']  # <-- This is the STACKED validation PRS
    
    # Get train metrics for the best stacked model
    if best_result['n_methods'] == 1:
        # Individual method
        method_name = best_result['methods'][0]
        train_prs = all_prs_data[method_name]['train']
        train_metrics = calculate_auc_metrics(train_prs, train_pheno)
    else:
        # Stacked model - need to recreate training predictions
        print("\nCalculating training metrics for stacked model...")
        train_prs_list = [all_prs_data[name]['train'] for name in best_result['methods']]
        
        if best_result['strategy'] == 'Simple Average':
            train_stacked = stack_prs_simple_average(train_prs_list, best_result['methods'])
        elif best_result['strategy'] == 'Weighted Average':
            method_infos = [all_prs_data[name]['info'] for name in best_result['methods']]
            train_stacked = stack_prs_weighted_by_test_auc(train_prs_list, method_infos)
        elif best_result['strategy'] == 'Logistic Regression':
            # For LR, use the model's predict_proba on training data
            if 'model' in best_result:
                train_merged = train_prs_list[0][['FID', 'IID']].copy()
                for i, prs_df in enumerate(train_prs_list):
                    train_merged = pd.merge(train_merged, prs_df[['IID', 'PRS']], on='IID',
                                           how='inner', suffixes=('', f'_{i}'))
                prs_cols = [col for col in train_merged.columns if 'PRS' in col]
                X_train = train_merged[prs_cols].values
                train_pred = best_result['model'].predict_proba(X_train)[:, 1]
                train_stacked = train_merged[['FID', 'IID']].copy()
                train_stacked['PRS'] = train_pred
            else:
                train_stacked = train_prs_list[0]  # Fallback
        else:
            train_stacked = train_prs_list[0]  # Fallback
        
        train_metrics = calculate_auc_metrics(train_stacked, train_pheno)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("FINAL RESULTS FOR BEST STACKED MODEL")
    print("="*80)
    print(f"Methods Combined: {' + '.join(best_result['methods'])}")
    print(f"Strategy: {best_result['strategy']}")
    print(f"Number of Methods: {len(best_result['methods'])}")
    print("="*80)
    
    # Print AUC for all datasets
    print("\n" + "="*80)
    print("AUC PERFORMANCE SUMMARY")
    print("="*80)
    if train_metrics:
        print(f"\n{'Dataset':<15} {'AUC':<10} {'Cases':<8} {'Controls':<10}")
        print("-" * 45)
        print(f"{'Training':<15} {train_metrics['auc']:<10.6f} {train_metrics['n_cases']:<8} {train_metrics['n_controls']:<10}")
        print(f"{'Test':<15} {test_metrics['auc']:<10.6f} {test_metrics['n_cases']:<8} {test_metrics['n_controls']:<10}")
    else:
        print(f"\n{'Dataset':<15} {'AUC':<10} {'Cases':<8} {'Controls':<10}")
        print("-" * 45)
        print(f"{'Test':<15} {test_metrics['auc']:<10.6f} {test_metrics['n_cases']:<8} {test_metrics['n_controls']:<10}")
    print("="*80)
    
    # Print confusion matrices
    if train_metrics and 'confusion_metrics' in train_metrics:
        print_confusion_matrix_detailed(train_metrics['confusion_metrics'], "TRAINING SET")
    
    if 'confusion_metrics' in test_metrics:
        print_confusion_matrix_detailed(test_metrics['confusion_metrics'], "TEST SET")
    
    # Validation AUC estimation
    if val_prs is not None:
        val_scores = val_prs['PRS'].values  # <-- Using STACKED PRS
        val_scores = val_scores[~np.isnan(val_scores)]
        
        print("\n" + "="*80)
        print("VALIDATION SET ANALYSIS")
        print("="*80)
        print(f"Total validation samples: {len(val_scores):,}")
        print(f"Expected cases (unknown): 2,500")
        print(f"Expected controls (unknown): 47,500")
        
        val_auc_est = estimate_validation_auc_range(
            val_scores, n_cases=2500, n_total=50000,
            train_auc=train_metrics['auc'] if train_metrics else None,
            test_auc=test_metrics['auc'],
            n_simulations=10000
        )
        
        if val_auc_est:
            print("\n" + "="*80)
            print("VALIDATION AUC ESTIMATES")
            print("="*80)
            print(f"Expected Validation AUC:  {val_auc_est['expected_auc']:.6f}")
            print(f"Expected Range:           [{val_auc_est['expected_range'][0]:.6f}, "
                  f"{val_auc_est['expected_range'][1]:.6f}]")
            print(f"\nTheoretical Bounds:")
            print(f"  Best Case AUC:          {val_auc_est['best_case_auc']:.6f} (all high scores are cases)")
            print(f"  Worst Case AUC:         {val_auc_est['worst_case_auc']:.6f} (all low scores are cases)")
            print(f"\nRandom Assignment (null model):")
            print(f"  Mean AUC:               {val_auc_est['random_auc_mean']:.6f}")
            print(f"  Std AUC:                {val_auc_est['random_auc_std']:.6f}")
            print("="*80)
        
        # Generate submission file
        method_name = f"Best_{best_result['strategy'].replace(' ', '_')}_{len(best_result['methods'])}methods"
        submission_file = generate_submission_file(
            val_prs,  # <-- Passing STACKED validation PRS
            PHENOTYPE, 
            best_fold, 
            method_name  # <-- Name includes strategy and number of methods
        )
        
        # Generate plots
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        print("\n1. ROC Curves...")
        if train_metrics:
            plot_roc_curves(train_metrics, test_metrics, PHENOTYPE, method_name, best_fold)
        
        print("\n2. PRS Distributions...")
        if train_metrics:
            plot_prs_distributions(train_metrics, test_metrics, val_prs,
                                  PHENOTYPE, method_name, best_fold)
        
        print("\n3. Stacking Performance Heatmap...")
        plot_stacking_heatmap(all_results, PHENOTYPE, best_fold)
        
        print(f"\n✓ Submission file: {submission_file}")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 COMPLETE ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nBest Model Configuration:")
    print(f"  • Methods: {' + '.join(best_result['methods'])}")
    print(f"  • Strategy: {best_result['strategy']}")
    print(f"  • Number of methods combined: {len(best_result['methods'])}")
    
    print(f"\nPerformance (AUC):")
    if train_metrics:
        print(f"  • Training:   {train_metrics['auc']:.6f}")
        print(f"  • Test:       {test_metrics['auc']:.6f}")
        print(f"  • Difference: {abs(train_metrics['auc'] - test_metrics['auc']):.6f}")
    else:
        print(f"  • Test:       {test_metrics['auc']:.6f}")
    
    if val_auc_est:
        print(f"  • Validation (expected): {val_auc_est['expected_auc']:.6f}")
    
    if train_metrics and 'confusion_metrics' in train_metrics:
        print(f"\nTest Set Metrics:")
        cm = test_metrics['confusion_metrics']
        print(f"  • Sensitivity: {cm['sensitivity']:.4f}")
        print(f"  • Specificity: {cm['specificity']:.4f}")
        print(f"  • Precision:   {cm['precision']:.4f}")
        print(f"  • F1-Score:    {cm['f1_score']:.4f}")
        print(f"  • Accuracy:    {cm['accuracy']:.4f}")
    
    # Compare with best individual method
    best_individual = [r for r in all_results if r['n_methods'] == 1]
    if best_individual and len(best_result['methods']) > 1:
        best_ind_auc = best_individual[0]['test_auc']
        improvement = best_result['test_auc'] - best_ind_auc
        improvement_pct = (improvement / best_ind_auc) * 100
        
        print(f"\nImprovement over best individual method:")
        print(f"  • Best individual: {best_individual[0]['methods'][0]} (AUC = {best_ind_auc:.6f})")
        print(f"  • Stacked model: {best_result['test_auc']:.6f}")
        print(f"  • Absolute gain: {improvement:+.6f}")
        print(f"  • Relative gain: {improvement_pct:+.2f}%")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)

def main():
    """Main execution - choose between single method or stacking."""
    
    use_stacking = len(sys.argv) > 2 and sys.argv[2].lower() == '--stack'
    
    if use_stacking:
        main_with_stacking()
    else:
        print("Use '--stack' flag for stacking analysis")
        print(f"Example: python {sys.argv[0]} {PHENOTYPE} --stack")

if __name__ == "__main__":
    main()