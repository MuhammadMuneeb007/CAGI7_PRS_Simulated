#!/usr/bin/env python
# coding: utf-8

"""
Helper functions for PRS stacking from multiple methods.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from matplotlib.patches import Patch
import glob
from itertools import combinations

def find_top_methods_and_parameters(phenotype, methods, top_n, 
                                    find_common_rows_func, 
                                    sum_and_average_columns_func,
                                    check_prs_files_exist_all_folds_func):
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
        
        extracted_common_rows_list = find_common_rows_func(allfoldsframe)
        
        if len(extracted_common_rows_list) == 0:
            print(f"  ✗ No common rows")
            continue
        
        averaged_df = sum_and_average_columns_func(extracted_common_rows_list)
        
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
            
            if check_prs_files_exist_all_folds_func(phenotype, method, available_folds, row):
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

def load_prs_for_method(phenotype, fold, method, best_row, data_type, 
                        read_prs_file_prsice_func, 
                        read_prs_file_plink_gcta_ldak_func):
    """Load PRS scores for a specific method and data type."""
    
    method_dir = os.path.join(phenotype, f"Fold_{fold}", method)
    pvalue = best_row.get('pvalue', None)
    
    if 'PRSice' in method:
        prs_file = glob.glob(os.path.join(method_dir, f"*{data_type.capitalize()}*.all_score"))
        if prs_file:
            return read_prs_file_prsice_func(prs_file[0], pvalue)
    else:
        return read_prs_file_plink_gcta_ldak_func(method_dir, pvalue, data_type)
    
    return None

def load_prs_for_multiple_methods(phenotype, fold, methods_list, 
                                   read_prs_file_prsice_func,
                                   read_prs_file_plink_gcta_ldak_func):
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
        train_prs = load_prs_for_method(phenotype, fold, method, best_row, 'train',
                                       read_prs_file_prsice_func,
                                       read_prs_file_plink_gcta_ldak_func)
        test_prs = load_prs_for_method(phenotype, fold, method, best_row, 'test',
                                      read_prs_file_prsice_func,
                                      read_prs_file_plink_gcta_ldak_func)
        val_prs = load_prs_for_method(phenotype, fold, method, best_row, 'val',
                                     read_prs_file_prsice_func,
                                     read_prs_file_plink_gcta_ldak_func)
        
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

def evaluate_stacking_strategies(all_prs_data, train_pheno, test_pheno, fold, 
                                 calculate_auc_metrics_func):
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
    
    test_metrics_avg = calculate_auc_metrics_func(test_avg, test_pheno)
    
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
    
    test_metrics_weighted = calculate_auc_metrics_func(test_weighted, test_pheno)
    
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
        
        test_metrics_lr = calculate_auc_metrics_func(test_lr, test_pheno)
        
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
            test_metrics = calculate_auc_metrics_func(prs_info['test'], test_pheno)
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

def evaluate_all_method_combinations(all_prs_data, train_pheno, test_pheno, fold, 
                                     calculate_auc_metrics_func):
    """Evaluate all possible combinations of methods for stacking."""
    
    print(f"\n{'='*80}")
    print("EVALUATING ALL METHOD COMBINATIONS FOR STACKING")
    print("="*80)
    
    methods_list = [info['info'] for method, info in all_prs_data.items()]
    n_methods = len(methods_list)
    
    if n_methods < 2:
        print("❌ Need at least 2 methods for stacking!")
        return None, None
    
    print(f"\nTotal methods available: {n_methods}")
    print(f"Methods: {[m['method'] for m in methods_list]}")
    
    all_combination_results = {}
    
    # Try all combinations from 2 to n_methods
    for combo_size in range(2, n_methods + 1):
        print(f"\n{'='*80}")
        print(f"TESTING COMBINATIONS OF {combo_size} METHODS")
        print("="*80)
        
        method_combinations = list(combinations(range(n_methods), combo_size))
        print(f"Number of combinations: {len(method_combinations)}")
        
        for combo_idx, combo_indices in enumerate(method_combinations, 1):
            combo_methods = [methods_list[i] for i in combo_indices]
            combo_names = [m['method'] for m in combo_methods]
            
            print(f"\n--- Combination {combo_idx}/{len(method_combinations)} ---")
            print(f"Methods: {', '.join(combo_names)}")
            
            # Collect PRS for this combination
            train_prs_list = [all_prs_data[m['method']]['train'] for m in combo_methods 
                             if all_prs_data[m['method']]['train'] is not None]
            test_prs_list = [all_prs_data[m['method']]['test'] for m in combo_methods 
                            if all_prs_data[m['method']]['test'] is not None]
            val_prs_list = [all_prs_data[m['method']]['val'] for m in combo_methods 
                           if all_prs_data[m['method']]['val'] is not None]
            
            if len(train_prs_list) != combo_size or len(test_prs_list) != combo_size:
                print(f"  ✗ Incomplete data for this combination")
                continue
            
            combo_results = {}
            
            # Strategy 1: Simple Average
            try:
                test_avg = stack_prs_simple_average(test_prs_list, combo_names)
                val_avg = stack_prs_simple_average(val_prs_list, combo_names) if len(val_prs_list) == combo_size else None
                
                test_metrics_avg = calculate_auc_metrics_func(test_avg, test_pheno)
                
                if test_metrics_avg:
                    combo_results['simple_average'] = {
                        'test_auc': test_metrics_avg['auc'],
                        'test_prs': test_avg,
                        'val_prs': val_avg,
                        'test_metrics': test_metrics_avg,
                        'name': f'Simple Average ({combo_size} methods)',
                        'methods': combo_names,
                        'n_methods': combo_size
                    }
                    print(f"  Simple Average AUC: {test_metrics_avg['auc']:.6f}")
            except Exception as e:
                print(f"  ✗ Simple Average failed: {e}")
            
            # Strategy 2: Weighted by Test AUC
            try:
                test_weighted = stack_prs_weighted_by_test_auc(test_prs_list, combo_methods)
                val_weighted = stack_prs_weighted_by_test_auc(val_prs_list, combo_methods) if len(val_prs_list) == combo_size else None
                
                test_metrics_weighted = calculate_auc_metrics_func(test_weighted, test_pheno)
                
                if test_metrics_weighted:
                    combo_results['weighted_average'] = {
                        'test_auc': test_metrics_weighted['auc'],
                        'test_prs': test_weighted,
                        'val_prs': val_weighted,
                        'test_metrics': test_metrics_weighted,
                        'name': f'Weighted Average ({combo_size} methods)',
                        'methods': combo_names,
                        'n_methods': combo_size
                    }
                    print(f"  Weighted Average AUC: {test_metrics_weighted['auc']:.6f}")
            except Exception as e:
                print(f"  ✗ Weighted Average failed: {e}")
            
            # Strategy 3: Logistic Regression (only if we have enough samples)
            if combo_size <= 4:  # Avoid overfitting with too many features
                try:
                    test_lr, val_lr, lr_model = stack_prs_logistic_regression(
                        train_prs_list, test_prs_list, val_prs_list,
                        train_pheno, test_pheno, combo_methods
                    )
                    
                    test_metrics_lr = calculate_auc_metrics_func(test_lr, test_pheno)
                    
                    if test_metrics_lr:
                        combo_results['logistic_regression'] = {
                            'test_auc': test_metrics_lr['auc'],
                            'test_prs': test_lr,
                            'val_prs': val_lr,
                            'test_metrics': test_metrics_lr,
                            'model': lr_model,
                            'name': f'Logistic Regression ({combo_size} methods)',
                            'methods': combo_names,
                            'n_methods': combo_size
                        }
                        print(f"  Logistic Regression AUC: {test_metrics_lr['auc']:.6f}")
                except Exception as e:
                    print(f"  ✗ Logistic Regression failed: {e}")
            
            # Store results for this combination
            if combo_results:
                combo_key = '_'.join(sorted(combo_names))
                all_combination_results[combo_key] = combo_results
    
    # Compare with individual methods
    print(f"\n{'='*80}")
    print("INDIVIDUAL METHOD PERFORMANCE (baseline)")
    print("="*80)
    
    individual_aucs = {}
    for method_name, prs_info in all_prs_data.items():
        if prs_info['test'] is not None:
            test_metrics = calculate_auc_metrics_func(prs_info['test'], test_pheno)
            if test_metrics:
                individual_aucs[method_name] = test_metrics['auc']
                print(f"  {method_name}: Test AUC = {test_metrics['auc']:.6f}")
    
    # Find best overall combination and strategy
    print(f"\n{'='*80}")
    print("FINDING BEST STACKING COMBINATION")
    print("="*80)
    
    all_results_flat = []
    for combo_key, combo_results in all_combination_results.items():
        for strategy_name, result in combo_results.items():
            all_results_flat.append({
                'combo_key': combo_key,
                'strategy': strategy_name,
                'result': result,
                'test_auc': result['test_auc']
            })
    
    if not all_results_flat:
        print("❌ No valid stacking combinations found!")
        return None, all_combination_results
    
    # Sort by test AUC
    all_results_flat.sort(key=lambda x: x['test_auc'], reverse=True)
    
    # Print top 10 combinations
    print(f"\nTop 10 Stacking Combinations:")
    print(f"{'Rank':<6} {'Methods':<50} {'Strategy':<30} {'Test AUC':<12}")
    print("-" * 100)
    
    for i, item in enumerate(all_results_flat[:10], 1):
        methods_str = ', '.join(item['result']['methods'])
        if len(methods_str) > 47:
            methods_str = methods_str[:44] + '...'
        strategy_str = item['result']['name']
        if len(strategy_str) > 27:
            strategy_str = strategy_str[:24] + '...'
        print(f"{i:<6} {methods_str:<50} {strategy_str:<30} {item['test_auc']:.6f}")
    
    # Best combination
    best_item = all_results_flat[0]
    best_result = best_item['result']
    
    print(f"\n{'='*80}")
    print("🏆 BEST STACKING COMBINATION")
    print("="*80)
    print(f"Methods: {', '.join(best_result['methods'])}")
    print(f"Strategy: {best_result['name']}")
    print(f"Test AUC: {best_result['test_auc']:.6f}")
    
    # Calculate improvement over best individual method
    if individual_aucs:
        best_individual_auc = max(individual_aucs.values())
        best_individual_method = max(individual_aucs.items(), key=lambda x: x[1])[0]
        improvement = best_result['test_auc'] - best_individual_auc
        improvement_pct = (improvement / best_individual_auc) * 100
        
        print(f"\n📈 Improvement over best individual method:")
        print(f"   Best individual: {best_individual_method} (AUC = {best_individual_auc:.6f})")
        print(f"   Best stacked: {best_result['test_auc']:.6f}")
        print(f"   Absolute improvement: {improvement:+.6f}")
        print(f"   Relative improvement: {improvement_pct:+.2f}%")
    
    print("="*80)
    
    return best_result, all_combination_results

def plot_stacking_comparison(stacking_results, all_prs_data, test_pheno, phenotype, fold,
                            calculate_auc_metrics_func, results_output_dir):
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
            test_metrics = calculate_auc_metrics_func(prs_info['test'], test_pheno)
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
    plt.savefig(f"{results_output_dir}/{phenotype}_Fold{fold}_Stacking_Comparison.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{results_output_dir}/{phenotype}_Fold{fold}_Stacking_Comparison.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated stacking comparison plot")
