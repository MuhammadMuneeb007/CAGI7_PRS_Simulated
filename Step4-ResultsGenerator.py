#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

# Configuration
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set publication-quality matplotlib parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
plt.rcParams['ps.fonttype'] = 42

def sum_and_average_columns(data_frames):
    """Sum and average numerical columns across multiple DataFrames, and keep non-numerical columns unchanged."""
    summed_df = pd.DataFrame()
    non_numerical_df = pd.DataFrame()
    
    for df in data_frames:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        non_numerical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if summed_df.empty:
            summed_df = pd.DataFrame(0, index=range(len(df)), columns=numerical_cols)
        
        summed_df[numerical_cols] = summed_df[numerical_cols].add(df[numerical_cols], fill_value=0)
        
        if non_numerical_df.empty:
            non_numerical_df = df[non_numerical_cols]
        else:
            non_numerical_df[non_numerical_cols] = non_numerical_df[non_numerical_cols].combine_first(df[non_numerical_cols])
    
    averaged_df = summed_df / len(data_frames)
    result_df = pd.concat([averaged_df, non_numerical_df], axis=1)
    
    return result_df

def find_common_rows(allfoldsframe):
    """Find common rows across all available folds based on important columns."""
    if len(allfoldsframe) == 0:
        return []
    
    performance_columns = [
        'Train_pure_prs',
        'Test_pure_prs'
    ]
    
    important_columns = [
        'clump_p1',
        'clump_r2',
        'clump_kb',
        'p_window_size',
        'p_slide_size',
        'p_LD_threshold',
        'pvalue',
        'referencepanel',
        'PRSice-2_Model',
        'effectsizes',
        'h2model',
        'model',
        'numberofpca',
        'tempalpha',
        'l1weight',
        "ldaksubmodel", 
        "ldakmodel", 
        "ldakpower",

        
        "ldradius",
        "ldfilename",
        "colname",
        
         
        "gibsfraction",
        "gibsburn",
        "gibsiterations",
        'LDpred-funct-bins',
        "heritability_model",
        "unique_h2",
        "grid_pvalue",
        "burn_in", 
        "num_iter",
        "sparse",
        "temp_pvalue",              
        "allow_jump_sign" ,
        "shrink_corr" ,
        "use_MLE" ,
      
        "lasso_parameters_count",

    ]
    
    def drop_performance_columns(df):
        return df.drop(columns=performance_columns, errors='ignore')
    
    def get_important_columns(df):
        existing_columns = [col for col in important_columns if col in df.columns]
        if existing_columns:
            return df[existing_columns].copy()
        else:
            return pd.DataFrame()
    
    allfoldsframe_dropped = [drop_performance_columns(df) for df in allfoldsframe]
    allfoldsframe_dropped = [get_important_columns(df) for df in allfoldsframe_dropped]
    
    # Check if any DataFrame is empty
    if any(df.empty for df in allfoldsframe_dropped):
        print("Warning: One or more DataFrames have no important columns after filtering.")
        return []
    
    # Get common columns across all DataFrames
    common_columns = set(allfoldsframe_dropped[0].columns)
    for df in allfoldsframe_dropped[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    common_columns = list(common_columns)
    
    if len(common_columns) == 0:
        print("Warning: No common columns found across all folds.")
        return []
    
    print(f"Common columns for merging: {common_columns}")
    
    # Keep only common columns in all DataFrames
    allfoldsframe_dropped = [df[common_columns] for df in allfoldsframe_dropped]
    
    common_rows = allfoldsframe_dropped[0]
    for i in range(1, len(allfoldsframe_dropped)):
        next_df = allfoldsframe_dropped[i]
        unique_in_common = common_rows.shape[0]
        unique_in_next = next_df.shape[0]
        common_rows = pd.merge(common_rows, next_df, how='inner', on=common_columns)
        common_count = common_rows.shape[0]
        print(f"Iteration {i}:")
        print(f"Unique rows in current common DataFrame: {unique_in_common}")
        print(f"Unique rows in next DataFrame: {unique_in_next}")
        print(f"Common rows after merge: {common_count}\n")
    
    if common_rows.empty:
        print("Warning: No common rows found after merging all folds.")
        return []
    
    extracted_common_rows_frames = []
    for original_df in allfoldsframe:
        extracted_common_rows = pd.merge(common_rows, original_df, how='inner', on=common_columns)
        extracted_common_rows_frames.append(extracted_common_rows)
    
    for i, df in enumerate(extracted_common_rows_frames):
        print(f"DataFrame {i + 1} with extracted common rows has {df.shape[0]} rows.")
    
    return extracted_common_rows_frames

def process_phenotype_method(filedirec, result_directory):
    """Process a single phenotype and method combination by reading available fold results."""
    
    print(f"\n{'='*80}")
    print(f"Processing: {filedirec} - {result_directory}")
    print(f"{'='*80}\n")
    
    # Check and load results from all folds
    allfoldsframe = []
    available_folds = []
    missing_folds = []
    
    for loop in range(0, 5):
        file_path = filedirec + os.sep + "Fold_" + str(loop) + os.sep + result_directory + os.sep + "Results.csv"
        if os.path.exists(file_path):
            try:
                temp = pd.read_csv(file_path)
                allfoldsframe.append(temp)
                available_folds.append(loop)
                print(f"Fold_{loop}: ✓ File exists. Number of rows: {len(temp)}")
            except Exception as e:
                print(f"Fold_{loop}: ✗ Error reading file: {e}")
                missing_folds.append(loop)
        else:
            print(f"Fold_{loop}: ✗ File does not exist.")
            missing_folds.append(loop)
    
    if len(allfoldsframe) == 0:
        print(f"\n⚠️  WARNING: No valid results found for {filedirec} - {result_directory}")
        print(f"All folds are missing: {list(range(5))}")
        return None
    
    # Report fold availability
    print(f"\n📊 Fold Summary:")
    print(f"   Available folds: {available_folds} (Total: {len(available_folds)}/5)")
    if missing_folds:
        print(f"   Missing folds: {missing_folds}")
        print(f"   ⚠️  Using {len(available_folds)} fold(s) for averaging")
    
    print(f"\nEnsuring common rows across available folds...")
    
    # Find common rows and average results from available folds
    extracted_common_rows_list = find_common_rows(allfoldsframe)
    
    if len(extracted_common_rows_list) == 0 or all(df.empty for df in extracted_common_rows_list):
        print(f"\n⚠️  WARNING: No common rows found across available folds for {filedirec} - {result_directory}")
        return None
    
    divided_result = sum_and_average_columns(extracted_common_rows_list)
    
    if divided_result.empty:
        print(f"\n⚠️  WARNING: Averaged result is empty for {filedirec} - {result_directory}")
        return None
    
    print(f"\n✓ Successfully averaged results from {len(available_folds)} fold(s)")
    print(f"Final result has {len(divided_result)} row(s)")
    
    # Check if required columns exist
    required_cols = ['Train_pure_prs', 'Test_pure_prs']
    if not all(col in divided_result.columns for col in required_cols):
        print(f"\n⚠️  WARNING: Missing required performance columns for {filedirec} - {result_directory}")
        return None
    
    # 1. Report based on best training performance
    print("\n1. Reporting Based on Best Training Performance:\n")
    df = divided_result.sort_values(by='Train_pure_prs', ascending=False)
    best_train_row = df.iloc[0]
    print(best_train_row.to_markdown())
    
    # Generate Train vs Test plot
    try:
        generate_train_test_plot(divided_result, filedirec, result_directory)
    except Exception as e:
        print(f"⚠️  Warning: Could not generate train-test plot: {e}")
    
    # 2. Report generalized performance
    print("\n2. Reporting Generalized Performance:\n")
    df = divided_result.copy()
    df['Difference'] = abs(df['Train_pure_prs'] - df['Test_pure_prs'])
    df['Sum'] = df['Train_pure_prs'] + df['Test_pure_prs']
    sorted_df = df.sort_values(by=['Sum', 'Difference'], ascending=[False, True])
    general_row = sorted_df.iloc[0]
    print(general_row.to_markdown())
    
    # 3. Hyperparameter correlation analysis
    print("\n3. Reporting the correlation of hyperparameters and performance metrics:\n")
 
    
    # Prepare summary for this phenotype-method combination
    summary = {
        'Phenotype': filedirec,
        'Method': result_directory,
        'Available_Folds': len(available_folds),
        'Missing_Folds': len(missing_folds),
        'Best_Train_P_Value': best_train_row.get('pvalue', np.nan),
        'Best_Train_Train_Performance': best_train_row['Train_pure_prs'],
        'Best_Train_Test_Performance': best_train_row['Test_pure_prs'],
        'General_P_Value': general_row.get('pvalue', np.nan),
        'General_Train_Performance': general_row['Train_pure_prs'],
        'General_Test_Performance': general_row['Test_pure_prs'],
    }
    
    return summary

def generate_train_test_plot(divided_result, phenotype, method):
    """Generate Train vs Test Best Models plot with academic quality."""
    
    df = divided_result.copy()
    
    # Check if pvalue column exists
    if 'pvalue' not in df.columns:
        print("Warning: 'pvalue' column not found. Skipping train-test plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    ax.plot(df['pvalue'], df['Train_pure_prs'], 
            label='Training Performance', marker='o', color='#1f77b4', 
            linewidth=2, markersize=6, alpha=0.8)
    ax.plot(df['pvalue'], df['Test_pure_prs'], 
            label='Test Performance', marker='s', color='#ff7f0e', 
            linewidth=2, markersize=6, alpha=0.8)
    
    # Highlight best training performance
    best_index = df['Train_pure_prs'].idxmax()
    best_pvalue = df.loc[best_index, 'pvalue']
    best_train = df.loc[best_index, 'Train_pure_prs']
    best_test = df.loc[best_index, 'Test_pure_prs']
    
    ax.scatter(best_pvalue, best_train, color='#d62728', s=150, 
               label='Best Training Model', edgecolor='black', zorder=5, linewidth=1.5)
    ax.scatter(best_pvalue, best_test, color='#9467bd', s=150, 
               label='Best Test Performance', edgecolor='black', zorder=5, linewidth=1.5)
    
    # Format pvalue appropriately based on type
    if pd.api.types.is_numeric_dtype(df['pvalue']):
        pvalue_str = f'p={best_pvalue:.4g}'
    else:
        pvalue_str = f'p={best_pvalue}'
    
    # Add annotations with better positioning
    ax.annotate(f'{pvalue_str}\nTrain={best_train:.4f}', 
                xy=(best_pvalue, best_train),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='#d62728', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#d62728', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))
    
    ax.annotate(f'{pvalue_str}\nTest={best_test:.4f}', 
                xy=(best_pvalue, best_test),
                xytext=(10, -20), textcoords='offset points',
                fontsize=9, color='#9467bd', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#9467bd', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5))
    
    # Highlight general performance
    df['Difference'] = abs(df['Train_pure_prs'] - df['Test_pure_prs'])
    df['Sum'] = df['Train_pure_prs'] + df['Test_pure_prs']
    sorted_df = df.sort_values(by=['Sum', 'Difference'], ascending=[False, True])
    
    general_index = sorted_df.index[0]
    general_pvalue = sorted_df.loc[general_index, 'pvalue']
    general_train = sorted_df.loc[general_index, 'Train_pure_prs']
    general_test = sorted_df.loc[general_index, 'Test_pure_prs']
    
    ax.scatter(general_pvalue, general_train, color='#2ca02c', s=200, 
               marker='D', label='Generalized Performance', 
               edgecolor='black', zorder=6, linewidth=1.5)
    
    # Format labels and title
    ax.set_xlabel('P-value Threshold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Model Performance (R²)', fontweight='bold', fontsize=12)
    ax.set_title(f'Training vs Test Performance\n{phenotype} - {method}', 
                fontweight='bold', fontsize=14, pad=15)
    
    # Improve legend
    ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, 
             framealpha=0.95, edgecolor='black', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(f"{RESULTS_DIR}/{phenotype}_{method}_Train_vs_Test_Best_Models.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/{phenotype}_{method}_Train_vs_Test_Best_Models.pdf", 
                bbox_inches='tight', facecolor='white')
    plt.close()

def generate_correlation_plots(divided_result, phenotype, method):
    """Generate correlation plots for hyperparameters vs performance metrics."""
    
    df = divided_result.copy()
    
    # Performance metrics to correlate with
    performance_metrics = [
        'Train_pure_prs',
        'Test_pure_prs'
    ]
    
    # Get all hyperparameter columns (excluding performance metrics)
    hyperparameter_cols = [col for col in df.columns if col not in performance_metrics]
    
    # Filter hyperparameters with more than one unique value
    variable_hyperparameters = []
    for col in hyperparameter_cols:
        if df[col].nunique() > 1:
            variable_hyperparameters.append(col)
    
    if len(variable_hyperparameters) == 0:
        print("No variable hyperparameters found for correlation analysis.")
        return
    
    # Prepare data for correlation: one-hot encode string columns
    df_encoded = df.copy()
    for col in variable_hyperparameters:
        if df_encoded[col].dtype == 'object':
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
    
    # Select only numeric columns for correlation
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    correlation_data = df_encoded[numeric_cols]
    
    # Calculate correlation matrix
    corr_matrix = correlation_data.corr()
    
    # Extract correlations with performance metrics
    performance_correlations = corr_matrix.loc[performance_metrics, :].T
    performance_correlations = performance_correlations.loc[
        ~performance_correlations.index.isin(performance_metrics)
    ]
    
    # Generate heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(12, max(8, len(performance_correlations) * 0.3)))
    
    im = ax.imshow(performance_correlations.values, cmap='coolwarm', 
                   aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(performance_correlations.columns)))
    ax.set_yticks(np.arange(len(performance_correlations.index)))
    
    # Set labels
    ax.set_xticklabels(performance_correlations.columns, fontsize=10, fontweight='bold')
    ax.set_yticklabels(performance_correlations.index, fontsize=9)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(performance_correlations.index)):
        for j in range(len(performance_correlations.columns)):
            value = performance_correlations.values[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}',
                   ha="center", va="center", color=text_color, 
                   fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Labels and title
    ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Hyperparameters', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Hyperparameter-Performance Correlation\n{phenotype} - {method}', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add grid
    ax.set_xticks(np.arange(len(performance_correlations.columns)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(performance_correlations.index)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{phenotype}_{method}_Hyperparameter_Correlation_Heatmap.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/{phenotype}_{method}_Hyperparameter_Correlation_Heatmap.pdf", 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Generated correlation heatmap for {phenotype} - {method}")

def generate_best_prs_plot(summary_df):
    """Generate a plot showing the best PRS performance for each phenotype."""
    
    print("\n" + "="*60)
    print("Generating Best PRS Models Plot...")
    print("="*60)
    
    # Get best model for each phenotype based on test performance
    best_models = summary_df.loc[summary_df.groupby('Phenotype')['Best_Train_Test_Performance'].idxmax()]
    
    # Sort by phenotype naturally
    best_models = best_models.sort_values('Phenotype', key=lambda x: x.apply(natural_sort_key))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Define colors for methods
    methods = best_models['Method'].unique()
    color_map = {method: plt.cm.tab10(i % 10) for i, method in enumerate(methods)}
    colors = [color_map[method] for method in best_models['Method']]
    
    x_pos = np.arange(len(best_models))
    
    # Plot 1: Test Performance
    bars1 = ax1.bar(x_pos, best_models['Best_Train_Test_Performance'], 
                    color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Phenotype', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Test Performance (R²)', fontweight='bold', fontsize=14)
    ax1.set_title('Best PRS Model Performance per Phenotype', 
                  fontweight='bold', fontsize=16, pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(best_models['Phenotype'], rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, method) in enumerate(zip(bars1, best_models['Method'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n{method}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add legend for methods
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[method], edgecolor='black', 
                            label=method) for method in methods]
    ax1.legend(handles=legend_elements, title='Best Method', 
              fontsize=10, title_fontsize=11, loc='upper right',
              frameon=True, shadow=True, fancybox=True, 
              framealpha=0.95, edgecolor='black')
    
    # Improve spines
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Plot 2: Train vs Test comparison
    width = 0.35
    bars2 = ax2.bar(x_pos - width/2, best_models['Best_Train_Train_Performance'], 
                    width, label='Training', color='#1f77b4', alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x_pos + width/2, best_models['Best_Train_Test_Performance'], 
                    width, label='Test', color='#ff7f0e', alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Phenotype', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Performance (R²)', fontweight='bold', fontsize=14)
    ax2.set_title('Best PRS Model: Training vs Test Performance', 
                  fontweight='bold', fontsize=16, pad=20)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(best_models['Phenotype'], rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=12, frameon=True, shadow=True, fancybox=True, 
              framealpha=0.95, edgecolor='black')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Improve spines
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/Best_PRS_Models_Performance.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/Best_PRS_Models_Performance.pdf", 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated Best PRS Models Performance plot")

def generate_circular_barplot(summary_df):
    """Generate a circular barplot showing average test performance by method."""
    
    print("\n" + "="*60)
    print("Generating Circular Barplot...")
    print("="*60)
    
    # Calculate average test performance per method
    method_performance = summary_df.groupby('Method').agg({
        'Best_Train_Test_Performance': 'mean',
        'General_Test_Performance': 'mean'
    }).reset_index()
    
    # Sort by performance
    method_performance = method_performance.sort_values('Best_Train_Test_Performance', ascending=False)
    
    # Create figure with polar projection
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='polar')
    
    # Number of methods
    N = len(method_performance)
    
    # Compute the angle for each bar
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    
    # Width of each bar
    width = 2 * np.pi / N * 0.8
    
    # Heights (performances)
    heights_best = method_performance['Best_Train_Test_Performance'].values
    heights_general = method_performance['General_Test_Performance'].values
    
    # Create color gradient
    colors_best = plt.cm.viridis(heights_best / heights_best.max())
    colors_general = plt.cm.plasma(heights_general / heights_general.max())
    
    # Plot bars - Best Training Model
    bars_best = ax.bar(theta, heights_best, width=width, bottom=0.0, 
                       alpha=0.85, edgecolor='black', linewidth=2,
                       color=colors_best, label='Best Training Model')
    
    # Plot bars - Generalized Model (stacked)
    bars_general = ax.bar(theta, heights_general, width=width, bottom=heights_best,
                          alpha=0.7, edgecolor='black', linewidth=1.5,
                          color=colors_general, label='Generalized Model')
    
    # Add value labels
    for angle, height_b, height_g, method in zip(theta, heights_best, heights_general, 
                                                   method_performance['Method']):
        rotation = np.rad2deg(angle)
        alignment = "left"
        
        # Adjust rotation for readability
        if 90 < rotation < 270:
            rotation = rotation + 180
            alignment = "right"
        
        # Label at the top of stacked bars
        total_height = height_b + height_g
        ax.text(angle, total_height + 0.02, f'{total_height:.4f}',
                ha=alignment, va='bottom', rotation=rotation, 
                fontsize=10, fontweight='bold', color='black')
    
    # Set labels for each bar
    ax.set_xticks(theta)
    ax.set_xticklabels(method_performance['Method'], fontsize=12, fontweight='bold')
    
    # Configure radial axis
    ax.set_ylim(0, (heights_best + heights_general).max() * 1.15)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Title and legend
    ax.set_title('Average Test Performance by Method\n(Circular Barplot)', 
                fontweight='bold', fontsize=18, pad=40, y=1.08)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
             fontsize=12, frameon=True, shadow=True, fancybox=True,
             framealpha=0.95, edgecolor='black')
    
    # Add center annotation
    center_text = f"Average\nTest Performance\nAcross Methods"
    ax.text(0, 0, center_text, ha='center', va='center', 
           fontsize=13, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.8', fc='white', ec='black', 
                    alpha=0.9, linewidth=2))
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/Circular_Barplot_Method_Performance.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/Circular_Barplot_Method_Performance.pdf", 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated Circular Barplot")
    
    # Also create a second circular plot for individual phenotypes
    print("\nGenerating Circular Barplot for Phenotypes...")
    
    # Get best performance for each phenotype
    phenotype_performance = summary_df.loc[
        summary_df.groupby('Phenotype')['Best_Train_Test_Performance'].idxmax()
    ].sort_values('Phenotype', key=lambda x: x.apply(natural_sort_key))
    
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='polar')
    
    N = len(phenotype_performance)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N * 0.9
    
    heights = phenotype_performance['Best_Train_Test_Performance'].values
    colors = plt.cm.viridis(heights / heights.max())
    
    bars = ax.bar(theta, heights, width=width, bottom=0.0,
                  alpha=0.85, edgecolor='black', linewidth=2, color=colors)
    
    # Add labels
    for angle, height, phenotype, method in zip(theta, heights, 
                                                  phenotype_performance['Phenotype'],
                                                  phenotype_performance['Method']):
        rotation = np.rad2deg(angle)
        alignment = "left"
        
        if 90 < rotation < 270:
            rotation = rotation + 180
            alignment = "right"
        
        label_text = f'{phenotype}\n{method}\n{height:.4f}'
        ax.text(angle, height + 0.01, label_text,
                ha=alignment, va='bottom', rotation=rotation,
                fontsize=8, fontweight='bold')
    
    ax.set_xticks(theta)
    ax.set_xticklabels([])  # Hide default labels as we have custom ones
    ax.set_ylim(0, heights.max() * 1.2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    ax.set_title('Best PRS Model Performance per Phenotype\n(Circular Barplot)', 
                fontweight='bold', fontsize=18, pad=40, y=1.08)
    
    center_text = f"Best Model\nper Phenotype\n(N={N})"
    ax.text(0, 0, center_text, ha='center', va='center',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=1', fc='white', ec='black',
                    alpha=0.9, linewidth=2))
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/Circular_Barplot_Phenotype_Performance.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/Circular_Barplot_Phenotype_Performance.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated Circular Barplot for Phenotypes")

def natural_sort_key(phenotype_name):
    """Extract number from phenotype name for natural sorting."""
    import re
    numbers = re.findall(r'\d+', str(phenotype_name))
    return int(numbers[0]) if numbers else 0

def generate_performance_heatmap(summary_df, metric_name='Best_Train_Test_Performance'):
    """Generate academic-quality heatmap using matplotlib."""
    
    print(f"\n{'='*60}")
    print(f"Creating heatmap for: {metric_name}")
    print("="*60)
    
    # Create pivot table
    pivot_data = summary_df.pivot(index='Phenotype', columns='Method', values=metric_name)
    
    # Check if we have any data
    if pivot_data.empty or pivot_data.isna().all().all():
        print(f"Warning: No valid data found for metric '{metric_name}'. Skipping heatmap generation.")
        return
    
    # Sort phenotypes naturally
    sorted_phenotypes = sorted(pivot_data.index.tolist(), key=natural_sort_key)
    pivot_data = pivot_data.reindex(sorted_phenotypes)
    
    # Sort methods alphabetically
    pivot_data = pivot_data.sort_index(axis=1)
    
    print(f"Pivot shape: {pivot_data.shape}")
    print(f"Non-null cells: {pivot_data.notna().sum().sum()}/{pivot_data.size}")
    
    # Create figure
    n_rows, n_cols = pivot_data.shape
    fig_width = max(10, n_cols * 4)
    fig_height = max(12, n_rows * 0.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Get value range
    vmin = pivot_data.min().min()
    vmax = pivot_data.max().max()
    
    # Create heatmap using imshow with viridis colormap
    data = pivot_data.values
    im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    
    # Set labels
    ax.set_xticklabels(pivot_data.columns, fontsize=12, fontweight='bold')
    ax.set_yticklabels(pivot_data.index, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = data[i, j]
            if not np.isnan(value):
                # Choose text color based on background
                text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{value:.4f}',
                       ha="center", va="center", color=text_color, 
                       fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Performance Score', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Labels and title
    metric_display = metric_name.replace('_', ' ')
    ax.set_xlabel('Methods', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Phenotypes', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Performance Comparison: {metric_display}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(pivot_data.columns)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_data.index)+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout()
    
    # Save
    safe_name = metric_name.replace(' ', '_')
    png_path = f"{RESULTS_DIR}/Performance_Heatmap_{safe_name}.png"
    pdf_path = f"{RESULTS_DIR}/Performance_Heatmap_{safe_name}.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Generated: {png_path}")
    print(f"✓ Generated: {pdf_path}")

def generate_summary_visualizations(summary_df):
    """Generate academic-quality summary comparison plots."""
    
    print("\n" + "="*60)
    print("Generating summary visualizations...")
    print("="*60)
    
    # Sort phenotypes naturally
    def get_sort_key(x):
        return int(x.split('_')[1]) if '_' in x else 0
    
    phenotypes = sorted(summary_df['Phenotype'].unique(), key=get_sort_key)
    methods = sorted(summary_df['Method'].unique())
    
    # 1. Create 2x2 grid of heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(20, 24))
    axes = axes.flatten()
    
    metrics = [
        ('Best_Train_Test_Performance', 'Test Performance (Best Training Model)'),
        ('Best_Train_Train_Performance', 'Training Performance (Best Training Model)'),
        ('General_Test_Performance', 'Test Performance (Generalized)'),
        ('General_Train_Performance', 'Training Performance (Generalized)')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        pivot_data = summary_df.pivot(index='Phenotype', columns='Method', values=metric)
        sorted_phenotypes = sorted(pivot_data.index.tolist(), key=natural_sort_key)
        pivot_data = pivot_data.reindex(sorted_phenotypes)
        pivot_data = pivot_data.sort_index(axis=1)
        
        data = pivot_data.values
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        
        im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns, fontsize=11, fontweight='bold')
        ax.set_yticklabels(pivot_data.index, fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = data[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{value:.4f}',
                           ha="center", va="center", color=text_color,
                           fontsize=8, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        
        # Grid
        ax.set_xticks(np.arange(len(pivot_data.columns)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(pivot_data.index)+1)-.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Methods', fontsize=11, fontweight='bold')
        ax.set_ylabel('Phenotypes', fontsize=11, fontweight='bold')
    
    plt.suptitle('Performance Comparison Across All Metrics', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/Summary_Performance_Heatmaps.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/Summary_Performance_Heatmaps.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated summary performance heatmaps")
    
    # 2. Bar chart comparison - Fixed to handle missing data
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create a mapping of phenotype to x position
    phenotype_to_x = {pheno: i for i, pheno in enumerate(phenotypes)}
    width = 0.8 / len(methods)
    
    # Best Train Performance
    for i, method in enumerate(methods):
        method_data = summary_df[summary_df['Method'] == method].copy()
        
        # Get positions and values for this method's available phenotypes
        positions = []
        values = []
        for _, row in method_data.iterrows():
            if row['Phenotype'] in phenotype_to_x:
                x_pos = phenotype_to_x[row['Phenotype']]
                positions.append(x_pos + (i - len(methods)/2 + 0.5) * width)
                values.append(row['Best_Train_Test_Performance'])
        
        if positions:  # Only plot if we have data
            bars = axes[0].bar(positions, values, width, label=method, alpha=0.85, 
                              color=colors[i % len(colors)],
                              edgecolor='black', linewidth=1.5)
    
    axes[0].set_xlabel('Phenotype', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Test Performance (R²)', fontweight='bold', fontsize=13)
    axes[0].set_title('Test Performance - Best Training Model', fontweight='bold', fontsize=15, pad=15)
    axes[0].set_xticks(range(len(phenotypes)))
    axes[0].set_xticklabels(phenotypes, rotation=45, ha='right', fontsize=10)
    axes[0].legend(title='Method', fontsize=11, title_fontsize=12, 
                  frameon=True, shadow=True, fancybox=True, 
                  framealpha=0.95, edgecolor='black')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    axes[0].set_axisbelow(True)
    
    # Improve spines
    for spine in axes[0].spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # General Performance
    for i, method in enumerate(methods):
        method_data = summary_df[summary_df['Method'] == method].copy()
        
        # Get positions and values for this method's available phenotypes
        positions = []
        values = []
        for _, row in method_data.iterrows():
            if row['Phenotype'] in phenotype_to_x:
                x_pos = phenotype_to_x[row['Phenotype']]
                positions.append(x_pos + (i - len(methods)/2 + 0.5) * width)
                values.append(row['General_Test_Performance'])
        
        if positions:  # Only plot if we have data
            bars = axes[1].bar(positions, values, width, label=method, alpha=0.85, 
                              color=colors[i % len(colors)],
                              edgecolor='black', linewidth=1.5)
    
    axes[1].set_xlabel('Phenotype', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Test Performance (R²)', fontweight='bold', fontsize=13)
    axes[1].set_title('Test Performance - Generalized Model', fontweight='bold', fontsize=15, pad=15)
    axes[1].set_xticks(range(len(phenotypes)))
    axes[1].set_xticklabels(phenotypes, rotation=45, ha='right', fontsize=10)
    axes[1].legend(title='Method', fontsize=11, title_fontsize=12, 
                  frameon=True, shadow=True, fancybox=True, 
                  framealpha=0.95, edgecolor='black')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    axes[1].set_axisbelow(True)
    
    # Improve spines
    for spine in axes[1].spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/Summary_Performance_Barplots.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/Summary_Performance_Barplots.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated summary performance barplots")
    
    # 3. Create method comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate average performance per method
    x_pos = np.arange(len(methods))
    bar_width = 0.35
    
    # Best Train
    means1 = []
    stds1 = []
    for m in methods:
        method_vals = summary_df[summary_df['Method'] == m]['Best_Train_Test_Performance'].dropna()
        means1.append(method_vals.mean() if len(method_vals) > 0 else 0)
        stds1.append(method_vals.std() if len(method_vals) > 1 else 0)
    
    # General
    means2 = []
    stds2 = []
    for m in methods:
        method_vals = summary_df[summary_df['Method'] == m]['General_Test_Performance'].dropna()
        means2.append(method_vals.mean() if len(method_vals) > 0 else 0)
        stds2.append(method_vals.std() if len(method_vals) > 1 else 0)
    
    bars1 = ax.bar(x_pos - bar_width/2, means1, bar_width, 
                   label='Best Training Model', 
                   color='#1f77b4', alpha=0.85, 
                   edgecolor='black', linewidth=1.5,
                   yerr=stds1, capsize=5, error_kw={'linewidth': 2})
    
    bars2 = ax.bar(x_pos + bar_width/2, means2, bar_width, 
                   label='Generalized Model', 
                   color='#ff7f0e', alpha=0.85, 
                   edgecolor='black', linewidth=1.5,
                   yerr=stds2, capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Method', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Test Performance (R²)', fontweight='bold', fontsize=14)
    ax.set_title('Average Performance Comparison Across Methods', 
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, shadow=True, fancybox=True, 
             framealpha=0.95, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if there's actual data
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/Method_Comparison_Average.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{RESULTS_DIR}/Method_Comparison_Average.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated method comparison plot")
    
    # Generate Best PRS Models plot
    try:
        generate_best_prs_plot(summary_df)
    except Exception as e:
        print(f"⚠️  Warning: Could not generate Best PRS plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate Circular Barplots
    try:
        generate_circular_barplot(summary_df)
    except Exception as e:
        print(f"⚠️  Warning: Could not generate Circular Barplot: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function."""
    
    # Define phenotypes and methods
    # Format: [(phenotype_directory, method_directory), ...]
    phenotype_method_combinations = [
        (f'Phenotype_{i}', method)
        for i in range(1,31)
        #for method in ['Plink', 'PRSice-2','LDpred-inf','GCTA','LDAK-GWAS','ldpred2_lassosum2','LDpred-fast','LDpred-gibbs']
        for method in ['Plink3','GCTA3','LDAK-GWAS3','PRSice-2-3' ]
    
    ]
    
    # Override with command line arguments if provided
    # Format: python script.py "Phenotype_1:PRSice-2,Phenotype_1:Plink,Phenotype_2:PRSice-2"
    if len(sys.argv) > 1:
        combinations_str = sys.argv[1].split(',')
        phenotype_method_combinations = [tuple(c.split(':')) for c in combinations_str]
    
    print("="*80)
    print("MULTI-PHENOTYPE MULTI-METHOD ANALYSIS")
    print("="*80)
    print(f"Processing {len(phenotype_method_combinations)} phenotype-method combinations:")
    for pheno, method in phenotype_method_combinations:
        print(f"  - {pheno} / {method}")
    print(f"Output Directory: {RESULTS_DIR}/")
    print("="*80)
    
    # Process all combinations
    all_summaries = []
    failed_combinations = []
    partial_combinations = []
    
    for filedirec, result_directory in phenotype_method_combinations:
        summary = process_phenotype_method(filedirec, result_directory)
        if summary is not None:
            all_summaries.append(summary)
            # Track combinations with missing folds
            if summary['Missing_Folds'] > 0:
                partial_combinations.append((filedirec, result_directory, summary['Available_Folds'], summary['Missing_Folds']))
        else:
            failed_combinations.append((filedirec, result_directory))
    
    if len(all_summaries) == 0:
        print("\n❌ No results found for any phenotype-method combination!")
        return
    
    # Report partial and failed combinations
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"✓ Successfully processed: {len(all_summaries)} combinations")
    
    if partial_combinations:
        print(f"\n⚠️  Partial results (missing some folds): {len(partial_combinations)} combinations")
        for pheno, method, avail, miss in partial_combinations:
            print(f"   - {pheno} / {method}: {avail}/5 folds available, {miss} missing")
    
    if failed_combinations:
        print(f"\n❌ Failed to process: {len(failed_combinations)} combinations")
        for pheno, method in failed_combinations:
            print(f"   - {pheno} / {method}")
    
    print("="*80)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Reorder columns
    column_order = [
        'Phenotype', 'Method', 'Available_Folds', 'Missing_Folds',
        'Best_Train_P_Value', 'Best_Train_Train_Performance', 'Best_Train_Test_Performance',
        'General_P_Value', 'General_Train_Performance', 'General_Test_Performance'
    ]
    summary_df = summary_df[[col for col in column_order if col in summary_df.columns]]
    
    # Sort by Phenotype naturally
    summary_df['_sort_key'] = summary_df['Phenotype'].apply(natural_sort_key)
    summary_df = summary_df.sort_values(['_sort_key', 'Method']).drop('_sort_key', axis=1)
    summary_df = summary_df.reset_index(drop=True)
    
    # Save summary table
    summary_csv = f"{RESULTS_DIR}/Summary_Table.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE:")
    print("="*80)
    print(f"Successfully processed {len(all_summaries)} combinations")
    print(summary_df.to_string(index=False))
    
    # Generate comparison visualizations
    print(f"\n{'='*80}")
    print("Generating summary visualizations...")
    print("="*80)
    try:
        generate_summary_visualizations(summary_df)
    except Exception as e:
        print(f"⚠️  Warning: Could not generate summary visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate individual performance heatmaps for key metrics
    print(f"\n{'='*80}")
    print("Generating individual performance heatmaps...")
    print("="*80)
    
    metrics_to_plot = [
        'Best_Train_Test_Performance',
        'General_Test_Performance',
        'Best_Train_Train_Performance',
        'General_Train_Performance'
    ]
    
    for metric in metrics_to_plot:
        if metric in summary_df.columns:
            try:
                generate_performance_heatmap(summary_df, metric)
            except Exception as e:
                print(f"⚠️  Warning: Could not generate heatmap for {metric}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save formatted table
    summary_formatted = summary_df.copy()
    for col in summary_formatted.columns:
        if summary_formatted[col].dtype == 'float64':
            summary_formatted[col] = summary_formatted[col].apply(lambda x: f"{x:.6f}" if not pd.isna(x) else "NA")
    
    summary_formatted.to_csv(f"{RESULTS_DIR}/Summary_Table_Formatted.csv", index=False)
    
    # Generate fold availability report
    fold_report = summary_df[['Phenotype', 'Method', 'Available_Folds', 'Missing_Folds']].copy()
    fold_report.to_csv(f"{RESULTS_DIR}/Fold_Availability_Report.csv", index=False)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📊 Processing Statistics:")
    print(f"   ✓ Successfully processed: {len(all_summaries)} combinations")
    print(f"   ⚠️  Partial results: {len(partial_combinations)} combinations")
    print(f"   ❌ Failed: {len(failed_combinations)} combinations")
    print(f"   📁 Total attempted: {len(phenotype_method_combinations)} combinations")
    
    # Calculate fold statistics
    total_possible_folds = len(all_summaries) * 5
    total_available_folds = summary_df['Available_Folds'].sum()
    total_missing_folds = summary_df['Missing_Folds'].sum()
    fold_availability_rate = (total_available_folds / total_possible_folds) * 100
    
    print(f"\n📂 Fold Statistics:")
    print(f"   Total possible folds: {total_possible_folds}")
    print(f"   Available folds: {total_available_folds}")
    print(f"   Missing folds: {total_missing_folds}")
    print(f"   Fold availability rate: {fold_availability_rate:.2f}%")
    
    print(f"\n📁 Generated files in {RESULTS_DIR}/:")
    print(f"   - Summary_Table.csv")
    print(f"   - Summary_Table_Formatted.csv")
    print(f"   - Fold_Availability_Report.csv")
    print(f"   - Summary_Performance_Heatmaps.png/.pdf")
    print(f"   - Summary_Performance_Barplots.png/.pdf")
    print(f"   - Method_Comparison_Average.png/.pdf")
    print(f"   - Best_PRS_Models_Performance.png/.pdf")
    print(f"   - Circular_Barplot_Method_Performance.png/.pdf")
    print(f"   - Circular_Barplot_Phenotype_Performance.png/.pdf")
    for metric in metrics_to_plot:
        safe_metric_name = metric.replace(' ', '_')
        print(f"   - Performance_Heatmap_{safe_metric_name}.png/.pdf")
    
    print(f"\n📊 Individual files for each phenotype-method combination:")
    successful_combinations = [(s['Phenotype'], s['Method']) for s in all_summaries]
    for pheno, method in successful_combinations[:5]:  # Show first 5
        print(f"   - {pheno}_{method}_Train_vs_Test_Best_Models.png/.pdf")
        print(f"   - {pheno}_{method}_Hyperparameter_Correlation_Heatmap.png/.pdf")
    if len(successful_combinations) > 5:
        print(f"   ... and {(len(successful_combinations) - 5) * 2} more files")
    
    print("="*80)
    print("\n✓ All processing complete! Check the Results/ directory for outputs.")
    print("✓ All plots are publication-quality (300 DPI) with both PNG and PDF formats.")
    print("="*80)


if __name__ == "__main__":
    main()