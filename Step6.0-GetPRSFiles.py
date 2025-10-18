#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np
import shutil
import glob

# Define important columns to display
IMPORTANT_COLUMNS = [
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
    'ldaksubmodel', 
    'ldakmodel', 
    'ldakpower',
    'ldradius',
    'ldfilename',
    'colname',
    'gibsfraction',
    'gibsburn',
    'gibsiterations',
    'LDpred-funct-bins',
    'heritability_model',
    'unique_h2',
    'grid_pvalue',
    'burn_in', 
    'num_iter',
    'sparse',
    'temp_pvalue',
    'allow_jump_sign',
    'shrink_corr',
    'use_MLE',
    'lasso_parameters_count',
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
        if existing_columns:
            return df[existing_columns].copy()
        else:
            return pd.DataFrame()
    
    allfoldsframe_dropped = [drop_performance_columns(df) for df in allfoldsframe]
    allfoldsframe_dropped = [get_important_columns(df) for df in allfoldsframe_dropped]
    
    if any(df.empty for df in allfoldsframe_dropped):
        return []
    
    # Get common columns
    common_columns = set(allfoldsframe_dropped[0].columns)
    for df in allfoldsframe_dropped[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    common_columns = list(common_columns)
    
    if len(common_columns) == 0:
        return []
    
    allfoldsframe_dropped = [df[common_columns] for df in allfoldsframe_dropped]
    
    # Find common rows
    common_rows = allfoldsframe_dropped[0]
    for i in range(1, len(allfoldsframe_dropped)):
        common_rows = pd.merge(common_rows, allfoldsframe_dropped[i], how='inner', on=common_columns)
    
    if common_rows.empty:
        return []
    
    # Extract common rows from original dataframes
    extracted_common_rows_frames = []
    for original_df in allfoldsframe:
        extracted_common_rows = pd.merge(common_rows, original_df, how='inner', on=common_columns)
        extracted_common_rows_frames.append(extracted_common_rows)
    
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
        else:
            non_numerical_df = df[non_numerical_cols].copy()
    
    averaged_df = summed_df / len(data_frames)
    result_df = pd.concat([averaged_df, non_numerical_df], axis=1)
    
    return result_df

def check_prs_files_exist_all_folds(filedirec, result_directory, available_folds, row):
    """Check if PRS files exist in ALL folds for a given row."""
    
    pvalue = row['pvalue']
    
    if 'PRSice' in result_directory:
        # For PRSice, check if all_score files exist in all folds
        pattern = f"*{int(row['numberofpca'])}*{row['PRSice-2_Model']}*.all_score"
        
        for fold_num in available_folds:
            source_dir = os.path.join(filedirec, f"Fold_{fold_num}", result_directory)
            files = glob.glob(os.path.join(source_dir, pattern))
            if not files:
                return False
        return True
    
    else:
        # For Plink, GCTA, LDAK - check if profile files exist in all folds
        pvalue_str = str(pvalue)
        
        for fold_num in available_folds:
            source_dir = os.path.join(filedirec, f"Fold_{fold_num}", result_directory)
            all_profiles = glob.glob(os.path.join(source_dir, "*.profile"))
            
            # Check if any profile contains this pvalue
            found = False
            for profile in all_profiles:
                if pvalue_str in os.path.basename(profile):
                    found = True
                    break
            
            if not found:
                return False
        
        return True

def find_prs_files_plink_gcta_ldak(source_dir, pvalue):
    """Find PRS files for Plink/GCTA/LDAK by searching all profile files."""
    found_files = []
    
    # Get all profile files
    all_profiles = glob.glob(os.path.join(source_dir, "*.profile"))
    
    # Convert pvalue to string and check variations
    pvalue_str = str(pvalue)
    
    # Try to match files containing the pvalue
    for profile in all_profiles:
        basename = os.path.basename(profile)
        # Check if pvalue appears in filename
        if pvalue_str in basename:
            found_files.append(profile)
    
    # Filter out files with "Model_1" if multiple files exist
    if len(found_files) > 1:
        filtered_files = [f for f in found_files if "Model_1" not in os.path.basename(f)]
        if filtered_files:  # Only use filtered list if it's not empty
            found_files = filtered_files
    
    return found_files

def find_closest_pvalue_column(df, target_pvalue):
    """Find the column that most closely matches the target p-value."""
    # Look for numeric columns (excluding ID columns)
    pvalue_cols = [col for col in df.columns if col not in ['FID', 'IID']]
    
    if not pvalue_cols:
        return None
    
    # Extract numeric values from column names
    col_pvalues = []
    for col in pvalue_cols:
        try:
            pval_float = float(col)
            col_pvalues.append((col, pval_float))
        except:
            continue
    
    if not col_pvalues:
        return None
    
    # Find closest match
    target = float(target_pvalue)
    closest_col = min(col_pvalues, key=lambda x: abs(x[1] - target))
    
    return closest_col[0]

def extract_prs_column_prsice(all_score_file, pvalue):
    """Extract ONLY the specific p-value column from PRSice all_score file."""
    if not os.path.exists(all_score_file):
        return None, None


    # Read the file line by line to handle headers properly
    with open(all_score_file, 'r') as f:
        header_line = f.readline().strip()
    
    # Split header and remove 'Pt_' prefix
    original_headers = header_line.split()
    #cleaned_headers = [col.replace('Pt_', '') if col.startswith('Pt_') else col for col in original_headers]
    
    # Now read the file with the cleaned headers
    df = pd.read_csv(all_score_file, sep=r'\s+', dtype=str)
    
    # Convert numeric columns (but keep column names as strings)
    for col in df.columns:
        if col not in ['FID', 'IID']:  # Keep ID columns as strings
          
                df[col] = pd.to_numeric(df[col])
             
    # Now look for pvalue as a column name (without 'Pt_' prefix)
    target_col_name = str(pvalue)
    
    if target_col_name in df.columns:
        matching_col = target_col_name
    else:
        # If exact match not found, find closest numeric column
        matching_col = find_closest_pvalue_column(df, pvalue)
    
    if matching_col is None:
        print(f"      ⚠️  Could not find p-value column for {pvalue}")
        print(f"      Available columns: {[col for col in df.columns if col not in ['FID', 'IID']]}")
        return None, None
    
    return df, matching_col
    
  

def copy_prs_files(source_dir, dest_dir, row, result_directory, fold_num, rank=1):
    """Copy PRS files based on method type."""
    copied_files = []
    
    if 'PRSice' in result_directory:
        # For PRSice, find all .all_score files
        all_score_pattern = f"*{int(row['numberofpca'])}*{row['PRSice-2_Model']}*.all_score"
        all_score_files = glob.glob(os.path.join(source_dir, all_score_pattern))
        
        for all_score_file in all_score_files:
            basename = os.path.basename(all_score_file)
            
            # Read the all_score file and get ONLY the specific p-value column
            df, matching_col = extract_prs_column_prsice(all_score_file, row['pvalue'])
            
            if df is not None and matching_col is not None:
                # Save ONLY IID, FID, and the SPECIFIC matching PRS column
                cols_to_save = ['IID']
                if 'FID' in df.columns:
                    cols_to_save.insert(0, 'FID')
                cols_to_save.append(matching_col)
                
                output_df = df[cols_to_save]
                
                # Rename the PRS column to a standard name
                output_df = output_df.rename(columns={matching_col: 'PRS'})
                
                # Determine if it's train, test, or val and use standardized naming
                if 'Train' in basename:
                    output_file = os.path.join(dest_dir, f"PRS{rank}.train")
                elif 'Test' in basename:
                    output_file = os.path.join(dest_dir, f"PRS{rank}.test")
                elif 'Val' in basename:
                    output_file = os.path.join(dest_dir, f"PRS{rank}.val")
                else:
                    output_file = os.path.join(dest_dir, f"PRS{rank}_{basename}")
                
                output_df.to_csv(output_file, sep='\t', index=False)
                copied_files.append(os.path.basename(output_file))
                print(f"        ✓ Extracted column '{matching_col}' and saved as: {os.path.basename(output_file)}")
    
    else:
        # For Plink, GCTA, LDAK - find profile files
        pvalue = row['pvalue']
        
        # Find all matching profile files
        found_files = find_prs_files_plink_gcta_ldak(source_dir, pvalue)
        
        for file in found_files:
            basename = os.path.basename(file).lower()
            
            # Determine output filename based on content
            if 'train' in basename:
                output_filename = f"PRS{rank}.train"
            elif 'test' in basename:
                output_filename = f"PRS{rank}.test"
            elif 'val' in basename:
                output_filename = f"PRS{rank}.val"
            else:
                output_filename = f"PRS{rank}_{os.path.basename(file)}"
            
            dest_file = os.path.join(dest_dir, output_filename)
            try:
                shutil.copy2(file, dest_file)
                copied_files.append(output_filename)
                print(f"        ✓ Copied: {os.path.basename(file)} → {output_filename}")
            except Exception as e:
                print(f"        ✗ Error copying {file}: {e}")
    
    return copied_files

def process_phenotype_method(filedirec, result_directory):
    """Process a single phenotype and method combination."""
    
    print(f"\n{'='*80}")
    print(f"Processing: {filedirec} - {result_directory}")
    print(f"{'='*80}\n")
    
    # Load results from all folds
    allfoldsframe = []
    available_folds = []
    
    for fold in range(0, 5):
        file_path = os.path.join(filedirec, f"Fold_{fold}", result_directory, "Results.csv")
        
        if os.path.exists(file_path):
            try:
                temp = pd.read_csv(file_path)
                allfoldsframe.append(temp)
                available_folds.append(fold)
                print(f"Fold_{fold}: ✓ Loaded ({len(temp)} rows)")
            except Exception as e:
                print(f"Fold_{fold}: ✗ Error: {e}")
        else:
            print(f"Fold_{fold}: ✗ Not found")
    
    if len(allfoldsframe) == 0:
        print(f"\n⚠️  No data found for {filedirec} - {result_directory}")
        return
    
    print(f"\nUsing {len(available_folds)} fold(s): {available_folds}")
    
    # Find common rows across folds
    extracted_common_rows_list = find_common_rows(allfoldsframe)
    
    if len(extracted_common_rows_list) == 0:
        print(f"\n⚠️  No common rows found across folds")
        return
    
    # Average the results
    averaged_df = sum_and_average_columns(extracted_common_rows_list)
    
    if averaged_df.empty:
        print(f"\n⚠️  Averaged result is empty")
        return
    
    print(f"✓ Averaged {len(averaged_df)} common row(s) across {len(available_folds)} fold(s)\n")
    
    # Determine sorting order based on method
    # LDpred-gibbs3 uses lowest (ascending=True), others use highest (ascending=False)
    use_lowest = 'LDpred-gibbs3' in result_directory
    sort_order = use_lowest
    sort_direction = "LOWEST" if use_lowest else "HIGHEST"
    
    print(f"Sorting by {'LOWEST' if use_lowest else 'HIGHEST'} Train_pure_prs for {result_directory}")
    averaged_df_sorted = averaged_df.sort_values(by='Train_pure_prs', ascending=sort_order)
    
    print(f"{'='*60}")
    print(f"CHECKING FILE AVAILABILITY ACROSS ALL FOLDS")
    print(f"{'='*60}\n")
    
    # Check which of the top rows have files in ALL folds
    valid_rows = []
    for idx, (row_idx, row) in enumerate(averaged_df_sorted.iterrows()):
        if len(valid_rows) >= 3:
            break
        
        print(f"Checking row {idx+1} (Train PRS: {row['Train_pure_prs']:.6f})...")
        if check_prs_files_exist_all_folds(filedirec, result_directory, available_folds, row):
            print(f"  ✓ Files exist in all folds")
            valid_rows.append((row_idx, row))
        else:
            print(f"  ✗ Files missing in some folds - skipping")
    
    if len(valid_rows) == 0:
        print(f"\n⚠️  No rows with files available in all folds")
        return
    
    print(f"\n✓ Found {len(valid_rows)} row(s) with files in all folds")
    
    print(f"\n{'='*60}")
    print(f"TOP {len(valid_rows)} {sort_direction} TRAINING PERFORMANCES (with files in all folds)")
    print(f"{'='*60}")
    
    # Get columns that exist in this dataframe
    display_cols = [col for col in IMPORTANT_COLUMNS if col in averaged_df.columns]
    display_cols.extend(['Train_pure_prs', 'Test_pure_prs'])
    
    for rank, (row_idx, row) in enumerate(valid_rows, 1):
        print(f"\nRank {rank}:")
        print("-" * 60)
        print(row[display_cols].to_string())
    
    print()
    
    # Get the parameter combinations for valid rows
    param_cols = [col for col in IMPORTANT_COLUMNS if col in averaged_df.columns]
    valid_params = averaged_df.loc[[r[0] for r in valid_rows], param_cols]
    
    # Save PRS files for each fold
    print(f"\n{'='*60}")
    print(f"COPYING TOP {len(valid_rows)} PRS FILES FOR EACH FOLD")
    print(f"{'='*60}")
    
    for fold_idx, fold_num in enumerate(available_folds):
        print(f"\n  Processing Fold_{fold_num}:")
        
        source_dir = os.path.join(filedirec, f"Fold_{fold_num}", result_directory)
        dest_dir = os.path.join(filedirec, "Results", f"Fold_{fold_num}", result_directory)
        
        # Create destination directory
        os.makedirs(dest_dir, exist_ok=True)
        print(f"    Created/verified directory: {dest_dir}")
        
        fold_df = extracted_common_rows_list[fold_idx]
        
        # Merge to get the rows matching valid parameters
        valid_fold = pd.merge(valid_params, fold_df, on=param_cols, how='inner')
        
        print(f"    Found {len(valid_fold)} matching rows for this fold")
        
        # Save the parameters to CSV
        top_csv = os.path.join(dest_dir, f"Top_{len(valid_rows)}_Parameters.csv")
        valid_fold.to_csv(top_csv, index=False)
        print(f"    ✓ Saved: {top_csv}")
        
        # Copy PRS files for each valid row
        for rank, (idx, row) in enumerate(valid_fold.iterrows(), 1):
            print(f"\n    Rank {rank} (pvalue={row['pvalue']}):")
            copied = copy_prs_files(source_dir, dest_dir, row, result_directory, fold_num, rank)
            
            if not copied:
                print(f"      ⚠️  No PRS files copied (unexpected)")

def main():
    """Main execution function."""
    
    # Define phenotypes and methods
    phenotype_method_combinations = [
        (f'Phenotype_{i}', method)
        for i in range(30, 31)
        for method in ['Plink3', 'GCTA3', 'LDAK-GWAS3', 'PRSice-2-3','LDpred-gibbs3']
        
        
    ]
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        combinations_str = sys.argv[1].split(',')
        phenotype_method_combinations = [tuple(c.split(':')) for c in combinations_str]
    
    print("="*80)
    print("BEST PRS PERFORMANCE FINDER AND FILE COPIER")
    print("="*80)
    print(f"Processing {len(phenotype_method_combinations)} phenotype-method combinations")
    print("="*80)
    
    # Process all combinations
    for filedirec, result_directory in phenotype_method_combinations:
        process_phenotype_method(filedirec, result_directory)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()