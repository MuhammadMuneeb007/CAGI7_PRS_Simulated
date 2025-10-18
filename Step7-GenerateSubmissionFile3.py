import os
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def process_all_phenotypes():
    # Create Submission3 directory in current working directory
    os.makedirs("Submission3", exist_ok=True)
    
    # Create results container
    all_results = []
    
    # Process each phenotype
    for phenotype_num in range(1, 31):
        phenotype = f"Phenotype_{phenotype_num}"
        print(f"\n{'='*80}")
        print(f"PROCESSING {phenotype}")
        print(f"{'='*80}")
        
        # Skip if the phenotype directory doesn't exist
        if not os.path.exists(phenotype):
            print(f"Directory for {phenotype} not found. Skipping.")
            continue
        
        # Find the best result for this phenotype
        best_result = find_best_result(phenotype)
        
        if best_result['test_auc'] > 0:
            # Generate submission file
            val_metrics = generate_submission_file(phenotype, best_result)
            
            # Store results for heatmap
            result_row = {
                'Phenotype': phenotype,
                'SNP_Count': best_result['snp_count'],
                'Fold': best_result['fold'],
                'Train_AUC': best_result['train_auc'],
                'Test_AUC': best_result['test_auc'],
                'AUC_Gap': best_result['train_auc'] - best_result['test_auc'],
                'Expected_Val_AUC': val_metrics.get('expected_val_auc', np.nan),
                'N_Samples': val_metrics.get('n_samples', 0),
                'Pred_Cases_Pct': val_metrics.get('pred_cases_pct', 0)
            }
            all_results.append(result_row)
    
    # Create results dataframe
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_file = os.path.join("Submission3", "All_Phenotypes_Best_Results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        
        # Create heatmap
        create_metrics_heatmap(results_df)
        
        # Create merged submission file
        create_merged_submission_file()
    else:
        print("No results found for any phenotype.")

def find_best_result(phenotype):
    print(f"Finding best result for {phenotype}...")
    
    # Find all SNP counts
    snp_dirs = glob.glob(f"{phenotype}/Fold_*/Top_*_SNPs")
    snp_counts = set()
    for dir_path in snp_dirs:
        snp_count = int(dir_path.split("Top_")[1].split("_SNPs")[0])
        snp_counts.add(snp_count)
    
    if not snp_counts:
        print(f"No SNP count directories found for {phenotype}")
        return {'test_auc': 0, 'snp_count': None, 'fold': None}
    
    print(f"Found SNP counts: {sorted(snp_counts)}")
    
    # Track overall best result
    overall_best = {
        'test_auc': 0,
        'train_auc': 0,
        'snp_count': None,
        'fold': None,
        'predictions_file': None
    }
    
    # For each SNP count, find the best test AUC across all folds
    for snp_count in sorted(snp_counts):
        for fold in range(5):  # Assuming 5 folds (0-4)
            results_file = f"{phenotype}/Fold_{fold}/Top_{snp_count}_SNPs/ensemble_results.csv"
            
            if os.path.exists(results_file):
                try:
                    df = pd.read_csv(results_file)
                    if not df.empty:
                        test_auc = df['Test_AUC'].values[0]
                        train_auc = df['Train_AUC'].values[0]
                        
                        # Check if this is the best result overall
                        if test_auc > overall_best['test_auc']:
                            predictions_file = f"{phenotype}/Fold_{fold}/Top_{snp_count}_SNPs/predictions.pkl"
                            
                            if os.path.exists(predictions_file):
                                overall_best['test_auc'] = test_auc
                                overall_best['train_auc'] = train_auc
                                overall_best['snp_count'] = snp_count
                                overall_best['fold'] = fold
                                overall_best['predictions_file'] = predictions_file
                                print(f"  New best result! Fold {fold}, SNP count {snp_count}, Test AUC: {test_auc:.6f}")
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
    
    # Print best result summary
    if overall_best['test_auc'] > 0:
        print(f"\nBEST RESULT: SNP Count = {overall_best['snp_count']}, Fold = {overall_best['fold']}")
        print(f"Test AUC: {overall_best['test_auc']:.6f}, Train AUC: {overall_best['train_auc']:.6f}")
    else:
        print(f"No valid results found for {phenotype}.")
    
    return overall_best

def generate_submission_file(phenotype, best_result):
    if best_result['test_auc'] == 0:
        return {}
        
    # Metrics to return
    val_metrics = {}
    
    # Load the predictions file
    with open(best_result['predictions_file'], 'rb') as f:
        predictions = pickle.load(f)
    
    # Get validation predictions
    val_predictions = predictions.get('Val_Predictions', None)
    
    if val_predictions is not None and len(val_predictions) > 0:
        print(f"\nValidation predictions found. Shape: {val_predictions.shape if hasattr(val_predictions, 'shape') else len(val_predictions)}")
        
        # Try to find expected validation AUC if available
        expected_val_auc = np.nan
        
        # Check if ensemble_results.csv has expected validation AUC
        results_file = f"{phenotype}/Fold_{best_result['fold']}/Top_{best_result['snp_count']}_SNPs/ensemble_results.csv"
        if os.path.exists(results_file):
            try:
                df = pd.read_csv(results_file)
                if 'Val_Expected_AUC_MonteCarlo' in df.columns:
                    expected_val_auc = df['Val_Expected_AUC_MonteCarlo'].values[0]
                elif 'Val_Expected_AUC_Threshold' in df.columns:
                    expected_val_auc = df['Val_Expected_AUC_Threshold'].values[0]
            except:
                pass
        
        val_metrics['expected_val_auc'] = expected_val_auc
        
        # Load validation IDs from val.PHENO
        val_pheno_file = f"{phenotype}/Fold_{best_result['fold']}/val.PHENO"
        
        if os.path.exists(val_pheno_file):
            print(f"Using val.PHENO file: {val_pheno_file}")
            # PHENO file format: FID IID PHENO
            try:
                val_ids = pd.read_csv(val_pheno_file, sep='\s+')
                print(f"Validation IDs from PHENO shape: {val_ids.shape}")
            except Exception as e:
                print(f"Error reading val.PHENO file: {e}")
                try:
                    # Try alternative parsing
                    val_ids = pd.read_csv(val_pheno_file, delim_whitespace=True)
                    print(f"Validation IDs from PHENO (alternate parsing) shape: {val_ids.shape}")
                except:
                    print(f"Failed to parse val.PHENO file. Trying other ID sources...")
                    val_ids = None
        else:
            print(f"No val.PHENO file found. Looking for other ID sources...")
            val_ids = None
        
        # If val.PHENO not found or failed to parse, try validation_ids.txt or val.fam
        if val_ids is None:
            val_ids_file = f"{phenotype}/Fold_{best_result['fold']}/validation_ids.txt"
            val_fam_file = f"{phenotype}/Fold_{best_result['fold']}/val.fam"
            
            if os.path.exists(val_ids_file):
                print(f"Using validation_ids.txt file: {val_ids_file}")
                val_ids = pd.read_csv(val_ids_file, sep='\t')
            elif os.path.exists(val_fam_file):
                print(f"Using val.fam file: {val_fam_file}")
                # FAM file format: FID IID PAT MAT SEX PHENO
                val_ids = pd.read_csv(val_fam_file, sep='\s+', header=None)
                if val_ids.shape[1] >= 2:
                    val_ids.columns = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'][:val_ids.shape[1]]
                    val_ids = val_ids[['FID', 'IID']]  # Keep only FID and IID columns
            
        if val_ids is None:
            print(f"No validation IDs found. Cannot create submission file.")
            return val_metrics
        
        # Check if lengths match
        if len(val_ids) != len(val_predictions):
            print(f"Warning: Length mismatch! Val IDs: {len(val_ids)}, Val predictions: {len(val_predictions)}")
            # Truncate to the smaller length
            min_len = min(len(val_ids), len(val_predictions))
            val_ids = val_ids.iloc[:min_len]
            val_predictions = val_predictions[:min_len]
            print(f"Truncated to first {min_len} entries")
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'FID': val_ids['FID'].values,
            'IID': val_ids['IID'].values,
            'PRS': val_predictions
        })
        
        # Save submission file in the Submission3 folder in current working directory
        submission_file = os.path.join("Submission3", f"{phenotype}_submission_3.txt")
        submission_df.to_csv(submission_file, sep='\t', index=False, float_format='%.10f')
        print(f"Submission file saved: {submission_file}")
        
        # Calculate submission statistics
        val_metrics['n_samples'] = len(submission_df)
        
        # Count predicted cases and controls
        n_pred_cases = int((submission_df['PRS'] >= 0.5).sum())
        pred_cases_pct = (n_pred_cases / len(submission_df) * 100) if len(submission_df) > 0 else 0
        val_metrics['pred_cases_pct'] = pred_cases_pct
        
        print("\nSubmission Statistics:")
        print(f"Total samples: {len(submission_df)}")
        print(f"PRS mean: {submission_df['PRS'].mean():.6f}")
        print(f"PRS std: {submission_df['PRS'].std():.6f}")
        print(f"PRS range: [{submission_df['PRS'].min():.6f}, {submission_df['PRS'].max():.6f}]")
        print(f"Predicted cases (≥0.5): {n_pred_cases} ({pred_cases_pct:.2f}%)")
        print(f"Predicted controls (<0.5): {len(submission_df) - n_pred_cases} ({100 - pred_cases_pct:.2f}%)")
    else:
        print(f"No validation predictions found for {phenotype}.")
    
    return val_metrics

def create_merged_submission_file():
    # Find all submission files in the Submission3 directory
    submission_files = glob.glob("Submission3/*_submission_3.txt")
    
    if not submission_files:
        print("No submission files found to merge.")
        return
    
    print(f"\nMerging {len(submission_files)} submission files...")
    
    # Create a dictionary to store all submissions
    all_submissions = {}
    
    for file in submission_files:
        phenotype = os.path.basename(file).split('_submission')[0]
        
        try:
            df = pd.read_csv(file, sep='\t')
            
            # Extract FID, IID, and PRS
            for i, row in df.iterrows():
                key = f"{row['FID']}_{row['IID']}"
                
                if key not in all_submissions:
                    all_submissions[key] = {
                        'FID': row['FID'],
                        'IID': row['IID']
                    }
                
                # Add phenotype-specific PRS
                all_submissions[key][f"{phenotype}_PRS"] = row['PRS']
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Create merged dataframe
    if all_submissions:
        merged_df = pd.DataFrame.from_dict(all_submissions, orient='index')
        merged_df = merged_df.reset_index(drop=True)
        
        # Sort columns
        id_cols = ['FID', 'IID']
        prs_cols = [col for col in merged_df.columns if col not in id_cols]
        prs_cols = sorted(prs_cols)
        
        merged_df = merged_df[id_cols + prs_cols]
        
        # Save merged file in Submission3 directory
        merged_file = os.path.join("Submission3", "Merged_PRS_Submissions.txt")
        merged_df.to_csv(merged_file, sep='\t', index=False, float_format='%.10f')
        print(f"Merged submission file saved: {merged_file}")
        
        # Print statistics
        print(f"Total samples: {len(merged_df)}")
        print(f"Total phenotypes: {len(prs_cols)}")
    else:
        print("No valid submissions to merge.")

def create_metrics_heatmap(results_df):
    # Extract phenotype numbers for sorting
    results_df['Phenotype_Num'] = results_df['Phenotype'].str.extract(r'Phenotype_(\d+)').astype(int)
    
    # Sort by phenotype number in ascending order
    results_df = results_df.sort_values('Phenotype_Num', ascending=False)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Number of phenotypes
    n_phenotypes = len(results_df)
    
    # Define the metrics to include
    metrics = ['Train_AUC', 'Test_AUC', 'Expected_Val_AUC']
    n_metrics = len(metrics)
    
    # Define a custom colormap
    cmap = plt.cm.viridis  # Reverse viridis for better visibility
    
    # Create a grid for the heatmap
    ax = plt.subplot(111)
    
    # Draw the heatmap cells one by one with annotations
    for i, (_, row) in enumerate(results_df.iterrows()):
        for j, metric in enumerate(metrics):
            value = row[metric]
            
            # Handle NaN values
            if pd.isna(value):
                color = 'lightgray'
                text = 'N/A'
            else:
                # Map value to color (AUC values typically range from 0.5 to 1)
                norm_value = (value - 0.5) / 0.5  # Normalize to [0, 1]
                norm_value = max(0, min(norm_value, 1))  # Clip to [0, 1]
                color = cmap(norm_value)
                text = f'{value:.4f}'
            
            # Draw rectangle
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white')
            ax.add_patch(rect)
            
            # Add text
            plt.text(j + 0.5, i + 0.5, text, 
                     ha='center', va='center',
                     color='black' if norm_value > 0.5 else 'white')
    
    # Set the limits and remove ticks
    plt.xlim(0, n_metrics)
    plt.ylim(0, n_phenotypes)
    plt.xticks(np.arange(n_metrics) + 0.5, metrics)
    plt.yticks(np.arange(n_phenotypes) + 0.5, results_df['Phenotype'])
    
    # Set labels and title
    plt.title('AUC Metrics Comparison for Best Models', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Phenotypes', fontsize=14)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.5, 1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('AUC Value', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('Submission3/Phenotype_AUC_Metrics_Heatmap.png', dpi=300, bbox_inches='tight')
    print("\nHeatmap saved to Submission3/Phenotype_AUC_Metrics_Heatmap.png")
    
    # Create a table with the metrics
    plt.figure(figsize=(12, n_phenotypes * 0.4))
    
    # Disable axis
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create table
    cell_text = []
    for _, row in results_df.iterrows():
        cell_text.append([
            row['Phenotype'],
            f"{row['SNP_Count']}",
            f"{row['Fold']}",
            f"{row['Train_AUC']:.4f}",
            f"{row['Test_AUC']:.4f}",
            f"{row['Expected_Val_AUC']:.4f}" if not pd.isna(row['Expected_Val_AUC']) else 'N/A',
            f"{row['AUC_Gap']:.4f}",
            f"{row['N_Samples']}",
            f"{row['Pred_Cases_Pct']:.2f}%"
        ])
    
    table = plt.table(
        cellText=cell_text,
        colLabels=['Phenotype', 'SNP Count', 'Fold', 'Train AUC', 'Test AUC', 'Expected Val AUC', 'AUC Gap', 'N Samples', 'Pred Cases %'],
        loc='center',
        cellLoc='center'
    )
    
    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Detailed Metrics for Best Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save table
    plt.savefig('Submission3/Phenotype_Metrics_Table.png', dpi=300, bbox_inches='tight')
    print("Metrics table saved to Submission3/Phenotype_Metrics_Table.png")

# Run the processing for all phenotypes
if __name__ == "__main__":
    process_all_phenotypes()