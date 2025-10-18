import os
import pandas as pd
import numpy as np
import sys
import glob

filedirec = sys.argv[1]
# GWAS = filedirec + os.sep + filedirec+".gz"
# df = pd.read_csv(GWAS,compression= "gzip",sep="\s+")

# if "BETA" in df.columns.to_list():
#     # For Continous Phenotype.
#     df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]

# else:
#     df["BETA"] = np.log(df["OR"])
#     df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]



# df.to_csv(filedirec + os.sep +filedirec+".txt",sep="\t",index=False)

# df.to_csv(filedirec + os.sep +filedirec+"_Plink.txt",sep="\t",index=False)

# print(df.head().to_markdown())
# print("Length of DataFrame!",len(df))

def check_phenotype_is_binary_or_continous(filedirec):
    df = pd.read_csv(filedirec+os.sep+filedirec+'.fam',sep="\s+",header=None)
    column_values = df[5].unique()
    print(column_values)
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"

from operator import index
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

def create_directory(directory):
    """Function to create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

foldnumber = sys.argv[2]

folddirec = filedirec + os.sep + "Fold_" + foldnumber
trainfilename = "train_data"
newtrainfilename = "train_data.QC"
testfilename = "test_data"
newtestfilename = "test_data.QC"

# Validation dataset path
valfilename = "val"
valdirec = "/data/ascher02/uqmmune1/CAGI7/Data/PRS/simulated/val"

# Number of PCA to be included as a covariate.
numberofpca = ["6"]

# Clumping parameters.
clump_p1 = [1]
clump_r2 = [0.1]
clump_kb = [200]

# Pruning parameters.
p_window_size = [200]
p_slide_size = [50]
p_LD_threshold = [0.25]

# P-value thresholds
minimumpvalue = 180
numberofintervals = 5000
allpvalues = np.logspace(-minimumpvalue, 0, numberofintervals, endpoint=True)

print("Minimum P-value", allpvalues[0])
print("Maximum P-value", allpvalues[-1])
print("Number of P-value", len(allpvalues))

count = 1
with open(folddirec + os.sep + 'range_list', 'w') as file:
    for value in allpvalues:
        file.write(f'pv_{value} 0 {value}\n')
        count = count + 1

pvaluefile = folddirec + os.sep + 'range_list'

# Initializing DataFrame for results - SIMPLIFIED for pure PRS only
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                   "pvalue", "numberofpca", "Train_pure_prs", "Test_pure_prs"])


def perform_clumping_and_pruning_on_individual_data(traindirec, newtrainfilename, numberofpca, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    
    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename,
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--allow-no-sex",
        "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)

    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename,
        "--clump-p1", c1_val,
        "--extract", traindirec+os.sep+trainfilename+".prune.in",
        "--clump-r2", c2_val,
        "--clump-kb", c3_val,
        "--allow-no-sex",
        "--clump", filedirec+os.sep+filedirec+".txt",
        "--clump-snp-field", "SNP",
        "--clump-field", "P",
        "--out", traindirec+os.sep+trainfilename
    ]    
    subprocess.run(command)

    # Extract valid SNPs
    command = f"awk 'NR!=1{{print $3}}' {traindirec}{os.sep}{trainfilename}.clumped > {traindirec}{os.sep}{trainfilename}.valid.snp"
    os.system(command)
    
    command = [
        "./plink",
        "--make-bed",
        "--bfile", traindirec+os.sep+newtrainfilename,
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--allow-no-sex"
    ]
    subprocess.run(command)
    
    command = [
        "./plink",
        "--make-bed",
        "--bfile", traindirec+os.sep+testfilename,
        "--allow-no-sex",
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", traindirec+os.sep+testfilename+".clumped.pruned"
    ]
    subprocess.run(command)    


def calculate_pca_for_traindata_testdata_for_clumped_pruned_snps(traindirec, newtrainfilename, p):
    
    command = [
        "./plink",
        "--bfile", folddirec+os.sep+testfilename+".clumped.pruned",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--out", folddirec+os.sep+testfilename
    ]
    subprocess.run(command)

    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)


def cleanup_prs_files(traindirec, Name, top_n=10):
    """
    Keep only the top N PRS score files based on test AUC and delete the rest
    Cleans up train, test, and validation files
    Also updates Results.csv to only contain the top N results
    """
    print("\n" + "="*80)
    print(f"CLEANING UP: Keeping only top {top_n} PRS files")
    print("="*80)
    
    # Read results to find top performers
    results_file = traindirec+os.sep+Name+os.sep+"Results.csv"
    if not os.path.exists(results_file):
        print("Results file not found, skipping cleanup")
        return
    
    results_df = pd.read_csv(results_file)
    original_count = len(results_df)
    
    print(f"Total results in CSV: {original_count}")
    
    # Remove any duplicate p-values (keep the first occurrence)
    results_df = results_df.drop_duplicates(subset=['pvalue'], keep='first')
    print(f"After removing duplicates: {len(results_df)}")
    
    # Get top N rows based on Test_pure_prs (or fewer if not enough results)
    actual_top_n = min(top_n, len(results_df))
    top_results = results_df.nlargest(actual_top_n, 'Test_pure_prs')
    top_pvalues = top_results['pvalue'].values
    
    print(f"\nActual number of top results to keep: {actual_top_n}")
    print(f"Top {actual_top_n} p-values to keep:")
    for i, pval in enumerate(top_pvalues, 1):
        test_auc = top_results[top_results['pvalue'] == pval]['Test_pure_prs'].values[0]
        train_auc = top_results[top_results['pvalue'] == pval]['Train_pure_prs'].values[0]
        print(f"  {i}. p-value: {pval:.6e}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    # Find all .profile files
    profile_dir = traindirec+os.sep+Name
    train_profiles = glob.glob(profile_dir + os.sep + "train_data.pv_*.profile")
    test_profiles = glob.glob(profile_dir + os.sep + "test_data.pv_*.profile")
    val_profiles = glob.glob(profile_dir + os.sep + "val.pv_*.profile")
    
    print(f"\nProfile files found:")
    print(f"  Train: {len(train_profiles)}")
    print(f"  Test: {len(test_profiles)}")
    print(f"  Val: {len(val_profiles)}")
    
    all_profiles = train_profiles + test_profiles + val_profiles
    
    # Create a mapping of all existing file p-values
    existing_pvalues = {}
    for profile_file in all_profiles:
        try:
            filename = os.path.basename(profile_file)
            pval_str = filename.split('.pv_')[1].split('.profile')[0]
            pval = float(pval_str)
            existing_pvalues[profile_file] = pval
        except Exception as e:
            print(f"Warning: Could not parse {filename}: {e}")
            continue
    
    print(f"Successfully parsed {len(existing_pvalues)} profile files")
    
    # Match each top p-value to the closest file p-value
    files_to_keep = set()
    pvalues_kept = []
    
    for idx, target_pval in enumerate(top_pvalues):
        if len(existing_pvalues) == 0:
            print(f"Warning: No more profile files to match for p-value {target_pval}")
            continue
            
        # Find the closest matching file p-value
        closest_file_pval = min(existing_pvalues.values(), 
                               key=lambda x: abs(x - target_pval))
        
        # Check if the match is reasonable (within 1% relative difference)
        relative_diff = abs(closest_file_pval - target_pval) / max(abs(target_pval), 1e-300)
        if relative_diff > 0.01:
            print(f"Warning: No close match found for p-value {target_pval:.6e}")
            print(f"  Closest file p-value: {closest_file_pval:.6e} (relative diff: {relative_diff:.2e})")
            continue
        
        pvalues_kept.append(closest_file_pval)
        
        # Count how many files match this p-value
        matching_files = [f for f, pv in existing_pvalues.items() if pv == closest_file_pval]
        print(f"  Rank {idx+1}: Matched p-value {closest_file_pval:.6e} -> {len(matching_files)} files")
        
        # Keep all files (train, test, val) with this p-value
        for file_path, file_pval in existing_pvalues.items():
            if file_pval == closest_file_pval:
                files_to_keep.add(file_path)
    
    print(f"\nTotal unique p-values to keep: {len(set(pvalues_kept))}")
    print(f"Total profile files to keep: {len(files_to_keep)}")
    
    # Delete files not in the keep set
    files_kept = 0
    files_deleted = 0
    
    for profile_file in all_profiles:
        if profile_file in files_to_keep:
            files_kept += 1
        else:
            try:
                os.remove(profile_file)
                files_deleted += 1
            except Exception as e:
                print(f"Warning: Could not delete {profile_file}: {e}")
    
    # Filter and save the updated Results.csv
    # Keep only rows whose p-values match the files we kept
    def is_close_to_kept_pvalue(pval, kept_pvals, tolerance=1e-10):
        """Check if p-value is close to any kept p-value"""
        for kept_pval in kept_pvals:
            rel_diff = abs(pval - kept_pval) / max(abs(pval), abs(kept_pval), 1e-300)
            if rel_diff < tolerance:
                return True
        return False
    
    filtered_results = results_df[results_df['pvalue'].apply(
        lambda x: is_close_to_kept_pvalue(x, pvalues_kept)
    )].copy()
    
    # Sort by Test_pure_prs descending
    filtered_results = filtered_results.sort_values('Test_pure_prs', ascending=False)
    
    # Save the filtered results
    filtered_results.to_csv(results_file, index=False)
    
    print(f"\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    print(f"Profile files:")
    print(f"  Kept: {files_kept}")
    print(f"  Deleted: {files_deleted}")
    print(f"  Expected: {actual_top_n * 3} (train + test + val for {actual_top_n} p-values)")
    print(f"\nResults.csv:")
    print(f"  Original rows: {original_count}")
    print(f"  Rows kept: {len(filtered_results)}")
    print(f"  Rows deleted: {original_count - len(filtered_results)}")
    
    if files_kept != actual_top_n * 3:
        print(f"\n⚠️  WARNING: Expected {actual_top_n * 3} files but kept {files_kept}")
        print("   This may indicate missing profile files for some p-values")
    
    print("="*80)
    
def evaluate_pure_prs_performance(traindirec, newtrainfilename, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    """
    Evaluate PURE PRS performance without any model fitting
    This is what you need for CAGI submission - just the discriminative power of raw PRS scores
    """
    threshold_values = allpvalues

    # Read phenotypes for train
    tempphenotype_train = pd.read_table(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".fam", sep="\s+", header=None)
    print("\n" + "="*80)
    print("TRAIN PHENOTYPE DISTRIBUTION")
    print("="*80)
    print(tempphenotype_train[5].value_counts())
    
    phenotype_train = pd.DataFrame()
    phenotype_train["Phenotype"] = tempphenotype_train[5].values
    
    # Read phenotypes for test
    tempphenotype_test = pd.read_table(traindirec+os.sep+testfilename+".clumped.pruned"+".fam", sep="\s+", header=None)
    print("\n" + "="*80)
    print("TEST PHENOTYPE DISTRIBUTION")
    print("="*80)
    print(tempphenotype_test[5].value_counts())
    
    phenotype_test = pd.DataFrame()
    phenotype_test["Phenotype"] = tempphenotype_test[5].values
    
    # Convert phenotypes: 1 -> 0 (control), 2 -> 1 (case)
    phenotype_train["Phenotype"] = phenotype_train["Phenotype"].replace({1: 0, 2: 1}) 
    phenotype_test["Phenotype"] = phenotype_test["Phenotype"].replace({1: 0, 2: 1})
   
    print("\n" + "="*80)
    print("EVALUATING PURE PRS ACROSS ALL P-VALUE THRESHOLDS")
    print("="*80)
    
    global prs_result 
    
    for idx, i in enumerate(threshold_values):
        if idx % 100 == 0:  # Print progress every 100 thresholds
            print(f"Processing p-value threshold {idx+1}/{len(threshold_values)}: {i}")
        
        try:
            # Read PRS scores for training data
            prs_train = pd.read_table(traindirec+os.sep+Name+os.sep+"train_data.pv_"+f"{i}.profile", 
                                     sep="\s+", usecols=["FID", "IID", "SCORE"])
        except:
            continue

        prs_train['FID'] = prs_train['FID'].astype(str)
        prs_train['IID'] = prs_train['IID'].astype(str)
        
        try:
            # Read PRS scores for test data
            prs_test = pd.read_table(traindirec+os.sep+Name+os.sep+"test_data.pv_"+f"{i}.profile", 
                                    sep="\s+", usecols=["FID", "IID", "SCORE"])
        except:
            continue
            
        prs_test['FID'] = prs_test['FID'].astype(str)
        prs_test['IID'] = prs_test['IID'].astype(str)
        
        # Calculate PURE PRS AUC - no model fitting!
        try:
            train_pure_auc = roc_auc_score(phenotype_train["Phenotype"].values, prs_train['SCORE'].values)
            test_pure_auc = roc_auc_score(phenotype_test["Phenotype"].values, prs_test['SCORE'].values)
        except:
            continue
        
        # Save results
        prs_result = prs_result._append({
            "clump_p1": c1_val,
            "clump_r2": c2_val,
            "clump_kb": c3_val,
            "p_window_size": p1_val,
            "p_slide_size": p2_val,
            "p_LD_threshold": p3_val,
            "pvalue": i,
            "numberofpca": p,
            "Train_pure_prs": train_pure_auc,
            "Test_pure_prs": test_pure_auc,
        }, ignore_index=True)
        
        # Save incrementally
        prs_result.to_csv(traindirec+os.sep+Name+os.sep+"Results.csv", index=False)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PURE PRS EVALUATION SUMMARY")
    print("="*80)
    print(f"Best Test AUC: {prs_result['Test_pure_prs'].max():.4f}")
    print(f"Best Train AUC: {prs_result['Train_pure_prs'].max():.4f}")
    
    best_idx = prs_result['Test_pure_prs'].idxmax()
    best_pvalue = prs_result.loc[best_idx, 'pvalue']
    print(f"Best p-value threshold: {best_pvalue}")
    print(f"Train AUC at best p-value: {prs_result.loc[best_idx, 'Train_pure_prs']:.4f}")
    print(f"Test AUC at best p-value: {prs_result.loc[best_idx, 'Test_pure_prs']:.4f}")
    
    # Cleanup: keep only top 10 PRS files
    #cleanup_prs_files(traindirec, Name, top_n=10)
    
    return


prs_result = pd.DataFrame()

def transform_plink_data(traindirec, newtrainfilename, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    
    # Perform clumping and pruning
    perform_clumping_and_pruning_on_individual_data(traindirec, newtrainfilename, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile)
    
    #Calculate PCA
    calculate_pca_for_traindata_testdata_for_clumped_pruned_snps(traindirec, newtrainfilename, p)

    # Extract p-values from GWAS file
    os.system("awk "+"\'"+"{print $3,$8}"+"\'"+" ./"+filedirec+os.sep+filedirec+".txt >  ./"+traindirec+os.sep+"SNP.pvalue")

    # Calculate PRS scores for training data
    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--score", filedirec + os.sep + filedirec+"_Plink.txt", "3", "4", "9", "header",
        "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", traindirec+os.sep+Name+os.sep+trainfilename
    ]
    subprocess.run(command)

    # Calculate PRS scores for test data
    command = [
        "./plink",
        "--bfile", folddirec+os.sep+testfilename+".clumped.pruned",
        "--score", filedirec + os.sep + filedirec+"_Plink.txt", "3", "4", "9", "header",
        "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", folddirec+os.sep+Name+os.sep+testfilename
    ]
    subprocess.run(command)

    # Calculate PRS scores for validation data
    print("\n" + "="*80)
    print("CALCULATING PRS SCORES FOR VALIDATION SET")
    print("="*80)
    command = [
        "./plink",
        "--bfile", valdirec+os.sep+valfilename,
        "--score", filedirec + os.sep + filedirec+"_Plink.txt", "3", "4", "9", "header",
        "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", traindirec+os.sep+Name+os.sep+valfilename
    ]
    subprocess.run(command)

    # Evaluate pure PRS performance (NO MODEL FITTING)
    print("\n" + "="*80)
    print("PURE PRS EVALUATION - NO MODEL FITTING")
    print("="*80)
    evaluate_pure_prs_performance(traindirec, newtrainfilename, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile)


result_directory = "Plink3"
create_directory(folddirec+os.sep+result_directory)

for p1_val in p_window_size:
    for p2_val in p_slide_size: 
        for p3_val in p_LD_threshold:
            for c1_val in clump_p1:
                for c2_val in clump_r2:
                    for c3_val in clump_kb:
                        for p in numberofpca:
                            transform_plink_data(folddirec, newtrainfilename, p, str(p1_val), str(p2_val), str(p3_val), str(c1_val), str(c2_val), str(c3_val), result_directory, pvaluefile)

print("\n" + "="*80)
print("✅ PURE PRS EVALUATION COMPLETED")
print("="*80)