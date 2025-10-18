import os
import pandas as pd
import numpy as np
import sys
import glob

def check_phenotype_is_binary_or_continous(filedirec):
    """Check if phenotype is binary or continuous."""
    df = pd.read_csv(filedirec+os.sep+filedirec+'.fam',sep="\s+",header=None)
    column_values = df[5].unique()
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"

filedirec = sys.argv[1]

print("="*80)
print("PREPARING GWAS DATA FOR GCTA")
print("="*80)

# Read the GWAS file
GWAS = filedirec + os.sep + filedirec+".gz"
# df = pd.read_csv(GWAS,compression= "gzip",sep="\s+")

# if check_phenotype_is_binary_or_continous(filedirec)=="Binary":
#     print("\nProcessing Binary Phenotype...")
    
#     if "BETA" in df.columns.to_list():
#         # For Binary Phenotypes with BETA (already on log scale)
#         df["OR"] = np.exp(df["BETA"])
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'OR', 'INFO', 'MAF']]
#         # SE is already on log-scale, no conversion needed
#         df["SE_BETA"] = df["SE"]
#     else:
#         # For Binary Phenotype with OR (need to convert to log scale)
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'MAF']]
#         # Convert OR to BETA (log-scale)
#         df["BETA"] = np.log(df["OR"])
        
#         # FIXED: Proper SE conversion from OR-scale to log-scale
#         # Using delta method: SE(log(OR)) ≈ SE(OR) / |OR|
#         df["SE_BETA"] = df["SE"] / np.abs(df["OR"])
        
#     # Transform to GCTA format - use BETA (log-scale) NOT OR
#     df_transformed = pd.DataFrame({
#         'SNP': df['SNP'],
#         'A1': df['A1'],
#         'A2': df['A2'],
#         'freq': df['MAF'],
#         'b': df['BETA'],        # BETA on log scale
#         'se': df['SE_BETA'],    # SE on log scale
#         'p': df['P'],
#         'N': df['N']
#     })
    
#     print("\nGCTA input statistics:")
#     print(f"  BETA range: {df['BETA'].min():.6f} to {df['BETA'].max():.6f}")
#     print(f"  SE_BETA range: {df['SE_BETA'].min():.6f} to {df['SE_BETA'].max():.6f}")
        
# elif check_phenotype_is_binary_or_continous(filedirec)=="Continous":
#     print("\nProcessing Continuous Phenotype...")
#     if "BETA" in df.columns.to_list():
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
#     else:
#         df["BETA"] = np.log(df["OR"])
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    
#     df_transformed = pd.DataFrame({
#         'SNP': df['SNP'],
#         'A1': df['A1'],
#         'A2': df['A2'],
#         'freq': df['MAF'],
#         'b': df['BETA'],
#         'se': df['SE'],
#         'p': df['P'],
#         'N': df['N']
#     })

# # Save GCTA input file
# output_file = filedirec + os.sep +filedirec+"_GCTA.txt"   
# df_transformed.to_csv(output_file,sep="\t",index=False)
# print("\n" + df_transformed.head().to_markdown())
# print(f"Length of DataFrame: {len(df_transformed)}")
# print(f"✓ Saved GCTA input file: {output_file}")

# # Save standard format file with BETA for clumping
# if "BETA" not in df.columns.to_list():
#     df["BETA"] = np.log(df["OR"])
#     df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]

# df.to_csv(filedirec + os.sep +filedirec+".txt",sep="\t",index=False)
# print("\n" + df.head().to_markdown())
# print(f"Length of DataFrame: {len(df)}")
# print(f"✓ Saved standard format file: {filedirec + os.sep + filedirec}.txt")

from operator import index
import pandas as pd
import numpy as np
import os
import subprocess
import sys
from sklearn.metrics import roc_auc_score

def create_directory(directory):
    """Create directory if it doesn't exist."""
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

numberofpca = ["6"]
clump_p1 = [1]
clump_r2 = [0.1]
clump_kb = [200]
p_window_size = [200]
p_slide_size = [50]
p_LD_threshold = [0.25]

# Reasonable number of p-value thresholds for testing
minimumpvalue = 100
numberofintervals = 2000
allpvalues = np.logspace(-minimumpvalue, 0, numberofintervals, endpoint=True)

print(f"\n{'='*80}")
print(f"P-VALUE THRESHOLD SETTINGS")
print(f"{'='*80}")
print(f"Number of p-value thresholds: {len(allpvalues)}")
print(f"Min p-value: {allpvalues[0]:.2e}")
print(f"Max p-value: {allpvalues[-1]:.2e}")

count = 1
with open(folddirec + os.sep + 'range_list', 'w') as file:
    for value in allpvalues:
        file.write(f'pv_{value} 0 {value}\n')
        count = count + 1

pvaluefile = folddirec + os.sep + 'range_list'

# UPDATED: Simplified results - PURE PRS ONLY
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                   "pvalue", "h2model", "numberofpca", "h2", "gcta_lambda", "numberofvariants",
                                   "Train_pure_prs", "Test_pure_prs"])

def perform_clumping_and_pruning_on_individual_data(traindirec, newtrainfilename, numberofpca, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    
    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename,
        "--allow-no-sex",
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)

    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename,
        "--clump-p1", c1_val,
        "--allow-no-sex",
        "--extract", traindirec+os.sep+trainfilename+".prune.in",
        "--clump-r2", c2_val,
        "--clump-kb", c3_val,
        "--clump", filedirec+os.sep+filedirec+".txt",
        "--clump-snp-field", "SNP",
        "--clump-field", "P",
        "--out", traindirec+os.sep+trainfilename
    ]    
    subprocess.run(command)

    command = f"awk 'NR!=1{{print $3}}' {traindirec}{os.sep}{trainfilename}.clumped > {traindirec}{os.sep}{trainfilename}.valid.snp"
    os.system(command)

    command = [
        "./plink",
        "--make-bed",
        "--bfile", traindirec+os.sep+newtrainfilename,
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--allow-no-sex",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", traindirec+os.sep+newtrainfilename+".clumped.pruned"
    ]
    subprocess.run(command)
    
    command = [
        "./plink",
        "--make-bed",
        "--allow-no-sex",
        "--bfile", traindirec+os.sep+testfilename,
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
        "--allow-no-sex",
        "--out", folddirec+os.sep+testfilename
    ]
    subprocess.run(command)

    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--allow-no-sex",
        "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)


def cleanup_gcta_files(traindirec, Name, top_n=10):
    """
    Keep only the top N GCTA model files based on test AUC and delete the rest
    Also updates Results.csv to only contain the top N results
    """
    print("\n" + "="*80)
    print(f"CLEANING UP: Keeping only top {top_n} GCTA results")
    print("="*80)
    
    # Read results to find top performers
    results_file = traindirec+os.sep+Name+os.sep+"Results.csv"
    if not os.path.exists(results_file):
        print("Results file not found, skipping cleanup")
        return
    
    results_df = pd.read_csv(results_file)
    original_count = len(results_df)
    
    print(f"Total results in CSV: {original_count}")
    
    # Remove any duplicate rows
    results_df = results_df.drop_duplicates(
        subset=['h2model', 'pvalue', 'numberofpca', 'clump_p1', 'clump_r2', 'clump_kb',
                'p_window_size', 'p_slide_size', 'p_LD_threshold'], 
        keep='first'
    )
    print(f"After removing duplicates: {len(results_df)}")
    
    # Get top N rows based on Test_pure_prs (or fewer if not enough results)
    actual_top_n = min(top_n, len(results_df))
    top_results = results_df.nlargest(actual_top_n, 'Test_pure_prs').copy()
    
    print(f"\nActual number of top results to keep: {actual_top_n}")
    print(f"\nTop {actual_top_n} model/p-value combinations to keep:")
    for i, (idx, row) in enumerate(top_results.iterrows(), 1):
        print(f"  {i}. Model: {row['h2model']:20s}, P-value: {row['pvalue']:.6e}, "
              f"h²: {row['h2']:.4f}, Train AUC: {row['Train_pure_prs']:.4f}, "
              f"Test AUC: {row['Test_pure_prs']:.4f}")
    
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
    
    for idx, target_pval in enumerate(top_results['pvalue'].values):
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
                print(f"Warning: Could not delete {os.path.basename(profile_file)}: {e}")
    
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
    filtered_results = filtered_results.sort_values('Test_pure_prs', ascending=False).reset_index(drop=True)
    
    # Save the filtered results
    filtered_results.to_csv(results_file, index=False)
    
    print(f"\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    print(f"Profile files:")
    print(f"  Total found: {len(all_profiles)}")
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

def evaluate_pure_prs_gcta(traindirec, newtrainfilename, models, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile, tempdata, _lambda1):
    """
    Evaluate PURE PRS performance without model fitting
    For CAGI submission - just raw PRS discrimination
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING PURE PRS - {models}")
    print(f"{'='*80}")
    
    threshold_values = allpvalues

    # Load phenotypes
    tempphenotype_train = pd.read_table(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".fam", 
                                       sep="\s+", header=None)
    phenotype_train = pd.DataFrame()
    phenotype_train["Phenotype"] = tempphenotype_train[5].replace({1: 0, 2: 1})
    
    tempphenotype_test = pd.read_table(traindirec+os.sep+testfilename+".clumped.pruned"+".fam", 
                                      sep="\s+", header=None)
    phenotype_test = pd.DataFrame()
    phenotype_test["Phenotype"] = tempphenotype_test[5].replace({1: 0, 2: 1})
    
    print(f"Train phenotype distribution:\n{phenotype_train['Phenotype'].value_counts()}")
    print(f"Test phenotype distribution:\n{phenotype_test['Phenotype'].value_counts()}")
    
    global prs_result
    
    print(f"Processing {len(threshold_values)} p-value thresholds...")
    best_train_auc = 0
    best_test_auc = 0
    best_threshold = None
    
    for idx, i in enumerate(threshold_values):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(threshold_values)}")
            
        try:
            prs_train = pd.read_table(traindirec+os.sep+Name+os.sep+"train_data.pv_"+f"{i}.profile", 
                                      sep="\s+", usecols=["FID", "IID", "SCORE"])
            prs_test = pd.read_table(traindirec+os.sep+Name+os.sep+"test_data.pv_"+f"{i}.profile", 
                                     sep="\s+", usecols=["FID", "IID", "SCORE"])
            
            # Check for NaN/inf in PRS scores
            if prs_train['SCORE'].isna().any() or prs_test['SCORE'].isna().any():
                continue
            if np.isinf(prs_train['SCORE']).any() or np.isinf(prs_test['SCORE']).any():
                continue
            
            # Calculate PURE PRS AUC (no model fitting!)
            train_auc = roc_auc_score(phenotype_train["Phenotype"].values, prs_train['SCORE'].values)
            test_auc = roc_auc_score(phenotype_test["Phenotype"].values, prs_test['SCORE'].values)
            
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_train_auc = train_auc
                best_threshold = i

            prs_result = prs_result._append({
                "clump_p1": c1_val,
                "clump_r2": c2_val,
                "clump_kb": c3_val,
                "p_window_size": p1_val,
                "p_slide_size": p2_val,
                "p_LD_threshold": p3_val,
                "pvalue": i,
                "numberofpca": p,
                "numberofvariants": len(pd.read_csv(traindirec+os.sep+newtrainfilename+".clumped.pruned.bim")),
                "h2model": models,
                "h2": tempdata,
                "gcta_lambda": _lambda1,
                "Train_pure_prs": train_auc,
                "Test_pure_prs": test_auc,
            }, ignore_index=True)
            
        except Exception as e:
            continue

        # Save incrementally
        prs_result.to_csv(traindirec+os.sep+Name+os.sep+"Results.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"BEST RESULT FOR {models}")
    print(f"{'='*80}")
    print(f"Best Test AUC: {best_test_auc:.4f}")
    print(f"Best Train AUC: {best_train_auc:.4f}")
    print(f"Best p-value: {best_threshold:.2e}")
    print(f"{'='*80}")
    
    return

def transform_gcta_data(traindirec, newtrainfilename, models, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    
    print(f"\n{'='*80}")
    print(f"Processing model: {models}")
    print(f"{'='*80}")
    
    os.system("awk '{print $3,$8}' ./" + filedirec + os.sep + filedirec + ".txt > ./" + traindirec + os.sep + "SNP.pvalue")

    tempphenotype_train = pd.read_table(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".fam", 
                                       sep="\s+", header=None)
    phenotype = pd.DataFrame()
    phenotype = tempphenotype_train[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+trainfilename+".PHENO", sep="\t", 
                    header=['FID', 'IID', 'PHENO'], index=False)

    pcs_train = pd.read_table(traindirec+os.sep+trainfilename+".eigenvec", sep="\s+", header=None, 
                               names=["FID", "IID"] + [f"PC{str(i)}" for i in range(1, int(p)+1)])

    covandpcs_train = pcs_train
    covandpcs_train.to_csv(traindirec+os.sep+trainfilename+".COV_PCA", sep="\t", index=False)
            
    _lambda1 = ""
    tempdata = 0.70  # Default h²
    
 
    # HERITABILITY ESTIMATION OPTIONS
    if models == "GCTA_genotype":
        if os.path.exists(traindirec+os.sep+"train_data.hsq"):
            os.remove(traindirec+os.sep+"train_data.hsq")
        command1 = [
            './gcta',
            '--bfile', traindirec+os.sep+newtrainfilename+".clumped.pruned",
            '--make-grm',
            '--out', traindirec+os.sep+trainfilename
        ]
        subprocess.run(command1)
 
        # Use prevalence-adjusted GREML for binary traits
        command2 = [
            './gcta',
            '--grm', traindirec+os.sep+trainfilename,
            '--pheno', traindirec+os.sep+trainfilename+".PHENO",
            '--reml',
            '--prevalence', '0.05',
            '--out', traindirec+os.sep+trainfilename
        ]
        subprocess.run(command2)
        
        try:
            tempdata = pd.read_csv(traindirec+os.sep+"train_data.hsq", sep="\t")
            tempdata = tempdata[tempdata["Source"]=="V(G)/Vp"]["Variance"].values[0]
            print(f"Estimated h² = {tempdata:.4f}")
            
            # Use fixed value if h² is unrealistically low
            if tempdata < 0.4:
                print(f"WARNING: h² ({tempdata:.4f}) is very low. Using fixed h² = 0.70")
                tempdata = 0.70
        except:
            print("Heritability estimation failed. Using fixed h² = 0.70")
            tempdata = 0.70

        variants = len(pd.read_csv(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".bim"))
        m = variants
        _lambda1 = m * (1 / float(tempdata) - 1)
        print(f"Lambda = {_lambda1:,.2f}")

 
    if float(_lambda1) > 0:
        print(f"\nRunning GCTA-SBLUP with lambda = {_lambda1:,.2f}")
        gcta_command = [
            './gcta',
            '--bfile', traindirec+os.sep+newtrainfilename+".clumped.pruned",
            '--cojo-file', filedirec + os.sep + filedirec+"_GCTA.txt",
            '--cojo-wind', str(c3_val),
            '--cojo-sblup', str(_lambda1),
            '--out', traindirec+os.sep+"Train_data_SBLUP"
        ]
        result = subprocess.run(gcta_command, capture_output=True, text=True)
        if result.returncode != 0:
            print("GCTA-SBLUP failed!")
            print(result.stderr)
            return
    else:
        print("There is an issue with GCTA command! Lambda is negative or zero.")
        return 

    # Read SBLUP output
    originalgwas = filedirec+os.sep+filedirec+".txt"
    data = pd.read_csv(originalgwas, sep="\s+")
    sblupfile = traindirec+os.sep+"Train_data_SBLUP.sblup.cojo"
    
    if not os.path.exists(sblupfile):
        print(f"SBLUP output file not found: {sblupfile}")
        return
        
    data2 = pd.read_csv(sblupfile, sep="\s+", header=None, 
                       names=["SNP","A1","GWAS_Effect","SBLUP_Effect"])
    
    # Show shrinkage statistics
    shrinkage = (1 - data2['SBLUP_Effect'].abs().mean() / data2['GWAS_Effect'].abs().mean()) * 100
    print(f"Mean shrinkage: {shrinkage:.2f}%")
    print(f"SBLUP effect range: {data2['SBLUP_Effect'].min():.6f} to {data2['SBLUP_Effect'].max():.6f}")
    
    # CRITICAL: For binary phenotypes, ALWAYS use BETA (log-scale) for PLINK scoring
    # PLINK's --score expects additive effects on the log scale
    if check_phenotype_is_binary_or_continous(filedirec) == "Binary":
        # SBLUP effects are already on log-scale (log OR)
        # DO NOT convert to OR - keep as BETA for proper additive scoring
        data2["BETA"] = data2["SBLUP_Effect"]
        data2 = data2[["SNP","A1","BETA"]]
        print(f"Using BETA (log-scale) for binary trait")
        print(f"BETA range: {data2['BETA'].min():.6f} to {data2['BETA'].max():.6f}")
    else:
        # For continuous phenotypes, use effects as-is
        data2["BETA"] = data2["SBLUP_Effect"]
        data2 = data2[["SNP","A1","BETA"]]
        print(f"Using BETA for continuous trait")
        print(f"BETA range: {data2['BETA'].min():.6f} to {data2['BETA'].max():.6f}")
    
    # Handle infinite and missing values
    data2.replace([-np.inf, np.inf], 0, inplace=True)
    data2.fillna(0, inplace=True)
    
    # Save GCTA GWAS file with BETA (not OR)
    data2.to_csv(traindirec+os.sep+"GCTA_GWAS.txt", sep="\t", header=False, index=False)
    
    data = data[data["SNP"].isin(data2["SNP"].values)]
    data[["SNP","P"]].to_csv(traindirec+os.sep+"SNP_SBLUP.pvalue", index=False, sep="\t")
    
    # Calculate PRS for training data using BETA
    print("Calculating training PRS...")
    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--score", traindirec+os.sep+"GCTA_GWAS.txt", "1", "2", "3",
        "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP_SBLUP.pvalue",
        "--allow-no-sex",
        "--out", traindirec+os.sep+Name+os.sep+trainfilename
    ]
    subprocess.run(command, capture_output=True)

    # Calculate PRS for test data using BETA
    print("Calculating test PRS...")
    command = [
        "./plink",
        "--bfile", folddirec+os.sep+testfilename+".clumped.pruned",
        "--score", traindirec+os.sep+"GCTA_GWAS.txt", "1", "2", "3",
        "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP_SBLUP.pvalue",
        "--allow-no-sex",
        "--out", folddirec+os.sep+Name+os.sep+testfilename
    ]
    subprocess.run(command, capture_output=True)

    # Calculate PRS for validation data using BETA
    print("Calculating validation PRS...")
    command = [
        "./plink",
        "--bfile", valdirec+os.sep+valfilename,
        "--score", traindirec+os.sep+"GCTA_GWAS.txt", "1", "2", "3",
        "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP_SBLUP.pvalue",
        "--allow-no-sex",
        "--out", traindirec+os.sep+Name+os.sep+valfilename
    ]
    subprocess.run(command, capture_output=True)

    print("Evaluating pure PRS (no model fitting)...")
    evaluate_pure_prs_gcta(traindirec, newtrainfilename, models, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile, tempdata, _lambda1)
    
    print(f"{'='*80}")
    print(f"Completed model: {models}")
    print(f"{'='*80}\n")

# Main execution
h2models = ["GCTA_genotype"]

result_directory = "GCTA3"
create_directory(folddirec+os.sep+result_directory)

print("\n" + "="*80)
print("STARTING GCTA ANALYSIS - PURE PRS ONLY")
print("="*80)
print(f"Phenotype: {filedirec}")
print(f"Fold: {foldnumber}")
print(f"Models to test: {h2models}")
print(f"Evaluation: PURE PRS AUC (no model fitting)")
print("="*80 + "\n")

for p1_val in p_window_size:
 for p2_val in p_slide_size: 
  for p3_val in p_LD_threshold:
   for c1_val in clump_p1:
    for c2_val in clump_r2:
     for c3_val in clump_kb:
      for p in numberofpca:
       # Perform clumping and pruning once per parameter set
    #    perform_clumping_and_pruning_on_individual_data(
    #        folddirec, newtrainfilename, p,
    #        str(p1_val), str(p2_val), str(p3_val),
    #        str(c1_val), str(c2_val), str(c3_val),
    #        result_directory, pvaluefile
    #    )
       
    #    # Calculate PCA once per parameter set
    #    calculate_pca_for_traindata_testdata_for_clumped_pruned_snps(
    #        folddirec, newtrainfilename, p
    #    )
       
       # Test each heritability estimation model
       for h2model in h2models:
        transform_gcta_data(folddirec, newtrainfilename, h2model, p, 
                           str(p1_val), str(p2_val), str(p3_val), 
                           str(c1_val), str(c2_val), str(c3_val), 
                           result_directory, pvaluefile)

# Cleanup: keep only top 10 performing model files
print("\n" + "="*80)
print("PERFORMING FINAL CLEANUP")
print("="*80)
 

print("\n" + "="*80)
print("✅ GCTA ANALYSIS COMPLETED!")
print("="*80)
print(f"Results saved in: {folddirec + os.sep + result_directory}")

# Display summary
if os.path.exists(folddirec + os.sep + result_directory + os.sep + "Results.csv"):
    results = pd.read_csv(folddirec + os.sep + result_directory + os.sep + "Results.csv")
    print("\n" + "="*80)
    print("SUMMARY: Best Pure PRS by Model")
    print("="*80)
    
    for model in h2models:
        model_results = results[results['h2model'] == model]
        if len(model_results) > 0:
            best_idx = model_results['Test_pure_prs'].idxmax()
            best_row = model_results.loc[best_idx]
            print(f"{model:30s}: Test AUC = {best_row['Test_pure_prs']:.4f}, "
                  f"Train AUC = {best_row['Train_pure_prs']:.4f}, "
                  f"h² = {best_row['h2']:.4f}, "
                  f"Lambda = {best_row['gcta_lambda']:,.0f}")
    
    # Overall best
    best_overall_idx = results['Test_pure_prs'].idxmax()
    best_overall = results.loc[best_overall_idx]
    
    print("\n" + "="*80)
    print("🏆 BEST OVERALL PRS ACROSS ALL GCTA MODELS")
    print("="*80)
    print(f"Model: {best_overall['h2model']}")
    print(f"Heritability (h²): {best_overall['h2']:.4f}")
    print(f"Lambda: {best_overall['gcta_lambda']:,.2f}")
    print(f"P-value threshold: {best_overall['pvalue']:.2e}")
    print(f"Number of variants: {int(best_overall['numberofvariants'])}")
    print(f"Test AUC: {best_overall['Test_pure_prs']:.4f}")
    print(f"Train AUC: {best_overall['Train_pure_prs']:.4f}")
    print("="*80)