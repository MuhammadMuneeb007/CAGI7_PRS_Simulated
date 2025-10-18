import os
import pandas as pd
import numpy as np
import sys
import glob

filedirec = sys.argv[1]

def check_phenotype_is_binary_or_continous(filedirec):
    # Read the processed quality controlled file for a phenotype
    df = pd.read_csv(filedirec+os.sep+filedirec+'.fam',sep="\s+",header=None)
    column_values = df[5].unique()
 
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"


# Read the GWAS file.
GWAS = filedirec + os.sep + filedirec+".gz"
# df = pd.read_csv(GWAS,compression= "gzip",sep="\s+")

# if check_phenotype_is_binary_or_continous(filedirec)=="Binary":
#     if "BETA" in df.columns.to_list():
#         # For Binary Phenotypes - convert BETA to OR
#         df["OR"] = np.exp(df["BETA"])
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'MAF']]
#     else:
#         # Already has OR
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'MAF']]
 
# elif check_phenotype_is_binary_or_continous(filedirec)=="Continous":
#     if "BETA" in df.columns.to_list():
#         # For Continous Phenotype - keep BETA
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
#     else:
#         # Convert OR to BETA
#         df["BETA"] = np.log(df["OR"])
#         df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]

 
# df.to_csv(filedirec + os.sep +filedirec+"_PRSice-2.txt",sep="\t",index=False)
# print(df.head().to_markdown())
# print("Length of DataFrame!",len(df))


from operator import index
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
import os

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

# P-value range for PRSice-2
minimumpvalue = str(1e-100)
maximumpvalue = str(1)
interval = str(0.001)

print("Minimum P-value:", minimumpvalue)
print("Maximum P-value:", maximumpvalue)
print("Interval:", interval)

# Initializing an empty DataFrame with specified column names
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                   "pvalue", "numberofpca", "PRSice-2_Model",
                                   "Train_pure_prs", "Test_pure_prs"])


def perform_clumping_and_pruning_on_individual_data(traindirec, newtrainfilename, numberofpca, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    """
    Perform clumping and pruning on training data
    """
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

    # Extract valid SNPs
    command = f"awk 'NR!=1{{print $3}}' {traindirec}{os.sep}{trainfilename}.clumped > {traindirec}{os.sep}{trainfilename}.valid.snp"
    os.system(command)
    
    # Create clumped and pruned files
    command = [
        "./plink",
        "--make-bed",
        "--allow-no-sex",
        "--bfile", traindirec+os.sep+newtrainfilename,
        "--indep-pairwise", p1_val, p2_val, p3_val,
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
    """
    Calculate PCA for both train and test data
    """
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

 

def evaluate_pure_prs_prsice(traindirec, newtrainfilename, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, prsicemodel):
    """
    Evaluate PURE PRS performance without model fitting
    For CAGI submission - just raw PRS discrimination
    """
    print("\n" + "="*80)
    print(f"EVALUATING PURE PRS - PRSice-2 Model: {prsicemodel}")
    print("="*80)
    
    # Read phenotypes
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
    
    # Read PRS scores from PRSice-2 output
    train_all_prs_direc = traindirec+os.sep+Name+os.sep+p+"_"+prsicemodel+"_Train_PRSice_PRS_"+p1_val+"_"+p2_val+"_"+p3_val+"_"+c1_val+"_"+c2_val+"_"+c3_val
    test_all_prs_direc = traindirec+os.sep+Name+os.sep+p+"_"+prsicemodel+"_Test_PRSice_PRS_"+p1_val+"_"+p2_val+"_"+p3_val+"_"+c1_val+"_"+c2_val+"_"+c3_val
    
    try:
        train_prs_frame = pd.read_csv(train_all_prs_direc+".all_score", sep="\s+")
        test_prs_frame = pd.read_csv(test_all_prs_direc+".all_score", sep="\s+")
    except:
        print(f"ERROR: Could not read PRS files for {prsicemodel}")
        return
    
    # Get all p-value columns (skip FID, IID)
    pvalues = [col for col in train_prs_frame.columns if col not in ['FID', 'IID']]
    
    print(f"Found {len(pvalues)} p-value thresholds")
    
    global prs_result
    
    for idx, pval_col in enumerate(pvalues):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(pvalues)}...")
        
        try:
            # Get PRS for this p-value threshold
            prs_train = train_prs_frame[pval_col].values
            prs_test = test_prs_frame[pval_col].values
            
            # Calculate pure PRS AUC (no model fitting!)
            train_auc = roc_auc_score(phenotype_train["Phenotype"].values, prs_train)
            test_auc = roc_auc_score(phenotype_test["Phenotype"].values, prs_test)
            
            # Save results
            prs_result = prs_result._append({
                "clump_p1": c1_val,
                "clump_r2": c2_val,
                "clump_kb": c3_val,
                "p_window_size": p1_val,
                "p_slide_size": p2_val,
                "p_LD_threshold": p3_val,
                "pvalue": pval_col,
                "numberofpca": p,
                "PRSice-2_Model": prsicemodel,
                "Train_pure_prs": train_auc,
                "Test_pure_prs": test_auc,
            }, ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {pval_col}: {e}")
            continue
    
    # Save results incrementally
    prs_result.to_csv(traindirec+os.sep+Name+os.sep+"Results.csv", index=False)
    
    # Print best result
    if len(prs_result) > 0:
        best_idx = prs_result['Test_pure_prs'].idxmax()
        best_result = prs_result.loc[best_idx]
        
        print(f"\n{'='*80}")
        print(f"BEST PRS FOR {prsicemodel}")
        print(f"{'='*80}")
        print(f"P-value: {best_result['pvalue']}")
        print(f"Train AUC: {best_result['Train_pure_prs']:.4f}")
        print(f"Test AUC: {best_result['Test_pure_prs']:.4f}")
    
    return

def calculate_pca_for_validation_data(traindirec, p):
    """
    Calculate PCA for validation data using FlashPCA and the same valid SNPs
    """
    import os
    print("\n" + "="*80)
    print("CALCULATING PCA FOR VALIDATION DATA USING FLASHPCA")
    print("="*80)
    
    # Step 1: Filter validation data to include only valid SNPs
    filtered_prefix = traindirec + os.sep + valfilename + ".filtered"
    
    command_filter = [
        "./plink",
        "--bfile", valdirec + os.sep + valfilename,
        "--extract", traindirec + os.sep + trainfilename + ".valid.snp",
        "--make-bed",
        "--allow-no-sex",
        "--out", filtered_prefix
    ]
    
    print("Step 1: Filtering validation data to valid SNPs")
    print(" ".join(command_filter))
    subprocess.run(command_filter)
    
    # Step 2: Run FlashPCA on filtered data
    command_flashpca = [
        "./flashpca_x86-64",
        "--bfile", filtered_prefix,
        "--ndim", p,
        "--numthreads", "8",
        "--outpc", traindirec + os.sep + valfilename + ".pcs.txt",
        "--outvec", traindirec + os.sep + valfilename + ".eigenvec",
        "--outval", traindirec + os.sep + valfilename + ".eigenval"
    ]
    
    print("\nStep 2: Running FlashPCA")
    print(" ".join(command_flashpca))
    subprocess.run(command_flashpca)
    import os

    # Remove filtered validation files to save space
    filtered_prefix = traindirec + os.sep + valfilename + ".filtered"
    for ext in [".bed", ".bim", ".fam", ".log", ".nosex"]:
        file_path = filtered_prefix + ext
        if os.path.exists(file_path):
            os.remove(file_path)




prs_result = pd.DataFrame()

def transform_prsice_data(traindirec, newtrainfilename, prsicemodel, numberofpca, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    """
    Main function to run PRSice-2 and evaluate pure PRS
    """
    print("\n" + "="*80)
    print(f"PROCESSING: {prsicemodel}")
    print(f"Parameters: p={numberofpca}, p1={p1_val}, p2={p2_val}, p3={p3_val}")
    print(f"            c1={c1_val}, c2={c2_val}, c3={c3_val}")
    print("="*80)
    
    # =====================================================================
    # TRAINING DATA PREPARATION
    # =====================================================================
    # Prepare phenotype file for training
    tempphenotype_train = pd.read_table(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".fam", 
                                       sep="\s+", header=None)
    phenotype = tempphenotype_train[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+trainfilename+".PHENO", sep="\t", 
                    header=['FID', 'IID', 'PHENO'], index=False)
 
    # Prepare PCA/Covariate file for training
    pcs_train = pd.read_table(traindirec+os.sep+trainfilename+".eigenvec", sep="\s+", header=None, 
                             names=["FID", "IID"] + [f"PC{str(i)}" for i in range(1, int(numberofpca)+1)])
    pcs_train.to_csv(traindirec+os.sep+trainfilename+".PCA", sep="\t", index=False)
    # Also save as .cov for reference
    pcs_train.to_csv(traindirec+os.sep+trainfilename+".cov", sep="\t", index=False)
    
    # =====================================================================
    # TEST DATA PREPARATION
    # =====================================================================
    # Prepare phenotype file for testing
    tempphenotype_test = pd.read_table(traindirec+os.sep+testfilename+".clumped.pruned"+".fam", 
                                      sep="\s+", header=None)
    phenotype = tempphenotype_test[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+testfilename+".PHENO", sep="\t", 
                    header=['FID', 'IID', 'PHENO'], index=False)
 
    # Prepare PCA/Covariate file for testing
    pcs_test = pd.read_table(traindirec+os.sep+testfilename+".eigenvec", sep="\s+", header=None, 
                            names=["FID", "IID"] + [f"PC{str(i)}" for i in range(1, int(numberofpca)+1)])
    pcs_test.to_csv(traindirec+os.sep+testfilename+".PCA", sep="\t", index=False)
    # Also save as .cov for reference
    pcs_test.to_csv(traindirec+os.sep+testfilename+".cov", sep="\t", index=False)
    
    # =====================================================================
    # VALIDATION DATA PREPARATION
    # =====================================================================
    # Prepare phenotype file for validation
    try:
        tempval_fam = pd.read_table(valdirec+os.sep+valfilename+".fam", sep="\s+", header=None)
        val_pheno = tempval_fam[[0,1,5]]
        val_pheno.to_csv(traindirec+os.sep+valfilename+".PHENO", sep="\t", 
                        header=['FID', 'IID', 'PHENO'], index=False)
        print(f"✓ Validation phenotype file created")
    except Exception as e:
        print(f"ERROR: Could not create phenotype file for validation: {e}")
        return
    
    # Read the ACTUAL PCA file for validation (must be calculated first!)
    try:
        pcs_val = pd.read_table(traindirec+os.sep+valfilename+".eigenvec", sep="\s+", header=None, 
                               names=["FID", "IID"] + [f"PC{str(i)}" for i in range(1, int(numberofpca)+1)])
        pcs_val.to_csv(traindirec+os.sep+valfilename+".PCA", sep="\t", index=False)
        # Also save as .cov for reference
        pcs_val.to_csv(traindirec+os.sep+valfilename+".cov", sep="\t", index=False)
        print(f"✓ Validation PCA/Covariate file created with {numberofpca} PCs")
    except Exception as e:
        print(f"ERROR: Could not read PCA file for validation: {e}")
        print(f"Make sure PCA was calculated using calculate_pca_for_validation_data()")
        return
    
    # Determine binary vs continuous
    binary_phenotype = ""
    stat_col = ""
    
    if check_phenotype_is_binary_or_continous(filedirec) == "Binary":
        binary_phenotype = 'T'
        stat_col = "OR"
    else:
        binary_phenotype = 'F'
        stat_col = "BETA"
    
    print(f"Phenotype type: {'Binary' if binary_phenotype == 'T' else 'Continuous'}")
    print(f"Using stat column: {stat_col}")
    
    # =====================================================================
    # RUN PRSICE-2 FOR TRAINING
    # =====================================================================
    command = [
        './PRSice',
        '--base', filedirec + os.sep + filedirec + "_PRSice-2.txt",
        '--target', traindirec + os.sep + newtrainfilename + ".clumped.pruned",
        '--pheno', traindirec + os.sep + trainfilename + ".PHENO",
        '--cov', traindirec + os.sep + trainfilename + ".PCA",
        '--stat', stat_col,
        '--ld', 'ref',
        '--all-score',
        '--lower', minimumpvalue, 
        '--upper', maximumpvalue,
        '--interval', interval,
        '--score', prsicemodel,
        '--binary-target', binary_phenotype,
        '--no-clump',
        '--out', traindirec + os.sep + Name + os.sep + numberofpca + "_" + prsicemodel + "_Train_PRSice_PRS_" + p1_val + "_" + p2_val + "_" + p3_val + "_" + c1_val + "_" + c2_val + "_" + c3_val
    ]
    print("\n" + "="*80)
    print("Running PRSice-2 for TRAINING data...")
    print("="*80)
    subprocess.run(command)
    
    # =====================================================================
    # RUN PRSICE-2 FOR TEST
    # =====================================================================
    command = [
        './PRSice',
        '--base', filedirec + os.sep + filedirec + "_PRSice-2.txt",
        '--target', traindirec + os.sep + testfilename + ".clumped.pruned",
        '--pheno', traindirec + os.sep + testfilename + ".PHENO",
        '--cov', traindirec + os.sep + testfilename + ".PCA",
        '--stat', stat_col,
        '--ld', 'ref',
        '--all-score',
        '--lower', minimumpvalue, 
        '--upper', maximumpvalue,
        '--interval', interval,
        '--score', prsicemodel,
        '--binary-target', binary_phenotype,
        '--no-clump',
        '--out', traindirec + os.sep + Name + os.sep + numberofpca + "_" + prsicemodel + "_Test_PRSice_PRS_" + p1_val + "_" + p2_val + "_" + p3_val + "_" + c1_val + "_" + c2_val + "_" + c3_val
    ]
    print("\n" + "="*80)
    print("Running PRSice-2 for TEST data...")
    print("="*80)
    subprocess.run(command)
    
    # =====================================================================
    # RUN PRSICE-2 FOR VALIDATION
    # =====================================================================
    command = [
        './PRSice',
        '--base', filedirec + os.sep + filedirec + "_PRSice-2.txt",
        '--target', valdirec + os.sep + valfilename,
        '--pheno', traindirec + os.sep + valfilename + ".PHENO",
        '--cov', traindirec + os.sep + valfilename + ".PCA",
        '--stat', stat_col,
        '--ld', 'ref',
        '--all-score',
        '--lower', minimumpvalue, 
        '--upper', maximumpvalue,
        '--interval', interval,
        '--score', prsicemodel,
        '--binary-target', binary_phenotype,
        '--no-clump',
        '--no-regress',
        '--out', traindirec + os.sep + Name + os.sep + numberofpca + "_" + prsicemodel + "_Val_PRSice_PRS_" + p1_val + "_" + p2_val + "_" + p3_val + "_" + c1_val + "_" + c2_val + "_" + c3_val
    ]
    print("\n" + "="*80)
    print("Running PRSice-2 for VALIDATION data...")
    print("="*80)
    print(" ".join(command))
    subprocess.run(command)
    
    # Evaluate pure PRS (no model fitting!)
    evaluate_pure_prs_prsice(traindirec, newtrainfilename, numberofpca, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, prsicemodel)
    
    return

# Main execution
PRSiceModels = ['avg', 'sum', 'std', 'con-std']
PRSiceModels = ['avg']

result_directory = "PRSice-2-3"

# Create result directory
create_directory(folddirec + os.sep + result_directory)

print("\n" + "="*80)
print("STARTING PRSICE-2 ANALYSIS")
print("="*80)
print(f"Dataset: {filedirec}")
print(f"Fold: {foldnumber}")
print(f"PRSice-2 Models: {PRSiceModels}")
print(f"Number of PCA: {numberofpca}")
print("="*80 + "\n")

# Nested loops to iterate over different parameter values
# Nested loops to iterate over different parameter values
for p1_val in p_window_size:
    for p2_val in p_slide_size: 
        for p3_val in p_LD_threshold:
            for c1_val in clump_p1:
                for c2_val in clump_r2:
                    for c3_val in clump_kb:
                        for p in numberofpca:
                            # First perform clumping and pruning (once per parameter set)
                            # perform_clumping_and_pruning_on_individual_data(
                            #     folddirec, newtrainfilename, p, 
                            #     str(p1_val), str(p2_val), str(p3_val), 
                            #     str(c1_val), str(c2_val), str(c3_val), 
                            #     result_directory, None
                            # )
                            
                            # # Calculate PCA for train and test (once per parameter set)
                            # calculate_pca_for_traindata_testdata_for_clumped_pruned_snps(
                            #     folddirec, newtrainfilename, p
                            # )
                            
                            # # Calculate PCA for validation (once per parameter set)
                            calculate_pca_for_validation_data(folddirec, p)
                            
                            # Try each PRSice-2 model
                            for prsicemodel in PRSiceModels:
                                transform_prsice_data(
                                    folddirec, newtrainfilename, prsicemodel, p, 
                                    str(p1_val), str(p2_val), str(p3_val), 
                                    str(c1_val), str(c2_val), str(c3_val), 
                                    result_directory, None
                                )

# Cleanup: keep only top 10 performing model files
print("\n" + "="*80)
print("PERFORMING FINAL CLEANUP")
print("="*80)
 
print("\n" + "="*80)
print("✅ PRSICE-2 ANALYSIS COMPLETE")
print("="*80)
print(f"Results saved to: {folddirec}/{result_directory}/Results.csv")

# Print overall best result
if len(prs_result) > 0:
    best_overall_idx = prs_result['Test_pure_prs'].idxmax()
    best_overall = prs_result.loc[best_overall_idx]
    
    print(f"\n{'='*80}")
    print(f"BEST OVERALL PRS ACROSS ALL MODELS")
    print(f"{'='*80}")
    print(f"Model: {best_overall['PRSice-2_Model']}")
    print(f"P-value: {best_overall['pvalue']}")
    print(f"Train AUC: {best_overall['Train_pure_prs']:.4f}")
    print(f"Test AUC: {best_overall['Test_pure_prs']:.4f}")
    print(f"Parameters: p={best_overall['numberofpca']}, "
          f"clump_p1={best_overall['clump_p1']}, "
          f"clump_r2={best_overall['clump_r2']}")
    print("="*80)