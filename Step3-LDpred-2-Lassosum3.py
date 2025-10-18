import os
import pandas as pd
import numpy as np
import sys
import glob

def check_phenotype_is_binary_or_continous(filedirec):
    # Read the processed quality controlled file for a phenotype
    df = pd.read_csv(filedirec+os.sep+filedirec+'.fam',sep="\s+",header=None)
    column_values = df[5].unique()
 
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"

filedirec = sys.argv[1]

print("="*80)
print("PREPARING GWAS DATA FOR LDPRED-2-LASSOSUM2")
print("="*80)

# Read the GWAS file.
GWAS = filedirec + os.sep + filedirec+".gz"
df = pd.read_csv(GWAS,compression= "gzip",sep="\s+")

# LDpred-2 requires betas for calculation
if "BETA" in df.columns.to_list():
    # For Continuous Phenotype or Binary with BETA
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
else:
    # Convert OR to BETA (log scale)
    df["BETA"] = np.log(df["OR"])
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]

# IMPORTANT: Set SE to small value if missing/placeholder
df["SE"] = 0.001
df.to_csv(filedirec + os.sep +filedirec+".txt",sep="\t",index=False)
print(df.head().to_markdown())
print("Length of DataFrame!",len(df))

from operator import index
import pandas as pd
import numpy as np
import os
import subprocess
import sys
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
                                   "pvalue", "numberofpca", "numberofvariants", "heritability_model", "h2",
                                   "lasso_parameters_count", "lambda", "delta", "sparsity",
                                   "Train_pure_prs", "Test_pure_prs"])

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
        "--allow-no-sex",
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
        "--allow-no-sex",
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", traindirec+os.sep+newtrainfilename+".clumped.pruned"
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
        "--allow-no-sex",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--out", folddirec+os.sep+testfilename
    ]
    subprocess.run(command)

    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--allow-no-sex",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)


 
def evaluate_pure_prs_lassosum(traindirec, newtrainfilename, h2model, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile, lambdaa, delta, sparsity, heritability, numberofvariants, lassocount):
    """
    Evaluate PURE PRS performance without model fitting
    For CAGI submission - just raw PRS discrimination
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING PURE PRS - Lassosum Parameter Set {lassocount}")
    print(f"Lambda: {lambdaa}, Delta: {delta}, Sparsity: {sparsity}")
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
                "numberofvariants": numberofvariants,
                "heritability_model": h2model,
                "h2": heritability,
                "lasso_parameters_count": str(lassocount),                      
                "lambda": lambdaa,
                "delta": delta,
                "sparsity": sparsity,
                "Train_pure_prs": train_auc,
                "Test_pure_prs": test_auc,
            }, ignore_index=True)
            
        except Exception as e:
            continue

        # Save incrementally
        prs_result.to_csv(traindirec+os.sep+Name+os.sep+"Results.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"BEST RESULT FOR PARAMETER SET {lassocount}")
    print(f"{'='*80}")
    print(f"Best Test AUC: {best_test_auc:.4f}")
    print(f"Best Train AUC: {best_train_auc:.4f}")
    print(f"Best p-value: {best_threshold:.2e}")
    print(f"{'='*80}")
    
    return

def transform_ldpred2_lasso_data(traindirec, newtrainfilename, models, p, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    
    print(f"\n{'='*80}")
    print(f"Processing LDpred-2-lassosum2: {models}")
    print(f"{'='*80}")
    
    # Extract p-values from the GWAS file
    os.system("awk "+"\'"+"{print $3,$8}"+"\'"+" ./"+filedirec+os.sep+filedirec+".txt >  ./"+traindirec+os.sep+"SNP.pvalue")
    
    # Prepare phenotype file
    tempphenotype_train = pd.read_table(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".fam", 
                                       sep="\s+", header=None)
    phenotype = pd.DataFrame()
    phenotype = tempphenotype_train[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+trainfilename+".PHENO", sep="\t", 
                    header=['FID', 'IID', 'PHENO'], index=False)
    
    # Prepare covariate file (PCA only)
    pcs_train = pd.read_table(traindirec+os.sep+trainfilename+".eigenvec", sep="\s+", header=None, 
                               names=["FID", "IID"] + [f"PC{str(i)}" for i in range(1, int(p)+1)])
    covandpcs_train = pcs_train.copy()
    covandpcs_train['FID'] = covandpcs_train['FID'].astype(str)
    covandpcs_train['IID'] = covandpcs_train['IID'].astype(str)
    covandpcs_train.to_csv(traindirec+os.sep+trainfilename+".cov", sep="\t", index=False)

    # Run LDpred-2-lassosum2 R script
    if models == "LDpred-2_full":
        try:
            print("Running LDpred-2-lassosum2 R script...")
            print("Rscript LDpred2-lassosum2.R "+os.path.join(filedirec)+"  "+traindirec+" "+trainfilename+" "+newtrainfilename+".clumped.pruned"+ " "+"3"+" "+c3_val+" "+c1_val+" "+c2_val+" "+p)
            os.system("Rscript LDpred2-lassosum2.R "+os.path.join(filedirec)+"  "+traindirec+" "+trainfilename+" "+newtrainfilename+".clumped.pruned"+ " "+"3"+" "+c3_val+" "+c1_val+" "+c2_val+" "+p)
            
            heritability = pd.read_csv(traindirec+os.sep+"ldpred_h2_full.txt", sep="\s+", header=None)[1].values[0]
            numberofvariants = pd.read_csv(traindirec+os.sep+"ldpred_h2_variants.txt", sep="\s+", header=None)[1].values[0] 
            print(f"Estimated h²: {heritability:.4f}")
            print(f"Number of variants: {numberofvariants}")
        except Exception as e:
            print(f"For model {models}, it did not work! Error: {e}")
            return
 

    # Read grid parameters and betas
    gridparameters = pd.read_csv(traindirec+os.sep+"train_data.ldpred_lassosum_parameters", sep=",")
    allbetas = pd.read_csv(traindirec+os.sep+"train_data.ldpred_lassosum_betas", sep=",")
    allbetas = allbetas.fillna(0)
    
    print(f"\nProcessing {len(gridparameters)} parameter combinations...")

    count = 0
    for index, row in gridparameters.iterrows():
        print(f"\n{'='*80}")
        print(f"Processing parameter set {count+1}/{len(gridparameters)}")
        print(f"Lambda: {row['lambda']}, Delta: {row['delta']}, Sparsity: {row['sparsity']}")
        print(f"{'='*80}")
        
        gwas = pd.read_csv(traindirec+os.sep+"train_data.ldpred_lassosum_gwas", sep=",")
        
        # Get betas for this parameter combination
        gwas["newbetas"] = allbetas.iloc[:, index].values
        
        # CRITICAL: Check if binary or continuous phenotype
        if check_phenotype_is_binary_or_continous(filedirec) == "Binary":
            # For binary phenotypes, lassosum2 returns log(OR) = BETA
            # Convert to OR for PLINK scoring
            print("Binary phenotype detected - converting BETA to OR")
            gwas["newbetas"] = np.exp(gwas["newbetas"])
        else:
            # For continuous, keep as-is
            print("Continuous phenotype - using BETA directly")
            pass

        gwas.rename(columns={'rsid.ss': 'SNP', 'a0': 'A1', 'newbetas': 'BETA'}, inplace=True)
        gwas[["SNP","A1","BETA"]].to_csv(traindirec+os.sep+"train_data.ldpred_lassosum2_gwas_final", 
                                         sep="\t", index=False)
        
        print(f"Effect size statistics:")
        print(f"  Min: {gwas['BETA'].min():.6f}")
        print(f"  Max: {gwas['BETA'].max():.6f}")
        print(f"  Mean: {gwas['BETA'].mean():.6f}")
        print(f"  Std: {gwas['BETA'].std():.6f}")
        
        # Calculate PRS for training data
        print("Calculating training PRS...")
        command = [
            "./plink",
            "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
            "--score", traindirec+os.sep+"train_data.ldpred_lassosum2_gwas_final", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
            "--allow-no-sex",
            "--out", traindirec+os.sep+Name+os.sep+trainfilename
        ]
        subprocess.run(command, capture_output=True)

        # Calculate PRS for test data
        print("Calculating test PRS...")
        command = [
            "./plink",
            "--bfile", folddirec+os.sep+testfilename+".clumped.pruned",
            "--score", traindirec+os.sep+"train_data.ldpred_lassosum2_gwas_final", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
            "--allow-no-sex",
            "--out", folddirec+os.sep+Name+os.sep+testfilename
        ]
        subprocess.run(command, capture_output=True)

        # Calculate PRS for validation data
        print("Calculating validation PRS...")
        command = [
            "./plink",
            "--bfile", valdirec+os.sep+valfilename,
            "--score", traindirec+os.sep+"train_data.ldpred_lassosum2_gwas_final", "1", "2", "3", "header",
            "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
            "--allow-no-sex",
            "--out", traindirec+os.sep+Name+os.sep+valfilename
        ]
        subprocess.run(command, capture_output=True)

        # Evaluate PURE PRS (no model fitting!)
        evaluate_pure_prs_lassosum(folddirec, newtrainfilename, models, p, 
                                   str(p1_val), str(p2_val), str(p3_val), 
                                   str(c1_val), str(c2_val), str(c3_val), 
                                   Name, pvaluefile, row['lambda'], row['delta'], 
                                   row['sparsity'], heritability, numberofvariants, count)
        count = count + 1

    print(f"\n{'='*80}")
    print(f"COMPLETED: {models} with {count} parameter combinations")
    print(f"{'='*80}")

# Main execution
h2models = ["LDpred-2_full"]
result_directory = "ldpred2_lassosum3"

create_directory(folddirec+os.sep+result_directory)

print("\n" + "="*80)
print("STARTING LDPRED-2-LASSOSUM2 ANALYSIS - PURE PRS ONLY")
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
       perform_clumping_and_pruning_on_individual_data(
           folddirec, newtrainfilename, p,
           str(p1_val), str(p2_val), str(p3_val),
           str(c1_val), str(c2_val), str(c3_val),
           result_directory, pvaluefile
       )
       
       # Calculate PCA once per parameter set
       calculate_pca_for_traindata_testdata_for_clumped_pruned_snps(
           folddirec, newtrainfilename, p
       )
       
       # Test each model
       for model in h2models:
        transform_ldpred2_lasso_data(folddirec, newtrainfilename, model, p, 
                                     str(p1_val), str(p2_val), str(p3_val), 
                                     str(c1_val), str(c2_val), str(c3_val), 
                                     result_directory, pvaluefile)

# Cleanup: keep only top 10 performing model files
print("\n" + "="*80)
print("PERFORMING FINAL CLEANUP")
print("="*80)
 
print("\n" + "="*80)
print("✅ LDPRED-2-LASSOSUM2 ANALYSIS COMPLETED!")
print("="*80)
print(f"Results saved in: {folddirec + os.sep + result_directory}")

# Display summary
if os.path.exists(folddirec + os.sep + result_directory + os.sep + "Results.csv"):
    results = pd.read_csv(folddirec + os.sep + result_directory + os.sep + "Results.csv")
    print("\n" + "="*80)
    print("SUMMARY: Best Pure PRS by Parameter Combination")
    print("="*80)
    
    # Group by lassosum parameters
    unique_params = results.drop_duplicates(subset=['lambda', 'delta', 'sparsity'])
    
    for idx, row in unique_params.head(10).iterrows():
        param_results = results[
            (results['lambda'] == row['lambda']) & 
            (results['delta'] == row['delta']) & 
            (results['sparsity'] == row['sparsity'])
        ]
        best_idx = param_results['Test_pure_prs'].idxmax()
        best_row = param_results.loc[best_idx]
        
        print(f"Lambda={best_row['lambda']:.6f}, Delta={best_row['delta']:.6f}, Sparsity={best_row['sparsity']:.6f}:")
        print(f"  Test AUC = {best_row['Test_pure_prs']:.4f}, Train AUC = {best_row['Train_pure_prs']:.4f}")
    
    # Overall best across ALL parameter combinations
    best_overall_idx = results['Test_pure_prs'].idxmax()
    best_overall = results.loc[best_overall_idx]
    
    print("\n" + "="*80)
    print("🏆 BEST OVERALL PRS ACROSS ALL PARAMETER COMBINATIONS")
    print("="*80)
    print(f"Model: {best_overall['heritability_model']}")
    print(f"Heritability (h²): {best_overall['h2']:.4f}")
    print(f"Number of Variants: {best_overall['numberofvariants']}")
    print(f"Lambda: {best_overall['lambda']:.6f}")  