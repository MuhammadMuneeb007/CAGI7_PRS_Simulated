import os
import pandas as pd
import numpy as np
import sys
import glob
from scipy.stats import norm

filedirec = sys.argv[1]

def check_phenotype_is_binary_or_continous(filedirec):
    """Check if phenotype is binary or continuous."""
    df = pd.read_csv(filedirec+os.sep+filedirec+'.fam',sep="\s+",header=None)
    column_values = df[5].unique()
    print(f"Phenotype values: {column_values}")
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"

# Read the GWAS file
print("="*80)
print("PREPARING GWAS DATA FOR LDAK")
print("="*80)

GWAS = filedirec + os.sep + filedirec+".gz"
df = pd.read_csv(GWAS, compression="gzip", sep="\s+")
print("\nOriginal GWAS data:")
print(df.head().to_markdown()) 

if "BETA" in df.columns.to_list():
    # Continuous phenotype
    print("\nProcessing Continuous Phenotype...")
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    
    # Check if SE is a placeholder (constant value)
    unique_se = df['SE'].nunique()
    if unique_se <= 10:  # Likely a placeholder
        print(f"⚠️  WARNING: SE has only {unique_se} unique values - calculating from P-values")
        Z = np.sign(df["BETA"]) * norm.ppf(1 - df["P"].clip(lower=1e-300, upper=1-1e-16) / 2)
        df["SE"] = np.abs(df["BETA"]) / np.abs(Z.replace(0, np.nan))
        df["SE"].fillna(df["SE"].median(), inplace=True)
        
else:
    # Binary phenotype - Convert OR to BETA
    print("\nProcessing Binary Phenotype...")
    df["BETA"] = np.log(df["OR"].replace(0, np.nan))
    
    # Calculate SE from P-value and BETA
    print("\nCalculating SE from P-values and BETA...")
    
    # Calculate Z-score from P-value: Z = Φ^(-1)(1 - P/2) * sign(BETA)
    Z = np.sign(df["BETA"]) * norm.ppf(1 - df["P"].clip(lower=1e-300, upper=1-1e-16) / 2)
    
    # Calculate SE: SE = |BETA| / |Z|
    df["SE"] = np.abs(df["BETA"]) / np.abs(Z.replace(0, np.nan))
    
    # Handle any remaining NaN or inf values
    df["SE"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["SE"].fillna(df["SE"].median(), inplace=True)
    
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]

# Replace infinities or NaNs globally
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

print("\nProcessed GWAS data:")
print(df.head().to_markdown())
print(f"\nStatistics:")
print(f"  BETA range: {df['BETA'].min():.6f} to {df['BETA'].max():.6f}")
print(f"  SE range: {df['SE'].min():.6f} to {df['SE'].max():.6f}")
print(f"  SE unique values: {df['SE'].nunique()}")
print(f"  SE mean: {df['SE'].mean():.6f}")

# Verify SE calculation is reasonable
df['Z_calculated'] = df['BETA'] / df['SE']
df['P_from_Z'] = 2 * (1 - norm.cdf(np.abs(df['Z_calculated'])))
correlation = df[['P', 'P_from_Z']].corr().iloc[0, 1]
print(f"\nValidation:")
print(f"  Correlation between original P and recalculated P: {correlation:.4f}")
if correlation > 0.95:
    print("  ✓ SE calculation is valid!")
else:
    print("  ⚠️  Warning: SE may need review")

# Save standard format for Plink clumping
df.to_csv(filedirec + os.sep + filedirec + ".txt", sep="\t", index=False)
print(f"\n✓ Saved standard format: {filedirec + os.sep + filedirec}.txt")

# Create LDAK-friendly summary file
df_transformed = pd.DataFrame({
    'Predictor': df['CHR'].astype(str) + ":" + df['BP'].astype(str),
    'A1': df['A1'],
    'A2': df['A2'],
    'n': df['N'],
    'Z': np.sign(df['BETA']) * norm.ppf(1 - df['P'].clip(lower=1e-300, upper=1 - 1e-16) / 2),
    'SNP': df['SNP']
})

# Remove multi-allelic SNPs
df_transformed = df_transformed[
    (df_transformed['A1'].str.len() == 1) &
    (df_transformed['A2'].str.len() == 1)
]

# Remove duplicates and reset index
df_transformed.drop_duplicates(subset=['Predictor'], inplace=True)
df_transformed.reset_index(drop=True, inplace=True)
df_transformed.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# Save LDAK format
output_path = f"{filedirec}{os.sep}{filedirec}.ldak"
df_transformed.to_csv(output_path, sep="\t", index=False)

print("\nLDAK summary file:")
print(df_transformed.head().to_markdown())
print(f"✓ Saved LDAK summary file: {output_path}")
print(f"Number of SNPs: {len(df_transformed)}")

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

# Simplified results - PURE PRS ONLY
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                   "pvalue", "numberofpca", "ldakmodel", "ldakpower", "ldaksubmodel",
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
        "--clump-kb", c3_val,
        "--allow-no-sex",
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
    
    command = [
        "./plink",
        "--allow-no-sex",
        "--bfile", folddirec+os.sep+testfilename+".clumped.pruned",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--out", folddirec+os.sep+testfilename
    ]
    subprocess.run(command)

    command = [
        "./plink",
        "--allow-no-sex",
        "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)

 

def evaluate_pure_prs_ldak(traindirec, newtrainfilename, p, ldakmodel, power, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile, ldaksubmodel):
    """
    Evaluate PURE PRS performance without model fitting
    This is what you need for CAGI submission
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING PURE PRS - {ldaksubmodel}")
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
            # Read PRS scores
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
                "ldakmodel": ldakmodel, 
                "ldakpower": str(power),
                "ldaksubmodel": ldaksubmodel,
                "Train_pure_prs": train_auc,
                "Test_pure_prs": test_auc,
            }, ignore_index=True)
            
        except Exception as e:
            continue
    
        # Save incrementally
        prs_result.to_csv(traindirec+os.sep+Name+os.sep+"Results.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"BEST RESULT FOR {ldaksubmodel}")
    print(f"{'='*80}")
    print(f"Best Test AUC: {best_test_auc:.4f}")
    print(f"Best Train AUC: {best_train_auc:.4f}")
    print(f"Best p-value: {best_threshold:.2e}")
    print(f"{'='*80}")
    
    return

prs_result = pd.DataFrame()

def transform_ldak_data(traindirec, newtrainfilename, numberofpca, ldakmodel, power, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):     
 
    print(f"\n{'='*80}")
    print(f"LDAK ANALYSIS: Model={ldakmodel}, Power={power}")
    print(f"{'='*80}")
    
    os.system("awk '{print $3,$8}' ./" + filedirec + os.sep + filedirec + ".txt > ./" + traindirec + os.sep + "SNP.pvalue")
    
    # Calculate correlations
    print("\nStep 1: Calculating LD correlations...")
    command2 = [
        './ldak',
        '--calc-cors', traindirec+os.sep+'cors',
        '--bfile', traindirec+os.sep+newtrainfilename+".clumped.pruned"
    ]
    subprocess.run(command2)
    print("✓ Correlations calculated")
    
    # Update cors.bim file to match GWAS format
    print("\nStep 2: Updating BIM file format...")
    df = pd.read_csv(filedirec + os.sep + filedirec + ".ldak", sep="\s+")
    t1 = pd.read_csv(traindirec+os.sep+'cors.cors.bim', sep="\s+", header=None)
    
    # Always update (file is regenerated each iteration)
    t1[1] = t1[0].astype(str) + ":" + t1[3].astype(str)
    t1.to_csv(traindirec+os.sep+'cors.cors.bim', sep="\t", header=False, index=None) 
    print("✓ BIM file updated")
    
    # Run LDAK mega-PRS
    print(f"\nStep 3: Running LDAK mega-PRS with {ldakmodel} model...")
    command3 = [
        './ldak',
        '--mega-prs', traindirec+os.sep+ldakmodel,
        '--model', ldakmodel,
        '--summary', filedirec + os.sep + filedirec + ".ldak",
        '--power', str(power),
        '--skip-cv', 'YES',
        '--cors', traindirec+os.sep+'cors',
        '--allow-ambiguous', 'YES',
    ]
    result = subprocess.run(command3, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in LDAK mega-PRS:")
        print(result.stderr)
        return
    print("✓ LDAK mega-PRS completed")
    
    # Process LDAK effects
    print("\nStep 4: Processing LDAK effect sizes...")
    df = pd.read_csv(filedirec + os.sep + filedirec + ".ldak", sep="\s+")
    
    effects_file = traindirec+os.sep+ldakmodel+".effects"
    if not os.path.exists(effects_file):
        print(f"ERROR: Effects file not found: {effects_file}")
        return
        
    df2 = pd.read_csv(effects_file, sep="\s+")
    
    df = df[df["Predictor"].isin(df2["Predictor"].values)]
    df2["SNP"] = df["SNP"].values
    df2.to_csv(traindirec+os.sep+ldakmodel+".effects", index=False, sep="\t")    
    
    temp = pd.read_csv(traindirec + os.sep + ldakmodel + ".effects", sep="\s+")
    numberofcolumns1 = temp.shape[1]
    print(f"Number of effect size columns: {numberofcolumns1}")
    
    for loop in range(4, numberofcolumns1 - 1):
        print(f"\n{'='*80}")
        print(f"Processing effect size model {loop-3}/{numberofcolumns1-5}")
        print(f"{'='*80}")
        
        print(f"Effect size column: {temp.columns[loop]}")
        print(f"  Min: {temp.iloc[:, loop].min():.6f}")
        print(f"  Max: {temp.iloc[:, loop].max():.6f}")
        print(f"  Mean: {temp.iloc[:, loop].mean():.6f}")
        print(f"  Std: {temp.iloc[:, loop].std():.6f}")
        print("  ✓ Using BETA (log-odds) scale for PLINK scoring")
        
        # Save: SNP, Effect Allele, Effect Size (no header)
        ordered_columns = [temp.columns[-1], temp.columns[1], temp.columns[loop]]
        temp[ordered_columns].to_csv(
            traindirec + os.sep + ldakmodel + "_ldak_gwas_final", 
            sep="\t", 
            header=False,
            index=False
        )
       
        # Calculate PRS for training data
        print("Calculating training PRS...")
        command = [
            "./plink",
            "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
            "--score", traindirec + os.sep + ldakmodel + "_ldak_gwas_final", "1", "2", "3",
            "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--allow-no-sex",
            "--out", traindirec+os.sep+Name+os.sep+trainfilename
        ]
        subprocess.run(command, capture_output=True)
        
        # Calculate PRS for test data
        print("Calculating test PRS...")
        command = [
            "./plink",
            "--bfile", folddirec+os.sep+testfilename+".clumped.pruned",
            "--score", traindirec + os.sep + ldakmodel + "_ldak_gwas_final", "1", "2", "3",
            "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--allow-no-sex",
            "--out", folddirec+os.sep+Name+os.sep+testfilename
        ]
        subprocess.run(command, capture_output=True)
        
        # Calculate PRS for validation data
        print("Calculating validation PRS...")
        command = [
            "./plink",
            "--bfile", valdirec+os.sep+valfilename,
            "--score", traindirec + os.sep + ldakmodel + "_ldak_gwas_final", "1", "2", "3",
            "--q-score-range", traindirec+os.sep+"range_list", traindirec+os.sep+"SNP.pvalue",
            "--extract", traindirec+os.sep+trainfilename+".valid.snp",
            "--allow-no-sex",
            "--out", traindirec+os.sep+Name+os.sep+valfilename
        ]
        subprocess.run(command, capture_output=True)
 
        # Evaluate PURE PRS (no model fitting!)
        evaluate_pure_prs_ldak(traindirec, newtrainfilename, numberofpca, ldakmodel, power, 
                               str(p1_val), str(p2_val), str(p3_val), str(c1_val), str(c2_val), str(c3_val), 
                               Name, pvaluefile, ldakmodel+"_Model_"+str(loop-3))

    print(f"\n{'='*80}")
    print(f"COMPLETED: {ldakmodel} with power {power}")
    print(f"{'='*80}")

# Main execution
powers = [-0.25]
ldakmodels = ["lasso", "lasso-sparse", "ridge", "bolt", "bayesr", "elastic"]
ldakmodels = ["lasso"]

result_directory = "LDAK-GWAS3"

create_directory(folddirec+os.sep+result_directory)

print("\n" + "="*80)
print("STARTING LDAK ANALYSIS - PURE PRS ONLY")
print("="*80)
print(f"Phenotype: {filedirec}")
print(f"Fold: {foldnumber}")
print(f"Models to test: {ldakmodels}")
print(f"Powers to test: {powers}")
print(f"Evaluation: PURE PRS AUC (no model fitting)")
print("="*80)

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
       
       # Test each LDAK model
       for ldakmodel in ldakmodels:
        for power in powers: 
         transform_ldak_data(folddirec, newtrainfilename, p, ldakmodel, power,
                           str(p1_val), str(p2_val), str(p3_val), 
                           str(c1_val), str(c2_val), str(c3_val),
                           result_directory, pvaluefile)

# Cleanup: keep only top 10 performing model files across ALL LDAK models
print("\n" + "="*80)
print("PERFORMING FINAL CLEANUP")
print("="*80)
 

print("\n" + "="*80)
print("✅ LDAK ANALYSIS COMPLETED!")
print("="*80)
print(f"Results saved in: {folddirec + os.sep + result_directory}")

# Display summary
if os.path.exists(folddirec + os.sep + result_directory + os.sep + "Results.csv"):
    results = pd.read_csv(folddirec + os.sep + result_directory + os.sep + "Results.csv")
    print("\n" + "="*80)
    print("SUMMARY: Best Pure PRS by Model")
    print("="*80)
    
    for model in ldakmodels:
        model_results = results[results['ldakmodel'] == model]
        if len(model_results) > 0:
            best_idx = model_results['Test_pure_prs'].idxmax()
            best_row = model_results.loc[best_idx]
            print(f"{model:15s}: Test AUC = {best_row['Test_pure_prs']:.4f}, "
                  f"Train AUC = {best_row['Train_pure_prs']:.4f}, "
                  f"Submodel: {best_row['ldaksubmodel']}")
    
    # Overall best across ALL LDAK models
    best_overall_idx = results['Test_pure_prs'].idxmax()
    best_overall = results.loc[best_overall_idx]
    
    print("\n" + "="*80)
    print("🏆 BEST OVERALL PRS ACROSS ALL LDAK MODELS")
    print("="*80)
    print(f"Model: {best_overall['ldakmodel']}")
    print(f"Submodel: {best_overall['ldaksubmodel']}")
    print(f"Power: {best_overall['ldakpower']}")
    print(f"P-value: {best_overall['pvalue']:.2e}")
    print(f"Test AUC: {best_overall['Test_pure_prs']:.4f}")
    print(f"Train AUC: {best_overall['Train_pure_prs']:.4f}")
    print("="*80)