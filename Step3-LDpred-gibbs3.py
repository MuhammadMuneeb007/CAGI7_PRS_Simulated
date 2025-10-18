import os
import pandas as pd
import numpy as np
import sys
import glob
from scipy.stats import norm

filedirec = sys.argv[1]

def check_phenotype_is_binary_or_continous(filedirec):
    df = pd.read_csv(filedirec+os.sep+filedirec+'.fam',sep="\s+",header=None)
    column_values = df[5].unique()
    if len(set(column_values)) == 2:
        return "Binary"
    else:
        return "Continous"

# Read the GWAS file
GWAS = filedirec + os.sep + filedirec+".gz"
df = pd.read_csv(GWAS,compression="gzip",sep="\s+")

print("="*80)
print("PREPARING GWAS DATA FOR LDpred-gibbs")
print("="*80)

if check_phenotype_is_binary_or_continous(filedirec)=="Binary":
    print("\nProcessing Binary Phenotype...")
    
    if "BETA" in df.columns.to_list():
        # BETA already provided (log-odds scale)
        df["OR"] = np.exp(df["BETA"])
        
        # CRITICAL FIX: SE handling
        # Check if SE is a placeholder (constant value)
        unique_se = df['SE'].nunique()
        if unique_se <= 10:
            print(f"⚠️  SE has only {unique_se} unique values - calculating from P-values")
            # Calculate SE from P-values and BETA
            Z = np.sign(df["BETA"]) * norm.ppf(1 - df["P"].clip(lower=1e-300, upper=1-1e-16) / 2)
            df["SE"] = np.abs(df["BETA"]) / np.abs(Z.replace(0, np.nan))
            df["SE"].fillna(df["SE"].median(), inplace=True)
        # else: SE is already on log-scale, keep as-is
        
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'OR', 'INFO', 'MAF']]
    else:
        # OR provided, need to calculate BETA
        df["BETA"] = np.log(df["OR"].replace(0, np.nan))
        
        # Calculate SE from P-values if it's a placeholder
        unique_se = df['SE'].nunique()
        if unique_se <= 10:
            print(f"⚠️  SE has only {unique_se} unique values - calculating from P-values")
            Z = np.sign(df["BETA"]) * norm.ppf(1 - df["P"].clip(lower=1e-300, upper=1-1e-16) / 2)
            df["SE"] = np.abs(df["BETA"]) / np.abs(Z.replace(0, np.nan))
            df["SE"].fillna(df["SE"].median(), inplace=True)
        
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'OR', 'INFO', 'MAF']]
    
    print(f"BETA range: {df['BETA'].min():.6f} to {df['BETA'].max():.6f}")
    print(f"SE range: {df['SE'].min():.6f} to {df['SE'].max():.6f}")
    
    # CRITICAL: For LDpred, use BETA (log-odds), not OR
    df = df.rename(columns={
        'CHR':'CHR',
        'BP': 'POS',
        'SNP': 'SNP_ID',
        'A1': 'REF',
        'A2': 'ALT',
        'MAF': 'REF_FRQ',
        'P': 'PVAL',
        'BETA':'BETA',  # Use BETA, not OR
    })
    df = df[['CHR', 'POS', 'SNP_ID', 'REF', 'ALT', 'REF_FRQ', 'PVAL', 'BETA', 'SE', 'N']]

elif check_phenotype_is_binary_or_continous(filedirec)=="Continous":
    print("\nProcessing Continuous Phenotype...")
    
    if "BETA" in df.columns.to_list():
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    else:
        df["BETA"] = np.log(df["OR"])
        df["SE"] = df["SE"]/df["OR"]
        df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    
    df = df.rename(columns={
        'CHR':'CHR',
        'BP': 'POS',
        'SNP': 'SNP_ID',
        'A1': 'REF',
        'A2': 'ALT',
        'MAF': 'REF_FRQ',
        'P': 'PVAL',
        'BETA':'BETA',
    })
    df = df[['CHR', 'POS', 'SNP_ID', 'REF', 'ALT', 'REF_FRQ', 'PVAL', 'BETA', 'SE', 'N']]

N = df["N"].mean()

df.to_csv(filedirec + os.sep +filedirec+"_ldpredgibs.txt",sep="\t",index=False)
print("\n" + df.head().to_markdown())
print(f"Length of DataFrame: {len(df)}")

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

# UPDATED: Simplified results - PURE PRS ONLY
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                   "pvalue", "numberofpca", "ldradius", "ldfilename", "gibsfraction", "gibsburn", "gibsiterations",
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
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--allow-no-sex",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--out", traindirec+os.sep+newtrainfilename+".clumped.pruned"
    ]
    subprocess.run(command)
    
    command = [
        "./plink",
        "--make-bed",
        "--bfile", traindirec+os.sep+testfilename,
        "--indep-pairwise", p1_val, p2_val, p3_val,
        "--allow-no-sex",
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



def evaluate_pure_prs_gibbs(traindirec, newtrainfilename, p, radius, betafile, fraction, burn, iterations, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    """
    Evaluate PURE PRS performance without model fitting
    For CAGI submission - just raw PRS discrimination
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING PURE PRS - LDpred-gibbs")
    print(f"Beta file: {betafile}")
    print(f"LD radius: {radius}, Fraction: {fraction}, Burn: {burn}, Iterations: {iterations}")
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
                "ldradius": radius,
                "ldfilename": betafile,
                "gibsfraction": fraction,
                "gibsburn": burn,
                "gibsiterations": iterations,
                "Train_pure_prs": train_auc,
                "Test_pure_prs": test_auc,
            }, ignore_index=True)
            
        except Exception as e:
            continue

        # Save incrementally
        prs_result.to_csv(traindirec+os.sep+Name+os.sep+"Results.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"BEST RESULT FOR {betafile}")
    print(f"{'='*80}")
    print(f"Best Test AUC: {best_test_auc:.4f}")
    print(f"Best Train AUC: {best_train_auc:.4f}")
    print(f"Best p-value: {best_threshold:.2e}")
    print(f"{'='*80}")
    
    return

prs_result = pd.DataFrame()

def transform_plink_data(traindirec, newtrainfilename, p, radius, ldpredmodel, fraction, burn, iterations, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile):
    
    print(f"\n{'='*80}")
    print(f"Processing LDpred-gibbs")
    print(f"LD radius: {radius}, Fraction: {fraction}, Burn-in: {burn}, Iterations: {iterations}")
    print(f"{'='*80}")
    
    os.system("awk "+"\'"+"{print $3,$8}"+"\'"+" ./"+filedirec+os.sep+filedirec+".txt >  ./"+traindirec+os.sep+"SNP.pvalue")
    
    # Delete files from previous iterations
    import shutil
    
    output_file = os.path.join(traindirec, "output_file.h5")
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing file: {output_file}")
        
    import glob
    file_pattern = os.path.join(traindirec, 'ld.h5_LDpred_*')
    file_list = glob.glob(file_pattern)
    file_list.append(traindirec+os.sep+'ld.h5_LDpred-inf.txt')
    
    for file_path in file_list:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
            
    file_pattern = os.path.join(traindirec, 'inf_*')
    file_list = glob.glob(file_pattern)
    for file_path in file_list:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
            
    # CRITICAL FIX: Always use BETA (log-odds) for LDpred
    # LDpred expects and outputs log-odds scale effects
    if check_phenotype_is_binary_or_continous(filedirec)=="Binary":
        eff_type = "LOGOR"  # FIXED: Use LOGOR not OR
        eff = "BETA"        # FIXED: Use BETA not OR
    else:
        eff_type = "LOGOR"
        eff = "BETA"

    gwas_file = filedirec + os.sep +filedirec+"_ldpredgibs.txt"
    bim_file = traindirec + os.sep + newtrainfilename+".clumped.pruned.bim"

    df = pd.read_csv(gwas_file, sep="\s+")
    bim = pd.read_csv(bim_file, delim_whitespace=True, header=None)

    print(f"GWAS SNPs: {len(df)}")
    print(f"BIM SNPs: {len(bim)}")

    # Match SNPs
    bim['match'] = bim[0].astype(str) + "_" + bim[3].astype(str) + "_" + bim[4].astype(str) + "_" + bim[5].astype(str)
    df['match'] = df['CHR'].astype(str) + "_" + df['POS'].astype(str) + "_" + df['REF'].astype(str) + "_" + df['ALT'].astype(str)

    df.drop_duplicates(subset='match', inplace=True)
    bim.drop_duplicates(subset='match', inplace=True)

    df = df[df['match'].isin(bim['match'].values)]
    bim = bim[bim['match'].isin(df['match'].values)]

    print(f"Matched SNPs: {len(df)}")
    
    del df["match"]
    del bim["match"]
    df.to_csv(traindirec+os.sep+filedirec+".ldpred",sep="\t",index=None)
    bim.to_csv(traindirec + os.sep + "commonsnps.txt",sep="\t",index=None)
    
    command = [
        './plink',
        '--bfile', traindirec+os.sep+newtrainfilename,
        '--extract', traindirec + os.sep + "commonsnps.txt",
        '--make-bed',
        '--allow-no-sex',
        '--chr','1-22',
        '--out', traindirec+os.sep+newtrainfilename+".clumped.pruned"
    ]
    subprocess.run(command)
    
    print("\n" + "="*80)
    print("Running LDpred coord")
    print("="*80)
    
    command = [
        "ldpred", "coord",
        "--gf", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--ssf", traindirec+os.sep+filedirec+".ldpred",
        "--out", traindirec+os.sep+"output_file.h5",
        "--N", str(int(N)),
        "--eff_type", eff_type,  # LOGOR for binary
        "--maf", "0.01",
        "--rs", "SNP_ID",
        "--A1", "REF",
        "--A2", "ALT",
        "--pos", "POS",
        "--chr", "CHR",
        "--pval", "PVAL",
        "--eff", eff,  # BETA column
    ]
    print(" ".join(command))
    subprocess.run(command)

    if ldpredmodel =="gibbs":
        print("\n" + "="*80)
        print("Running LDpred gibbs")
        print("="*80)
        
        command = [
            'ldpred', 'gibbs',
            '--cf', traindirec+os.sep+"output_file.h5",
            '--ldr', str(radius),
            '--n-burn-in',str(burn),
            '--n-iter',str(iterations),
            '--f',str(fraction),
            '--ldf', traindirec+os.sep+'inf_',
            '--out', traindirec+os.sep+"ld.h5",
        ]
        subprocess.run(command)
         
        import glob
        file_pattern = os.path.join(traindirec, 'ld.h5_LDpred_*')
        file_list = glob.glob(file_pattern)

        for betafile in file_list:
            print(f"\n{'='*80}")
            print(f"Processing: {betafile}")
            print(f"{'='*80}")
            
            temp = pd.read_csv(betafile,sep="\s+")
            
            if len(temp)<2:
                print("Skipping - insufficient data")
                continue
            
            # CRITICAL FIX: DO NOT exponentiate!
            # LDpred-gibbs outputs BETA (log-odds) which is correct for PLINK --score
            print(f"LDpred effect statistics:")
            print(f"  Min: {temp.iloc[:, 6].min():.6f}")
            print(f"  Max: {temp.iloc[:, 6].max():.6f}")
            print(f"  Mean: {temp.iloc[:, 6].mean():.6f}")
            print(f"  Std: {temp.iloc[:, 6].std():.6f}")
            print("✓ Using BETA (log-odds) scale for PLINK scoring")
            
            # Save as-is (BETA scale)
            temp.iloc[:,[2,3,6]].to_csv(traindirec+os.sep+"LDpred_gibbs_gwas",sep="\t",index=False)
            
            # Calculate PRS for training data
            print("Calculating training PRS...")
            command = [
                "./plink",
                "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
                "--score", traindirec+os.sep+"LDpred_gibbs_gwas", "1", "2", "3", "header",
                "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
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
                "--score", traindirec+os.sep+"LDpred_gibbs_gwas", "1", "2", "3", "header",
                "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
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
                "--score", traindirec+os.sep+"LDpred_gibbs_gwas", "1", "2", "3", "header",
                "--q-score-range", traindirec+os.sep+"range_list",traindirec+os.sep+"SNP.pvalue",
                "--extract", traindirec+os.sep+trainfilename+".valid.snp",
                "--allow-no-sex",
                "--out", traindirec+os.sep+Name+os.sep+valfilename
            ]
            subprocess.run(command, capture_output=True)

            # Evaluate PURE PRS (no model fitting!)
            evaluate_pure_prs_gibbs(traindirec, newtrainfilename, p, radius, os.path.basename(betafile), 
                                   fraction, burn, iterations, p1_val, p2_val, p3_val, c1_val, c2_val, c3_val, Name, pvaluefile)

# Main execution
gibssamplerfractions = [0.1]
gibssamplerburn = [5]
gibssampleriterations = [6]
ldradius = [4]
ldpredmodels = ['gibbs']

result_directory = "LDpred-gibbs3"

print("\n" + "="*80)
print("STARTING LDpred-gibbs ANALYSIS - PURE PRS ONLY")
print("="*80)
print(f"Phenotype: {filedirec}")
print(f"Fold: {foldnumber}")
print(f"Gibbs parameters:")
print(f"  Fractions: {gibssamplerfractions}")
print(f"  Burn-in iterations: {gibssamplerburn}")
print(f"  Total iterations: {gibssampleriterations}")
print(f"  LD radius: {ldradius}")
print(f"Evaluation: PURE PRS AUC (no model fitting)")
print("="*80 + "\n")

create_directory(folddirec+os.sep+result_directory)

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
       
       # Test each gibbs parameter combination
       for radius in ldradius:
        for ldpredmodel in ldpredmodels:
         for fraction in gibssamplerfractions:
          for burn in gibssamplerburn:
           for iterations in gibssampleriterations:
            transform_plink_data(folddirec, newtrainfilename, p, radius, ldpredmodel, fraction, burn, iterations, 
                                str(p1_val), str(p2_val), str(p3_val), str(c1_val), str(c2_val), str(c3_val), 
                                result_directory, pvaluefile)

# Cleanup: keep only top 10 performing model files
print("\n" + "="*80)
print("PERFORMING FINAL CLEANUP")
print("="*80)

print("\n" + "="*80)
print("✅ LDpred-gibbs ANALYSIS COMPLETED!")
print("="*80)
print(f"Results saved in: {folddirec + os.sep + result_directory}")

# Display summary if results exist
if os.path.exists(folddirec + os.sep + result_directory + os.sep + "Results.csv"):
    results = pd.read_csv(folddirec + os.sep + result_directory + os.sep + "Results.csv")
    if len(results) > 0:
        print("\n" + "="*80)
        print("SUMMARY: Best Pure PRS by Beta File")
        print("="*80)
        
        # Group by beta file
        unique_files = results['ldfilename'].unique()
        
        for betafile in unique_files[:10]:  # Show top 10
            file_results = results[results['ldfilename'] == betafile]
            best_idx = file_results['Test_pure_prs'].idxmax()
            best_row = file_results.loc[best_idx]
            
            print(f"{betafile}:")
            print(f"  Test AUC = {best_row['Test_pure_prs']:.4f}, Train AUC = {best_row['Train_pure_prs']:.4f}")
        
        # Overall best across ALL beta files
        best_overall_idx = results['Test_pure_prs'].idxmax()
        best_overall = results.loc[best_overall_idx]
        
        print("\n" + "="*80)
        print("🏆 BEST OVERALL PRS ACROSS ALL BETA FILES")
        print("="*80)
        print(f"Beta file: {best_overall['ldfilename']}")
        print(f"LD radius: {best_overall['ldradius']}")
        print(f"Fraction: {best_overall['gibsfraction']}")
        print(f"Burn-in: {best_overall['gibsburn']}")
        print(f"Iterations: {best_overall['gibsiterations']}")
        print(f"P-value threshold: {best_overall['pvalue']:.2e}")
        print(f"Test AUC: {best_overall['Test_pure_prs']:.4f}")
        print(f"Train AUC: {best_overall['Train_pure_prs']:.4f}")
        print("="*80)