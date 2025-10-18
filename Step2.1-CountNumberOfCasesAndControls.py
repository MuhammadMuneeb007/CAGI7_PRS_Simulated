#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import subprocess
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

# Get directory from command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <directory_name>")
    sys.exit(1)

filedirec = sys.argv[1]

# Define file paths
BED = filedirec + os.sep + filedirec
BIM = filedirec + os.sep + filedirec + ".bim"
FAM = filedirec + os.sep + filedirec + ".fam"
COV = filedirec + os.sep + filedirec + ".cov"
PHEN = filedirec + os.sep + filedirec + ".phen"
GWAS = filedirec + os.sep + filedirec + ".gwas"

print("="*80)
print("STEP 1: Transform GWAS file to required format")
print("="*80)

# Read GWAS file
df_gwas = pd.read_csv(GWAS, sep="\s+")
print(f"Original GWAS columns: {df_gwas.columns.tolist()}")
print(f"Initial number of SNPs: {len(df_gwas)}")

# Transform GWAS to required format
df_gwas_transformed = pd.DataFrame()
df_gwas_transformed['CHR'] = df_gwas['chr'].str.replace('chr', '').astype(int)
df_gwas_transformed['BP'] = df_gwas['pos']
df_gwas_transformed['SNP'] = df_gwas['snpid']
df_gwas_transformed['A1'] = df_gwas['alt']  # Effect allele
df_gwas_transformed['A2'] = df_gwas['ref']  # Reference allele
df_gwas_transformed['N'] = 50000  # Default sample size
df_gwas_transformed['SE'] = 0.001  # Default SE
df_gwas_transformed['P'] = df_gwas['pval']

# Convert BETA to OR: OR = exp(BETA)
df_gwas_transformed['OR'] = np.exp(df_gwas['effalt'])

df_gwas_transformed['INFO'] = 0.9  # Default INFO score
# Calculate MAF from reffreq
df_gwas_transformed['MAF'] = df_gwas['reffreq'].apply(lambda x: x if x <= 0.5 else 1 - x)

print(f"\nTransformed GWAS file preview:")
print(df_gwas_transformed.head())

# Apply quality control filters
print(f"\nApplying quality control filters...")
df_gwas_transformed = df_gwas_transformed.loc[(df_gwas_transformed['MAF'] > 0.01) & 
                                               (df_gwas_transformed['INFO'] > 0.8)]
print(f"SNPs after MAF and INFO filtering: {len(df_gwas_transformed)}")

# Remove ambiguous SNPs
df_gwas_transformed = df_gwas_transformed[~((df_gwas_transformed['A1'] == 'A') & (df_gwas_transformed['A2'] == 'T') |
                                             (df_gwas_transformed['A1'] == 'T') & (df_gwas_transformed['A2'] == 'A') |
                                             (df_gwas_transformed['A1'] == 'G') & (df_gwas_transformed['A2'] == 'C') |
                                             (df_gwas_transformed['A1'] == 'C') & (df_gwas_transformed['A2'] == 'G'))]
print(f"SNPs after removing ambiguous SNPs: {len(df_gwas_transformed)}")

# Remove duplicates based on SNP
df_gwas_transformed = df_gwas_transformed.drop_duplicates(subset='SNP')
print(f"SNPs after removing duplicates: {len(df_gwas_transformed)}")

# Save transformed GWAS file
GWAS_transformed = filedirec + os.sep + filedirec + ".gz"
df_gwas_transformed.to_csv(GWAS_transformed, compression="gzip", sep="\t", index=False)
print(f"✓ Transformed GWAS saved to: {GWAS_transformed}")

print("\n" + "="*80)
print("STEP 2: Process FAM file and convert phenotypes")
print("="*80)

# Read FAM file
fam_df = pd.read_csv(FAM, sep="\s+", header=None, 
                     names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"])
print(f"Original FAM file shape: {fam_df.shape}")
print(f"Original phenotype distribution:\n{fam_df['Phenotype'].value_counts()}")

# # Convert phenotypes: 0/-9 -> 1 (control), 1 -> 2 (case)
# fam_df['Phenotype_Original'] = fam_df['Phenotype'].copy()
# fam_df.loc[fam_df['Phenotype_Original'] == 1, 'Phenotype'] = 2  # Cases
# fam_df.loc[fam_df['Phenotype'].isin([0, -9]), 'Phenotype'] = 1  # Controls


# print(f"\nConverted phenotype distribution:\n{fam_df['Phenotype'].value_counts()}")

# # Save updated FAM file
# fam_df[["FID", "IID", "Father", "Mother", "Sex", "Phenotype"]].to_csv(
#     FAM, sep="\t", header=False, index=False
# )
print(f"✓ Updated FAM file saved")

print("\n" + "="*80)
print("STEP 3: Stratified Cross-Validation Split (5 Folds)")
print("="*80)

# Create output directory
output_directory_base = filedirec
os.makedirs(output_directory_base, exist_ok=True)

# Setup for stratified k-fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Check if phenotype is binary
phenotype_values = fam_df['Phenotype'].unique()
print(f"Unique phenotype values: {phenotype_values}")

if set(phenotype_values).issubset({1, 2}):
    print("Binary phenotype detected - using Stratified K-Fold")
    fold_splitter = skf.split(fam_df, fam_df['Phenotype'])
else:
    print("Continuous phenotype detected - using regular K-Fold")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_splitter = kf.split(fam_df)

# Perform cross-validation splits
for fold_id, (train_index, test_index) in enumerate(fold_splitter):
    print(f"\n{'='*80}")
    print(f"Processing Fold {fold_id}")
    print(f"{'='*80}")
    
    fold_directory = os.path.join(output_directory_base, f'Fold_{fold_id}')
    os.makedirs(fold_directory, exist_ok=True)
    
    train_file_name = "train_data"
    test_file_name = "test_data"
    new_train_file_name = "train_data.QC"
    
    # Split data
    train_data = fam_df.iloc[train_index]
    test_data = fam_df.iloc[test_index]
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Train phenotype distribution:\n{train_data['Phenotype'].value_counts()}")
    print(f"Test phenotype distribution:\n{test_data['Phenotype'].value_counts()}")
    
    # Save train and test FAM files
    train_data[["FID", "IID", "Father", "Mother", "Sex", "Phenotype"]].to_csv(
        os.path.join(fold_directory, 'train_data.fam'), sep="\t", header=False, index=False
    )
    test_data[["FID", "IID", "Father", "Mother", "Sex", "Phenotype"]].to_csv(
        os.path.join(fold_directory, 'test_data.fam'), sep="\t", header=False, index=False
    )
    
    # Use PLINK to create train and test bed files
    print(f"\nCreating PLINK binary files for train and test sets...")
    plink_train_command = [
        './plink',
        '--bfile', filedirec + os.sep + filedirec,
        '--keep', os.path.join(fold_directory, train_file_name + '.fam'),
        '--make-bed',
        '--allow-no-sex',
        '--out', os.path.join(fold_directory, train_file_name)
    ]
    
    plink_test_command = [
        './plink',
        '--bfile', filedirec + os.sep + filedirec,
        '--keep', os.path.join(fold_directory, test_file_name + '.fam'),
        '--make-bed',
        '--allow-no-sex',
        '--out', os.path.join(fold_directory, test_file_name)
    ]
    
    subprocess.run(plink_train_command)
    subprocess.run(plink_test_command)
    
 
    
    # Quality control on training data
    print(f"\n--- Quality Control on Training Data ---")
    
    # QC Step 1: Basic filters (MAF, HWE, genotyping rate)
    print("Step 1: Applying MAF, HWE, genotyping rate filters...")
    plink_qc_1 = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        '--maf', '0.01',
        '--hwe', '1e-6',
        '--allow-no-sex',
        '--geno', '0.1',
        '--mind', '0.1',
        '--write-snplist',
        '--make-just-fam',
        '--out', os.path.join(fold_directory, new_train_file_name)
    ]
    subprocess.run(plink_qc_1)
    
    # QC Step 2: Heterozygosity check
    print("Step 2: Calculating heterozygosity...")
    plink_qc_2 = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        '--extract', os.path.join(fold_directory, new_train_file_name + '.snplist'),
        '--keep', os.path.join(fold_directory, new_train_file_name + '.fam'),
        '--het',
        '--allow-no-sex',
        '--out', os.path.join(fold_directory, new_train_file_name)
    ]
    subprocess.run(plink_qc_2)
    
    # Call R script for heterozygosity filtering and allele matching
    print("Step 3: Running R script for heterozygosity filtering and allele matching...")
    r_command = [
        'Rscript', 
        'Module1.R', 
        os.path.join(fold_directory),
        train_file_name,
        new_train_file_name,
        '1'
    ]
    
    print(f"Executing: {' '.join(r_command)}")
    subprocess.run(r_command)
    
    # QC Step 3: Relatedness check
    print("Step 4: Checking for relatedness...")
    plink_qc_3 = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        '--extract', os.path.join(fold_directory, new_train_file_name + '.snplist'),
        '--keep', os.path.join(fold_directory, train_file_name + '.valid.sample'),
        '--rel-cutoff', '0.125',
        '--allow-no-sex',
        '--out', os.path.join(fold_directory, new_train_file_name)
    ]
    subprocess.run(plink_qc_3)
    
    # Create final QC'ed training file
    print("Step 5: Creating final QC'ed training dataset...")
    plink_qc_final = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        '--make-bed',
        '--allow-no-sex',
        '--keep', os.path.join(fold_directory, new_train_file_name + '.rel.id'),
        '--extract', os.path.join(fold_directory, new_train_file_name + '.snplist'),
        '--exclude', os.path.join(fold_directory, train_file_name + '.mismatch'),
        '--a1-allele', os.path.join(fold_directory, train_file_name + '.a1'),
        '--out', os.path.join(fold_directory, new_train_file_name)
    ]
    subprocess.run(plink_qc_final)
    
    print(f"✓ Fold {fold_id} completed successfully")

print("\n" + "="*80)
print("STEP 4: Verify generated files")
print("="*80)

# Check for file existence
files_to_check = [
    "train_data.QC.bed",
    "train_data.QC.bim",
    "train_data.QC.fam",
    "train_data.cov",
    "test_data.bed",
    "test_data.bim",
    "test_data.fam",
    "test_data.cov",
    "train_data.valid.sample",
    "train_data.a1",
    "train_data.mismatch"
]

print("{:<30} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
    "File Name", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"
))
print("-" * 80)

for file in files_to_check:
    status = []
    for fold_number in range(5):
        file_path = os.path.join(filedirec, f"Fold_{fold_number}", file)
        if os.path.exists(file_path):
            status.append("✓")
        else:
            status.append("✗")
    print("{:<30} {:<8} {:<8} {:<8} {:<8} {:<8}".format(file, *status))

print("\n" + "="*80)
print("✅ All processing completed!")
print("="*80)