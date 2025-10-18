#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import sys

# Get phenotype from command line
if len(sys.argv) < 2:
    print("Usage: python script.py <phenotype_name>")
    print("Example: python script.py Phenotype_1")
    sys.exit(1)

phenotype = sys.argv[1]

# Configuration
top_snp_counts = [ 500, 2000,3000,5000,10000]  # Number of top SNPs to extract
valdirec = "/data/ascher02/uqmmune1/CAGI7/Data/PRS/simulated/val"

print("="*80)
print(f"EXTRACTING TOP SNPs FOR: {phenotype}")
print(f"Top SNP counts: {top_snp_counts}")
print("="*80)

# Read GWAS file (same for all folds)
gwas_file = os.path.join(phenotype, f"{phenotype}.gwas")
if not os.path.exists(gwas_file):
    print(f"ERROR: GWAS file not found: {gwas_file}")
    sys.exit(1)

print(f"\nReading GWAS file: {gwas_file}")
gwasfile = pd.read_csv(gwas_file, sep=r'\s+')

# Check columns
print(f"GWAS columns: {list(gwasfile.columns)}")
print(f"Total SNPs in GWAS: {len(gwasfile)}")

# Ensure pval is numeric
gwasfile['pval'] = pd.to_numeric(gwasfile['pval'], errors='coerce')
gwasfile = gwasfile.dropna(subset=['pval'])

# Sort by p-value
gwasfile_sorted = gwasfile.sort_values('pval')

print(f"Valid SNPs after filtering: {len(gwasfile_sorted)}")
print(f"Minimum p-value: {gwasfile_sorted['pval'].min():.6e}")
print(f"Maximum p-value: {gwasfile_sorted['pval'].max():.6e}")

# Process each fold
for fold in range(5):
    fold_dir = os.path.join(phenotype, f"Fold_{fold}")
    
    print(f"\n{'='*80}")
    print(f"PROCESSING FOLD {fold}")
    print(f"{'='*80}")
    
    # Read phenotype files
    train_fam_file = os.path.join(fold_dir, "train_data.QC.clumped.pruned.fam")
    test_fam_file = os.path.join(fold_dir, "test_data.fam")
    
    if not os.path.exists(train_fam_file):
        print(f"Train FAM file not found: {train_fam_file}")
        continue
    if not os.path.exists(test_fam_file):
        print(f"Test FAM file not found: {test_fam_file}")
        continue
    
    train_fam = pd.read_csv(train_fam_file, sep=r'\s+', header=None,
                            names=['FID', 'IID', 'Father', 'Mother', 'Sex', 'PHENO'])
    test_fam = pd.read_csv(test_fam_file, sep=r'\s+', header=None,
                           names=['FID', 'IID', 'Father', 'Mother', 'Sex', 'PHENO'])
    
    # Convert phenotype: 1 (control) -> 0, 2 (case) -> 1
    y_train = train_fam['PHENO'].replace({1: 0, 2: 1}).values
    y_test = test_fam['PHENO'].replace({1: 0, 2: 1}).values
    
    print(f"\nTrain samples: {len(y_train)} (Cases: {y_train.sum()}, Controls: {len(y_train) - y_train.sum()})")
    print(f"Test samples: {len(y_test)} (Cases: {y_test.sum()}, Controls: {len(y_test) - y_test.sum()})")
    
    # Process each SNP count
    for snp_count in top_snp_counts:
        print(f"\n{'---'*20}")
        print(f"Extracting Top {snp_count} SNPs")
        print(f"{'---'*20}")
        
        # Select top N SNPs
        if snp_count > len(gwasfile_sorted):
            print(f"Warning: Only {len(gwasfile_sorted)} SNPs available, using all")
            top_snps = gwasfile_sorted
        else:
            top_snps = gwasfile_sorted.head(snp_count)
        
        actual_snp_count = len(top_snps)
        p_threshold = top_snps['pval'].max()
        
        print(f"Actual SNPs extracted: {actual_snp_count}")
        print(f"P-value threshold: {p_threshold:.6e}")
        
        # Create output directory
        output_dir = os.path.join(fold_dir, f"Top_{snp_count}_SNPs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save SNP list (using snpid column)
        snp_list_file = os.path.join(output_dir, "snp_list.txt")
        top_snps['snpid'].to_csv(snp_list_file, index=False, header=False)
        print(f"SNP list saved: {snp_list_file}")
        
        # Also save detailed SNP info
        snp_info_file = os.path.join(output_dir, "snp_info.txt")
        top_snps[['snpid', 'chr', 'pos', 'pval', 'effalt']].to_csv(snp_info_file, index=False, sep='\t')
        
        # Define BED file paths
        train_bed = os.path.join(fold_dir, "train_data.QC")
        test_bed = os.path.join(fold_dir, "test_data")
        val_bed = os.path.join(valdirec, "val")
        
        # Check if BED files exist
        if not os.path.exists(train_bed + ".bed"):
            print(f"ERROR: Train BED file not found: {train_bed}.bed")
            continue
        if not os.path.exists(test_bed + ".bed"):
            print(f"ERROR: Test BED file not found: {test_bed}.bed")
            continue
        
        # Extract and recode train data
        print(f"Extracting train data...")
        train_output = os.path.join(output_dir, "train_data")
        cmd_train = (f"./plink --bfile {train_bed} "
                    f"--extract {snp_list_file} "
                    f"--recodeA "
                    f"--out {train_output}")
        print(f"Running: {cmd_train}")
        os.system(cmd_train)
        
        # Extract and recode test data
        print(f"Extracting test data...")
        test_output = os.path.join(output_dir, "test_data")
        cmd_test = (f"./plink --bfile {test_bed} "
                   f"--extract {snp_list_file} "
                   f"--recodeA "
                   f"--out {test_output}")
        print(f"Running: {cmd_test}")
        os.system(cmd_test)
        
        # Extract and recode validation data
        if os.path.exists(val_bed + ".bed"):
            print(f"Extracting validation data...")
            val_output = os.path.join(output_dir, "val_data")
            cmd_val = (f"./plink --bfile {val_bed} "
                      f"--extract {snp_list_file} "
                      f"--recodeA "
                      f"--out {val_output}")
            print(f"Running: {cmd_val}")
            os.system(cmd_val)
        
        # Process train data
        train_raw_file = train_output + ".raw"
        if os.path.exists(train_raw_file):
            print(f"Processing train data from: {train_raw_file}")
            train_data = pd.read_csv(train_raw_file, sep=r'\s+')
            
            # Extract genotype matrix (skip first 6 columns: FID, IID, PAT, MAT, SEX, PHENOTYPE)
            X_train = train_data.iloc[:, 6:].values
            
            # Verify sample order matches
            train_data_iid = train_data['IID'].astype(str).values
            train_fam_iid = train_fam['IID'].astype(str).values
            
            if not np.array_equal(train_data_iid, train_fam_iid):
                print("Warning: Sample order mismatch in train data, reordering...")
                # Create mapping
                train_data['IID_str'] = train_data['IID'].astype(str)
                train_fam_sorted = train_fam.copy()
                train_fam_sorted['IID_str'] = train_fam_sorted['IID'].astype(str)
                
                # Reorder train_data to match train_fam
                train_data = train_data.set_index('IID_str').loc[train_fam_sorted['IID_str']].reset_index(drop=True)
                X_train = train_data.iloc[:, 6:].values
                y_train = train_fam_sorted['PHENO'].replace({1: 0, 2: 1}).values
            
            # Save X_train and y_train
            np.save(os.path.join(output_dir, "X_train.npy"), X_train)
            np.save(os.path.join(output_dir, "y_train.npy"), y_train)
            
            print(f"✓ Train data saved:")
            print(f"  X_train shape: {X_train.shape} (samples × SNPs)")
            print(f"  y_train shape: {y_train.shape}")
            print(f"  Cases: {y_train.sum()}, Controls: {len(y_train) - y_train.sum()}")
            print(f"  Files: X_train.npy, y_train.npy")
        else:
            print(f"✗ Train RAW file not created: {train_raw_file}")
        
        # Process test data
        test_raw_file = test_output + ".raw"
        if os.path.exists(test_raw_file):
            print(f"Processing test data from: {test_raw_file}")
            test_data = pd.read_csv(test_raw_file, sep=r'\s+')
            
            # Extract genotype matrix
            X_test = test_data.iloc[:, 6:].values
            
            # Verify sample order matches
            test_data_iid = test_data['IID'].astype(str).values
            test_fam_iid = test_fam['IID'].astype(str).values
            
            if not np.array_equal(test_data_iid, test_fam_iid):
                print("Warning: Sample order mismatch in test data, reordering...")
                test_data['IID_str'] = test_data['IID'].astype(str)
                test_fam_sorted = test_fam.copy()
                test_fam_sorted['IID_str'] = test_fam_sorted['IID'].astype(str)
                
                test_data = test_data.set_index('IID_str').loc[test_fam_sorted['IID_str']].reset_index(drop=True)
                X_test = test_data.iloc[:, 6:].values
                y_test = test_fam_sorted['PHENO'].replace({1: 0, 2: 1}).values
            
            # Save X_test and y_test
            np.save(os.path.join(output_dir, "X_test.npy"), X_test)
            np.save(os.path.join(output_dir, "y_test.npy"), y_test)
            
            print(f"✓ Test data saved:")
            print(f"  X_test shape: {X_test.shape} (samples × SNPs)")
            print(f"  y_test shape: {y_test.shape}")
            print(f"  Cases: {y_test.sum()}, Controls: {len(y_test) - y_test.sum()}")
            print(f"  Files: X_test.npy, y_test.npy")
        else:
            print(f"✗ Test RAW file not created: {test_raw_file}")
        
        # Process validation data
        val_raw_file = val_output + ".raw"
        if os.path.exists(val_raw_file):
            print(f"Processing validation data from: {val_raw_file}")
            val_data = pd.read_csv(val_raw_file, sep=r'\s+')
            
            # Extract genotype matrix (no phenotype labels for validation)
            X_val = val_data.iloc[:, 6:].values
            
            # Save X_val (no y_val since we don't have labels)
            np.save(os.path.join(output_dir, "X_val.npy"), X_val)
            
            # Save sample IDs for reference
            val_ids = val_data[['FID', 'IID']]
            val_ids.to_csv(os.path.join(output_dir, "val_sample_ids.txt"), 
                          index=False, sep='\t')
            
            print(f"✓ Validation data saved:")
            print(f"  X_val shape: {X_val.shape} (samples × SNPs)")
            print(f"  Files: X_val.npy, val_sample_ids.txt")
        else:
            print(f"✗ Validation RAW file not created: {val_raw_file}")
        
        # Create a summary file
        summary_file = os.path.join(output_dir, "README.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Top {snp_count} SNPs Dataset Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Phenotype: {phenotype}\n")
            f.write(f"Fold: {fold}\n")
            f.write(f"GWAS file: {gwas_file}\n")
            f.write(f"Requested SNPs: {snp_count}\n")
            f.write(f"Actual SNPs: {actual_snp_count}\n")
            f.write(f"P-value threshold: {p_threshold:.6e}\n\n")
            
            f.write("Files:\n")
            f.write("  - snp_list.txt: List of SNP IDs\n")
            f.write("  - snp_info.txt: SNP details (chr, pos, pval, effect)\n")
            f.write("  - X_train.npy: Training genotypes (numpy array)\n")
            f.write("  - y_train.npy: Training labels (0=control, 1=case)\n")
            f.write("  - X_test.npy: Test genotypes (numpy array)\n")
            f.write("  - y_test.npy: Test labels (0=control, 1=case)\n")
            f.write("  - X_val.npy: Validation genotypes (numpy array)\n")
            f.write("  - val_sample_ids.txt: Validation sample IDs\n\n")
            
            if os.path.exists(train_raw_file):
                f.write(f"Train samples: {X_train.shape[0]}\n")
                f.write(f"  Cases: {y_train.sum()}\n")
                f.write(f"  Controls: {len(y_train) - y_train.sum()}\n\n")
            
            if os.path.exists(test_raw_file):
                f.write(f"Test samples: {X_test.shape[0]}\n")
                f.write(f"  Cases: {y_test.sum()}\n")
                f.write(f"  Controls: {len(y_test) - y_test.sum()}\n\n")
            
            if os.path.exists(val_raw_file):
                f.write(f"Validation samples: {X_val.shape[0]}\n")
            
            f.write("\nColumn naming in .raw files:\n")
            f.write("  Columns 1-6: FID, IID, PAT, MAT, SEX, PHENOTYPE\n")
            f.write("  Columns 7+: SNP genotypes (0, 1, 2 for additive coding)\n")
        
        print(f"✓ Summary saved: {summary_file}")

print("\n" + "="*80)
print("✅ TOP SNP EXTRACTION COMPLETED")
print("="*80)
print(f"\nProcessed phenotype: {phenotype}")
print(f"GWAS file used: {gwas_file}")
print(f"SNP counts: {top_snp_counts}")
print(f"\nOutput structure:")
print(f"  {phenotype}/Fold_X/Top_Y_SNPs/")
print(f"    - X_train.npy, y_train.npy")
print(f"    - X_test.npy, y_test.npy")
print(f"    - X_val.npy, val_sample_ids.txt")
print(f"    - snp_list.txt, snp_info.txt")
print(f"    - README.txt")
print("\nUsage example:")
print("  import numpy as np")
print(f"  X_train = np.load('{phenotype}/Fold_0/Top_100_SNPs/X_train.npy')")
print(f"  y_train = np.load('{phenotype}/Fold_0/Top_100_SNPs/y_train.npy')")