import sys
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import pandas as pd

def process_one(i):
    dir_name = str(i)
    
    # Create the target directory name
    target_dir = f"Phenotype_{dir_name}"
    
    # Step 1: Create the directory
    try:
        os.makedirs(target_dir, exist_ok=True)
        print(f"✓ Created directory: {target_dir}")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return
    
    # Define source and destination file mappings
    file_mappings = [
        # (source_path, destination_filename)
        (f"CAGI7_PRS/simulated/training/{dir_name}_merged_with_phen.fam", 
         f"Phenotype_{dir_name}.fam"),
        
        (f"CAGI7_PRS/simulated/training/{dir_name}.phen", 
         f"Phenotype_{dir_name}.phen"),
        
        # (f"CAGI7_PRS/simulated/training/{dir_name}_merged_with_phen.bim", 
        #  f"Phenotype_{dir_name}.bim"),
        
        # (f"CAGI7_PRS/simulated/training/{dir_name}_merged_with_phen.bed", 
        #  f"Phenotype_{dir_name}.bed"),
        
        # (f"gwas/{dir_name}.gwas", 
        #  f"Phenotype_{dir_name}.gwas"),
    ]
    
    # Step 2-6: Copy all files
    for source_path, dest_filename in file_mappings:
        try:
            source = Path(source_path)
            destination = Path(target_dir) / dest_filename
            
            if not source.exists():
                print(f"⚠️  Warning: Source file not found: {source}")
                continue
            
            if dest_filename.endswith('.fam'):
                # Special handling for fam file: read, transform phenotype, write
                # Read FAM file with pandas using \s+ (one or more whitespace characters)
                df = pd.read_csv(source, sep=r'\s+', header=None, 
                                names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'])
                
                # Convert phenotypes: 1 -> 2, then -9 and 0 -> 1
                df['PHENO'] = df['PHENO'].astype(str)
                df.loc[df['PHENO'] == '1', 'PHENO'] = '2'
                df.loc[df['PHENO'].isin(['-9', '0']), 'PHENO'] = '1'
                
                # Write to destination with space separator, no header, no index
                df.to_csv(destination, sep='\t', header=False, index=False)
                print(f"✓ Modified and copied: {source} -> {destination}")
            else:
                shutil.copy2(source, destination)
                print(f"✓ Copied: {source} -> {destination}")
            
        except Exception as e:
            print(f"❌ Error copying {source_path}: {e}")
    
    print(f"\n✅ All operations completed for {target_dir}")

def main():
    # Use multiprocessing to parallelize the loop
    with multiprocessing.Pool() as pool:
        for _ in tqdm(pool.imap(process_one, range(1, 31)), total=30):
            pass

if __name__ == "__main__":
    main()