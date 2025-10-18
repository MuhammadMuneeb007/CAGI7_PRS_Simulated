import os
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing the files
directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Submissions2")

print(f"Looking for files in: {directory}")
if os.path.exists(directory):
    print("Directory exists.")
    all_files = os.listdir(directory)
    print(f"All files in directory: {all_files}")
else:
    print("Directory does not exist!")

# Rename submission files
print("Renaming submission files...")
submission_pattern = os.path.join(directory, "Phenotype_*_Fold*_*_submission*")
submission_files = glob.glob(submission_pattern)

print(f"Found {len(submission_files)} files matching pattern")

for file in submission_files:
    filename = os.path.basename(file)
    print(f"\nProcessing: {filename}")
    
    # Extract phenotype number and fold number (works with any method name including hyphens and numbers)
    match = re.search(r'Phenotype_(\d+)_Fold(\d+)_[^_]+_submission', file)
    if match:
        phenotype_num = match.group(1)
        fold_num = match.group(2)
        
        print(f"  Matched - Phenotype: {phenotype_num}, Fold: {fold_num}")
        
        # Get file extension
        _, ext = os.path.splitext(file)
        
        # Create new filename
        new_name = f"Phenotype{phenotype_num}_submission{int(fold_num) + 1}{ext}"
        new_path = os.path.join(directory, new_name)
        
        # Rename file
        os.rename(file, new_path)
        print(f"  Renamed to: {new_name}")
        
        # Update content inside the file
        try:
            with open(new_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace submission0 with submission1 in the content
            updated_content = content.replace('submission0', 'submission1')
            
            if content != updated_content:
                with open(new_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"  Updated file content: replaced 'submission0' with 'submission1'")
        except Exception as e:
            print(f"  Warning: Could not update file content: {e}")
    else:
        print(f"  ERROR: No match for: {filename}")

# Fix any files that have submission0 in the name (change to submission1)
print("\n\nFixing submission0 files to submission1...")
submission0_pattern = os.path.join(directory, "Phenotype*_submission0*")
submission0_files = glob.glob(submission0_pattern)

print(f"Found {len(submission0_files)} files with submission0")

for file in submission0_files:
    filename = os.path.basename(file)
    new_filename = filename.replace('submission0', 'submission1')
    new_path = os.path.join(directory, new_filename)
    
    os.rename(file, new_path)
    print(f"Renamed: {filename} -> {new_filename}")
    
    # Update content inside the file
    try:
        with open(new_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        updated_content = content.replace('submission0', 'submission1')
        
        if content != updated_content:
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"  Updated file content: replaced 'submission0' with 'submission1'")
    except Exception as e:
        print(f"  Warning: Could not update file content: {e}")

# Merge summary tables
print("\nMerging summary tables...")
summary_pattern = os.path.join(directory, "Phenotype_*_Summary_Table.csv")
summary_files = glob.glob(summary_pattern)

if summary_files:
    dfs = []
    for file in summary_files:
        # Try to read with common delimiters
        try:
            df = pd.read_csv(file)
        except:
            try:
                df = pd.read_csv(file, sep='\t')
            except:
                df = pd.read_csv(file, sep=' ')
        
        # Add source file column
        df['Source_File'] = os.path.basename(file)
        dfs.append(df)
        print(f"Loaded: {os.path.basename(file)}")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged file
    output_file = os.path.join(directory, "All_Phenotypes_Summary_Table.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged summary saved to: {output_file}")
    print(f"Total rows: {len(merged_df)}")
    
    # Create plots
    print("\nCreating AUC comparison plots...")
    
    # Create simple heatmap
    plt.figure(figsize=(10, 12))
    
    # Prepare data for heatmap
    heatmap_data = merged_df[['Phenotype', 'Train_AUC', 'Test_AUC', 'Expected_Validation_AUC']].copy()
    
    # Extract numeric part for proper sorting
    heatmap_data['Phenotype_Num'] = heatmap_data['Phenotype'].str.extract(r'(\d+)').astype(int)
    heatmap_data = heatmap_data.sort_values('Phenotype_Num')
    
    # Create label with just Phenotype number
    heatmap_data['Label'] = 'Phenotype_' + heatmap_data['Phenotype_Num'].astype(str)
    heatmap_data = heatmap_data.set_index('Label')[['Train_AUC', 'Test_AUC', 'Expected_Validation_AUC']]
    
    # Create heatmap (phenotypes on Y-axis, AUC types on X-axis)
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.7, 
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'AUC'})
    plt.title('AUC Heatmap: Train, Test, and Expected Validation')
    plt.ylabel('Phenotype')
    plt.xlabel('AUC Type')
    plt.tight_layout()
    
    plot_file = os.path.join(directory, "AUC_Heatmap.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {plot_file}")
    plt.close()
    
    # Additional summary statistics
    print("\n=== AUC Summary Statistics ===")
    print(f"Train AUC - Mean: {merged_df['Train_AUC'].mean():.4f}, Std: {merged_df['Train_AUC'].std():.4f}")
    print(f"Test AUC - Mean: {merged_df['Test_AUC'].mean():.4f}, Std: {merged_df['Test_AUC'].std():.4f}")
    print(f"Expected Val AUC - Mean: {merged_df['Expected_Validation_AUC'].mean():.4f}, Std: {merged_df['Expected_Validation_AUC'].std():.4f}")
    
else:
    print("No summary table files found!")

print("\nDone!")
