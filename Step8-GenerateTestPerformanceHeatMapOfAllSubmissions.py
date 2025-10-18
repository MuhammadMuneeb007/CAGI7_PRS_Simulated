import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re

# Set the base directory
base_dir = Path(r"c:\Users\kl\Desktop\The University of Queensland\AnnotationChallenge\PRSChallenge")

# Find all Submission directories
submission_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Submissions")],
                        key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', x.name)])

# Dictionary to store data: {submission_name: {phenotype: test_auc}}
data_dict = {}

# Read each submission's summary table
for submission_dir in submission_dirs:
    submission_name = submission_dir.name
    summary_file = submission_dir / "All_Phenotypes_Summary_Table.csv"
    
    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            
            # Extract phenotype and test AUC columns
            # Adjust column names based on your actual CSV structure
            if 'Phenotype' in df.columns and 'Test_AUC' in df.columns:
                phenotype_auc = dict(zip(df['Phenotype'], df['Test_AUC']))
            elif 'phenotype' in df.columns and 'test_auc' in df.columns:
                phenotype_auc = dict(zip(df['phenotype'], df['test_auc']))
            else:
                # Try to find columns containing these keywords
                phenotype_col = [col for col in df.columns if 'phenotype' in col.lower()][0]
                test_auc_col = [col for col in df.columns if 'test' in col.lower() and 'auc' in col.lower()][0]
                phenotype_auc = dict(zip(df[phenotype_col], df[test_auc_col]))
            
            data_dict[submission_name] = phenotype_auc
            print(f"Loaded {submission_name}: {len(phenotype_auc)} phenotypes")
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")
    else:
        print(f"File not found: {summary_file}")

# Convert to DataFrame for heatmap
df_heatmap = pd.DataFrame(data_dict)

# Sort by submission name alphanumerically
df_heatmap = df_heatmap.reindex(sorted(df_heatmap.columns, 
                                       key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', x)]), 
                                axis=1)

# Sort phenotypes alphanumerically
df_heatmap = df_heatmap.reindex(sorted(df_heatmap.index, 
                                       key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(x))]))

# Create the heatmap
plt.figure(figsize=(12, max(8, len(df_heatmap) * 0.4)))
sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='viridis', 
            cbar_kws={'label': 'Test AUC'}, 
            linewidths=0.5, linecolor='gray')

plt.xlabel('Submissions', fontsize=12)
plt.ylabel('Phenotypes', fontsize=12)
plt.title('Test AUC Performance Heatmap Across Submissions', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the heatmap
output_file = base_dir / "Test_Performance_Heatmap.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_file}")

# Also save the data as CSV
csv_output = base_dir / "Test_Performance_Matrix.csv"
df_heatmap.to_csv(csv_output)
print(f"Data matrix saved to: {csv_output}")

plt.savefig("asdf.png")
