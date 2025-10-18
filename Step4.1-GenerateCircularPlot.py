import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set publication-quality parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Read the data
data = pd.read_csv("Results/Summary_Table.csv")

print("Data Overview:")
print(data.head())
print("\nData Shape:", data.shape)

# Find best method for each phenotype
best_methods = []
for phenotype in data['Phenotype'].unique():
    pheno_data = data[data['Phenotype'] == phenotype]
    best_idx = pheno_data['General_Test_Performance'].idxmax()
    best_method = pheno_data.loc[best_idx, 'Method']
    best_performance = pheno_data.loc[best_idx, 'General_Test_Performance']
    
    best_methods.append({
        'Phenotype': phenotype,
        'Best_Method': best_method,
        'Performance': best_performance
    })

results = pd.DataFrame(best_methods)

# Sort by method and then by phenotype for grouping
results = results.sort_values(by=['Best_Method', 'Phenotype'])

print("\n" + "="*60)
print("BEST METHOD FOR EACH PHENOTYPE:")
print("="*60)
print(results)

# Save results
results.to_csv("Best_Methods_Per_Phenotype.csv", index=False)

# ============= CIRCULAR BAR CHART - PUBLICATION QUALITY =============

def get_label_rotation(angle, offset):
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    padding = 6
    for angle, value, label in zip(angles, values, labels):
        rotation, alignment = get_label_rotation(angle, offset)
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor",
            size=9,
            fontweight='normal'
        )

# Prepare data for circular plot
OFFSET = np.pi / 2
VALUES = (results["Performance"] * 100).values  # Convert to percentage scale
LABELS = results["Performance"].round(3).astype(str) + " | " + results["Phenotype"]
LABELS = LABELS.values
GROUP = results["Best_Method"].values

# Group by method
grouped = results.groupby('Best_Method')
GROUPS_SIZE = []
unique_methods = []
for name, group in grouped:
    GROUPS_SIZE.append(len(group))
    unique_methods.append(name)

print(f"\nGroups: {unique_methods}")
print(f"Group sizes: {GROUPS_SIZE}")

# Calculate angles
PAD = 3
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

# Calculate indices for bars
offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Create figure with higher DPI
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"}, dpi=300)
ax.set_theta_offset(OFFSET)
ax.set_ylim(-100, 110)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Professional color palette (colorblind-friendly)
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = [color_palette[i % len(color_palette)] for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

# Plot bars with professional styling
bars = ax.bar(ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, 
              edgecolor="white", linewidth=1.5, alpha=0.85)

# Add labels
add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

# Add group labels and reference lines with improved styling
offset = 0 
for i, (group, size) in enumerate(zip(unique_methods, GROUPS_SIZE)):
    # Group separator line (darker and thicker)
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=100)
    ax.plot(x1, [-8] * 100, color="#2d2d2d", linewidth=2)
    
    # Group label with background
    label_angle = np.mean(x1)
    ax.text(
        label_angle, -40, f"{group}\n(n={size})", 
        color="#2d2d2d", fontsize=11, 
        fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color_palette[i % len(color_palette)], 
                  alpha=0.2, edgecolor='none')
    )
    
    # Reference lines at 20, 40, 60, 80 with subtle styling
    x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=100)
    for ref_val in [20, 40, 60, 80]:
        ax.plot(x2, [ref_val] * 100, color="#d0d0d0", lw=1.0, linestyle='--', alpha=0.6)
    
    # Add percentage labels for reference lines (only on first group)
    if offset == 0:
        for ref_val in [20, 40, 60, 80]:
            ax.text(ANGLES[offset], ref_val, f'{ref_val}%', 
                   color="#666666", fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                            alpha=0.8, edgecolor='none'))
    
    offset += size + PAD

# Add title with better positioning
plt.text(0.5, 1.05, 'Best Method Performance by Phenotype', 
         transform=ax.transAxes, fontsize=16, fontweight='bold', 
         ha='center', va='bottom')
plt.text(0.5, 1.02, 'Grouped by Optimal Method (General Test Performance)', 
         transform=ax.transAxes, fontsize=11, style='italic',
         ha='center', va='bottom', color='#555555')

plt.tight_layout()

# Save in multiple formats for publication
plt.savefig('circular_bar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('circular_bar_chart.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('circular_bar_chart.svg', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('circular_bar_chart.tiff', dpi=300, bbox_inches='tight', facecolor='white')

print("\n✓ Saved publication-quality figures:")
print("  - circular_bar_chart.png (300 DPI)")
print("  - circular_bar_chart.pdf (vector)")
print("  - circular_bar_chart.svg (vector)")
print("  - circular_bar_chart.tiff (300 DPI)")

plt.show()

# Create supplementary table with statistics
print("\n" + "="*60)
print("SUPPLEMENTARY TABLE:")
print("="*60)

# Calculate additional statistics
summary_table = results.copy()
summary_table['Performance_Percent'] = (summary_table['Performance'] * 100).round(2)

# Add method counts
method_counts = results['Best_Method'].value_counts()
summary_table['Method_Frequency'] = summary_table['Best_Method'].map(method_counts)

# Sort for publication
summary_table = summary_table.sort_values(by=['Best_Method', 'Performance'], ascending=[True, False])

print(summary_table.to_string(index=False))
summary_table.to_csv("Supplementary_Table_S1.csv", index=False)
print("\n✓ Saved: Supplementary_Table_S1.csv")

print("\n" + "="*60)
print("PUBLICATION-QUALITY ANALYSIS COMPLETE!")
print("="*60)