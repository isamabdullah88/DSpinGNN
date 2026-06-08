import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Parse the Custom Data Format
angles_list = []
j_list = []

# Replace 'your_data_file.txt' with the actual name of your saved file
with open('Simulations/DOCheckpoints-Full-Exchange-DataSet1-1/biaxial_5x5_T5K_Amp0.3A_1000steps/data.txt', 'r') as file:
    # Read the first line to skip the header (Step Angles Exchange)
    header = file.readline()
    
    for line in file:
        # Split the line by whitespace
        parts = line.strip().split()
        
        # Ensure the line actually has all three components before parsing
        if len(parts) >= 4:
            step = parts[0]
            
            # Split the comma-separated strings into lists of floats
            try:
                angles = []
                exchanges = []
                for a, j in zip(parts[2].split(','), parts[3].split(',')):
                    if float(a) > 10.0:
                        angles.append(float(a))
                        exchanges.append(float(j))
                
                # Safety check to ensure they pair up perfectly
                if len(angles) == len(exchanges):
                    angles_list.extend(angles)
                    j_list.extend(exchanges)
                else:
                    print(f"Warning: Mismatch at Step {step} (Angles: {len(angles)}, Exchange: {len(exchanges)})")
            except ValueError:
                # Skips lines where conversion to float fails (e.g., trailing commas or empty values)
                continue

# Create the clean DataFrame
df = pd.DataFrame({
    'Bond_Angle': angles_list,
    'Local_J': j_list
})

print(f"Successfully loaded {len(df)} raw data points!")

# ==========================================
# --- UNIFORM SAMPLING LOGIC STARTS HERE ---
# ==========================================

# Define how many "buckets" we want across the angle range
num_bins = 40 

# Group the data into these bins based on the Bond_Angle
df['angle_bin'] = pd.cut(df['Bond_Angle'], bins=num_bins)

# Maximum points to keep per bin
max_points_per_bin = 10 

# Sample the data to flatten the heavy 90-degree cluster
sampled_df = df.groupby('angle_bin', observed=False).apply(
    lambda x: x.sample(n=min(len(x), max_points_per_bin), random_state=42)
).reset_index(drop=True)

# Clean up by dropping the temporary bin column
sampled_df = sampled_df.drop(columns=['angle_bin'])

print(f"Uniformly sampled points ready for plotting: {len(sampled_df)}")

# ==========================================
# --- PRB PUBLICATION PLOT CONFIGURATION ---
# ==========================================

# APS / PRB Styling Dictionary (Strictly enforced)
_PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # Matches Times serif styling for math
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",      # PRB requires inward pointing ticks
    "ytick.direction": "in",
    "xtick.top": True,            # Full bounding box ticks
    "ytick.right": True,
    "axes.spines.top": True,      # Full bounding box spines
    "axes.spines.right": True,
    "grid.linestyle": "--",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
    "legend.frameon": False,      # Legends float without a heavy box
    "legend.fontsize": 8,
}

plt.rcParams.update(_PUB_RC)

# PRB Single-Column exact width is 3.375 inches. 
# 2.6 height gives a nice, compact aspect ratio for a secondary figure.
fig, ax = plt.subplots(figsize=(3.375, 2.6), constrained_layout=True)

# Create the scatter plot using pure Matplotlib to avoid Seaborn style clashes
ax.scatter(
    sampled_df['Bond_Angle'], 
    sampled_df['Local_J'], 
    color='#1f77b4', 
    alpha=0.85, 
    edgecolors='white',
    linewidths=0.4,
    s=25, 
    zorder=3
)

# Draw the Physics Boundaries
ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=1.0, label='FM / AFM Boundary', zorder=2)
ax.axvline(x=90, color='gray', linestyle=':', linewidth=1.0, label=r'Pristine $\sim90^\circ$ Bond', zorder=2)

# Format labels
ax.set_xlabel(r'Cr-I-Cr Bond Angle ($^\circ$)')
ax.set_ylabel(r'Exchange Coupling $J$ (meV)')

# Add professional grid
ax.grid(True, zorder=0)

# Format legend
ax.legend(loc='lower right', handlelength=1.5)

# Save as a true vector PDF at 600 DPI (APS Standard)
plt.savefig('Paper_Goodenough_Kanamori_Sampled.pdf', format='pdf', dpi=600, transparent=True, bbox_inches='tight')
print("PRB-formatted vector plot saved successfully!")

plt.show()