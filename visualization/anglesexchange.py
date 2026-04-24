import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
                for a,j in zip(parts[2].split(','), parts[3].split(',')):
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
# --- UNIFORM SAMPLING LOGIC ENDS HERE ---
# ==========================================

# 2. Set up the figure for high readability on a poster
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid") # Clean white background looks best on posters

# 3. Create the scatter plot
# NOTE: We are now passing 'sampled_df' instead of 'df'
# Alpha is bumped up to 0.7 since there are fewer overlapping dots
sns.scatterplot(
    data=sampled_df, 
    x='Bond_Angle', 
    y='Local_J', 
    color='#1f77b4', 
    alpha=0.7, 
    edgecolor=None,
    s=40 # 's' controls dot size, making them slightly larger looks better sampled
)

# 4. Draw the Physics Boundaries (The Goodenough-Kanamori visual proof)
# A red dashed line separating Ferromagnetic (Positive) and Antiferromagnetic (Negative)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='FM / AFM Boundary')

# A gray dotted line showing where pristine, unstrained CrI3 sits
plt.axvline(x=90, color='gray', linestyle=':', linewidth=2, label='Pristine ~90° Bond')

# 5. Format labels and titles (Make text large for poster readability)
plt.xlabel('Cr-I-Cr Bond Angle (Degrees)', fontsize=16, fontweight='bold')
plt.ylabel('Local Exchange Coupling J (meV)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, loc='best')

plt.tight_layout()

# 6. Save as a massive, high-res PNG for Canva (400 DPI)
plt.savefig('Goodenough_Kanamori_Plot_Sampled.png', dpi=400, bbox_inches='tight')
print("High-res sampled plot saved and ready for Canva!")