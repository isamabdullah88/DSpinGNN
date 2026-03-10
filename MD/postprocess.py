from ase.io import read
from ase.geometry.analysis import Analysis
import numpy as np

import matplotlib.pyplot as plt

# 1. Set the base style
plt.style.use('seaborn-v0_8-paper') # or 'ggplot' for a gray background

# 2. Customize the "RC" (Runtime Configuration) params
plt.rcParams.update({
    "font.family": "serif",      # Use serif fonts like a real physics paper
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 1.5,
    "figure.figsize": (8, 5),    # Good default ratio
    "figure.dpi": 150,           # High resolution for your thesis PDF
    "savefig.bbox": "tight",     # Removes unnecessary whitespace when saving
    "axes.grid": True,           # Helpful for reading energy values
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# Optional: If you have LaTeX installed on your Mac, uncomment this for perfect math:
# plt.rcParams.update({"text.usetex": True})

def radial_distribution_function():
    # 1. Load the trajectory
    traj = read('aspirin_production.traj', index=':')

    for atoms in traj:
        atoms.set_cell([15, 15, 15])  # Ensure consistent cell for analysis
        atoms.center()

        atoms.set_pbc(False)

    # 2. Initialize Analysis
    ana = Analysis(traj)

    # 3. Calculate RDF for Carbon-Carbon pairs
    # rmax: maximum distance to look; nbins: resolution of the plot
    rdf_cc = ana.get_rdf(rmax=5.0, nbins=200, elements=['C', 'C'])

    # Average over all frames
    rdf_avg = np.mean(rdf_cc, axis=0)
    r = np.linspace(0, 5.0, 200)

    plt.plot(r, rdf_avg)
    plt.title("C-C Radial Distribution Function")
    plt.xlabel("Distance (Å)")
    plt.ylabel("g(r)")
    plt.show()

def vibrational_density_of_states():
    # Vibrational Density of States (VDOS) from Velocity Autocorrelation Function (VACF)
    # 1. Load Trajectory and Masses
    traj = read('aspirin_production.traj', index=':')

    for atoms in traj:
        atoms.set_center_of_mass([0, 0, 0])
        # Optionally remove rotation using ase.utils.geometry.remove_rotation
        atoms.set_pbc(False)

    # shape: [steps, atoms, 3]
    vels = np.array([atoms.get_velocities() for atoms in traj]) 
    masses = traj[0].get_masses() # Shape: [atoms]

    # 2. Fast VACF using FFT (The Sliding Window Fix)
    def compute_vacf_fast(vels, masses):
        # vels: (T, N, 3)
        n_steps = vels.shape[0]
        
        # Mass-weighting: w_i = sqrt(m_i)
        # We multiply velocities by sqrt(mass) so heavier atoms contribute more correctly to the DOS
        weights = np.sqrt(masses)
        # Broadcast weights: (1, N, 1) to match vels (T, N, 3)
        weighted_vels = vels * weights[None, :, None]
        
        # Flatten atoms and dimensions: (T, 3N)
        # This allows us to correlate all coordinates at once
        flat_vels = weighted_vels.reshape(n_steps, -1)
        
        # Pad with zeros to 2*N to avoid circular correlation from FFT
        # This is the standard "trick" for non-periodic correlation
        padding = np.zeros_like(flat_vels)
        padded_vels = np.concatenate([flat_vels, padding], axis=0)
        
        # FFT -> Power -> IFFT = Autocorrelation
        ft = np.fft.rfft(padded_vels, axis=0)
        power = ft * np.conj(ft)
        autocorr = np.fft.irfft(power, axis=0)
        
        # Take only the first half (real lags) and sum over all atoms/dims
        # Normalize by dividing by (N - lag) to account for fewer samples at large lags
        valid_len = n_steps
        norm_factor = np.arange(valid_len, 0, -1)
        vacf = np.sum(autocorr[:valid_len], axis=1) / norm_factor
        
        # Normalize so VACF starts at 1.0 (optional, but good for plotting VACF itself)
        return vacf / vacf[0]

    vacf = compute_vacf_fast(vels, masses)

    # 3. Apply Window Function (Crucial for clean peaks)
    # Without this, the sudden cutoff of VACF at the end causes "ripples" in the spectrum
    window = np.hanning(len(vacf))
    vacf_windowed = vacf * window

    # 4. Compute Spectrum
    dt = 0.5 # fs
    # Pad with zeros before final FFT to increase resolution (interpolation)
    pad_len = len(vacf) * 4 
    spectrum = np.abs(np.fft.rfft(vacf_windowed, n=pad_len))
    freqs = np.fft.rfftfreq(pad_len, d=dt * 1e-15) # Hz

    # 5. Convert to Wavenumbers (cm^-1)
    # c in cm/s
    c_cms = 2.99792458e10 
    wavenumbers = freqs / c_cms

    # 6. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(wavenumbers, spectrum, color='black')

    # Limit to relevant IR range
    plt.xlim(0, 4000)
    plt.xlabel(r'Wavenumber ($\text{cm}^{-1}$)')
    plt.ylabel('Density of States (a.u.)')
    plt.title('Vibrational Density of States (VDOS)')
    plt.grid(True, alpha=0.3)

    # Add "Aspirin Signature" region highlights
    plt.axvspan(1700, 1800, color='red', alpha=0.1, label='Carbonyl C=O')
    plt.axvspan(2800, 3200, color='blue', alpha=0.1, label='C-H Stretch')
    plt.legend()

    plt.show()

    return wavenumbers, spectrum


def normal_mode_analysis():
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations
    from .nequipcalc import NequIPCalculator
    from trainutils import loadmodel

    model = loadmodel('checkpoints/latest-model.pt', mps=True)

    # 1. Load and Optimize your molecule (Crucial!)
    atoms = read('aspirin.xyz')
    atoms.calc = NequIPCalculator(model=model, device='mps')

    # The molecule MUST be at its exact minimum for this to work
    opt = BFGS(atoms)
    opt.run(fmax=0.001)

    # 2. Run Vibrational Analysis (Displacement method)
    vib = Vibrations(atoms)
    vib.run()

    # 3. Print the frequencies
    vib.summary()

    # 4. Compare these 'clean' frequencies to your MD spectrum
    # If these are also wrong, your model learned the wrong force constants.
    return vib


def plot_comparison(wavenumbers, md_spectrum, vib_obj):
    # 1. Get frequencies from the Vibrations object
    # vib_obj.get_frequencies() returns values in cm^-1 by default
    nma_freqs = vib_obj.get_frequencies()
    
    # Filter out imaginary frequencies (usually first 6 are translation/rotation)
    real_freqs = [f.real for f in nma_freqs if f.real > 10.0]

    plt.figure(figsize=(10, 6))

    # 2. Plot the MD Power Spectrum (The "Cloud")
    plt.plot(wavenumbers, md_spectrum / np.max(md_spectrum), 
             label='MD Power Spectrum (300K)', color='black', alpha=0.7)

    # 3. Plot NMA Frequencies as "Sticks"
    # We use vlines to create the vertical sticks
    plt.vlines(real_freqs, ymin=0, ymax=1.05, colors='red', 
               linestyles='dashed', linewidth=1, label='Normal Modes (Hessian)')

    plt.xlim(0, 4000)
    plt.ylim(0, 1.1)
    plt.xlabel(r'Wavenumber ($\text{cm}^{-1}$)')
    plt.ylabel('Normalized Intensity')
    plt.title('Aspirin: MD Spectrum vs. Static Normal Modes')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

# Example usage:
# plot_comparison(wavenumbers, spectrum, vib)


def torsional_angle_analysis():
    # Torsional Angle Analysis
    # Load your production trajectory
    traj = read('aspirin_production.traj', index=':')

    # Replace these with the actual indices from your view(atoms) session
    # Example: [Ring-C, Bridge-O, Carbonyl-C, Methyl-C]
    target_indices = [5, 6, 13, 15] 

    dihedral_values = []

    for atoms in traj:
        # get_dihedral returns values in [0, 360]
        angle = atoms.get_dihedral(target_indices)
        # Wrap to [-180, 180] for easier plotting of oscillations
        if angle > 180:
            angle -= 360
        dihedral_values.append(angle)

    # --- Visualization 1: Time Series ---
    plt.figure(figsize=(10, 4))
    plt.plot(dihedral_values, color='teal', linewidth=1)
    plt.title("Acetyl Group Torsional Angle vs. Time")
    plt.xlabel("Step")
    plt.ylabel("Angle (Degrees)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Visualization 2: Free Energy Profile ---
    # We can estimate the "Potential of Mean Force" (PMF)
    counts, bins = np.histogram(dihedral_values, bins=50, density=True)
    # PMF = -kT * ln(Probability)
    # Using T=300K, kT approx 0.0257 eV
    prob = counts / np.sum(counts)
    pmf = -0.0257 * np.log(prob + 1e-10) # Add epsilon to avoid log(0)
    pmf -= np.min(pmf) # Shift so minimum is 0

    plt.plot(bins[:-1], pmf, marker='o', color='purple')
    plt.title("Estimated Torsional Energy Barrier (PMF)")
    plt.xlabel("Angle (Degrees )")
    plt.ylabel("Relative Energy (eV)")
    plt.show()




if __name__ == "__main__":
    # Uncomment the analysis you want to run
    # radial_distribution_function()
    wavenumbers, spectrum = vibrational_density_of_states()
    # torsional_angle_analysis()
    vib = normal_mode_analysis()
    plot_comparison(wavenumbers, spectrum, vib)
    pass