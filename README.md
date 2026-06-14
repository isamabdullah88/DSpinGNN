# DSpinGNN: Physics-Informed Equivariant GNN for Dynamic Magnetic Exchange in Strain-Deformed Monolayer CrI₃

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![e3nn](https://img.shields.io/badge/e3nn-equivariant-blueviolet.svg)](https://e3nn.org/)
[![ASE](https://img.shields.io/badge/ASE-dynamics-brightgreen.svg)](https://wiki.fysik.dtu.dk/ase/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.11685-b31b1b.svg)](https://arxiv.org/abs/2606.11685)

This repository contains the official implementation of **DSpinGNN**, a physics-informed, E(3)-equivariant Graph Neural Network for predicting instantaneous, position-dependent isotropic magnetic exchange couplings J(r) across a dynamically deforming crystal lattice. Trained on 8-atom DFT+U primitive cells and deployed on a 3,200-atom supercell, DSpinGNN bridges the length-scale gap between first-principles quantum mechanics and mesoscale spin-lattice dynamics in strain-engineered 2D magnetic materials.

---

## Overview

First-principles molecular dynamics captures quantum mechanical accuracy but is limited to ~100 atoms over picoseconds due to O(N³) scaling. Classical molecular dynamics reaches the required length and time scales but carries no representation of quantum magnetic exchange. DSpinGNN resolves both constraints simultaneously through a bifurcated architecture: an E(3)-equivariant GNN drives the structural dynamics while an independent physics-informed Δ-MLP predicts the instantaneous magnetic exchange landscape from the evolving bond geometry.

The Goodenough-Kanamori superexchange relationship is embedded directly as an analytical inductive bias in the exchange predictor, providing physically grounded extrapolation under extreme local deformations and making the model's predictions interpretable in terms of established quantum chemistry.

---

## Simulation Preview

*Mesoscale spin-lattice dynamics in a 3,200-atom CrI₃ monolayer at 5 K:*

[![Watch the Simulation](https://img.youtube.com/vi/Ma-eBKL-Knc/maxresdefault.jpg)](https://youtu.be/Ma-eBKL-Knc)

A propagating biaxial strain wave reflects at the periodic boundaries, and constructive superposition of incident and reflected components transiently drives localized regions beyond the DFT-established FM-to-AFM threshold (~6% compression). The result is a dynamic cycle of concentric ferromagnetic (FM) and antiferromagnetic (AFM) exchange zones that form, sharpen, and dissipate as the wave damps through the Langevin thermostat.

---

## Key Results

### Model Accuracy — Strictly Withheld Test Set

The 61-configuration test set was generated after all model development was complete, was never seen by the model at any stage of training or selection, and was evaluated exactly once. Energy, force, and exchange coupling predictions were assessed simultaneously against DFT+U ground truth.

| Quantity | Training MAE | Validation MAE | Test MAE |
|---|---|---|---|
| Energy (meV/atom) | 0.9 | 1.0 | 1.1 |
| Force (meV/Å) | 4.0 | 6.0 | 6.5 |
| J coupling (meV) | 0.20 | 0.27 | 0.18 |

Exchange coupling test R² = 0.91. Dataset splits were stratified across deformation mode (biaxial, uniaxial, shear) and strain magnitude to ensure each split is representative of the full deformation space.

### Length-Scale Transferability — 8 → 3,200 Atoms (400×)

Trained exclusively on 8-atom primitive cells, DSpinGNN is deployed without retraining on a 3,200-atom (20×20) supercell. E(3)-equivariance guarantees that the learned representations are invariant to rotation and translation, providing stable long-horizon dynamics without unphysical error accumulation.

### Mesoscale Observables Inaccessible to Direct DFT

Two quantitative predictions are extracted from the simulation trajectory:

- **Domain wall width**: ξ = 1.7 ± 0.3 nm (mean over N = 2 constructive interference events, from hyperbolic tangent fitting of the radial J profile)
- **Interference oscillation period**: τ = 0.27 ps (from the AFM Cr-fraction time series)

Both lie within the measurement range of cryogenic magnetic force microscopy under acoustic excitation, providing direct experimental validation targets.

### Physics-Informed Consistency

The predicted J(θ) relationship across the full 3,200-atom trajectory follows the Goodenough-Kanamori functional form with a FM-to-AFM crossover at θ ≈ 87°, confirming the analytical inductive bias is active and physically correct throughout the mesoscale domain.

---

## Architecture

```
Input: strained 8-atom CrI₃ configuration
       │
       ├─── E-GNN Branch (NequIP-style) ─────────────────────────────┐
       │     3 interaction layers, 7.0 Å cutoff                       │
       │     E(3)-equivariant message passing (e3nn)                  │
       │     → Total energy E  (scalar, invariant)                    │
       │     → Atomic forces F (vector, equivariant)                  │
       │                                                              │
       └─── Δ-MLP Branch (per Cr–Cr bond) ──────────────────────────┘
             Inputs: θ (Cr-I-Cr angle), l₁…l₄ (Cr-I legs), d(Cr-Cr)
             │
             ├── GK Analytical Block:
             │     J_analytical = (A cos²θ + B cosθ + C) · exp(−α(⟨l⟩ − l_ref))
             │     Parameters A, B, C, α learned end-to-end
             │
             └── Residual MLP:
                   2 hidden layers × 16 neurons, SiLU
                   192-dim input (CosineSmearing × 5 + ChebyshevAngleSmearing)
                   → J_residual

Final output: J_ij = J_analytical + J_residual   (meV, per bond)
```

All equivariant layers are implemented directly using e3nn and PyTorch operations. The NequIP codebase is not used as a dependency — only its architectural design is followed.

---

## Physical Approximations

DSpinGNN operates within three explicit approximations that define the scope of all predictions:

1. **Collinear Ising constraint** — Spin vectors are confined to ±z. This serves as a minimal proxy for the spin-orbit-coupling-driven single-ion anisotropy that stabilizes CrI₃ against Mermin-Wagner fluctuations. Magnon spectra and non-collinear order are outside scope.

2. **Collinear DFT, SOC omitted** — The target quantity is the isotropic exchange J₁, whose FM-to-AFM sign change under strain is driven entirely by the Goodenough-Kanamori competition between orthogonal superexchange and direct d-d overlap — a mechanism independent of spin-orbit coupling. SOC-driven terms (single-ion anisotropy, Kitaev exchange, DMI) are omitted.

3. **Adiabatic decoupling** — The Δ-MLP maps instantaneous bond geometry to J as a post-hoc operation; no magnetic energy enters the force calculation. This is justified by the femtosecond electronic timescale on which J adjusts to nuclear positions, far faster than the phonon dynamics driving the structural evolution.

---

## Full Computational Pipeline

DSpinGNN is one half of a two-repository research system:

```
SpinDFT (data generation)          DSpinGNN (ML + simulation)
─────────────────────────          ──────────────────────────
Strained CrI₃ structures           E-GNN + Δ-MLP architecture
   │                                    │
   ▼                                    ▼
QE → Wannier90 → TB2J   ────►   Training (345 configs)
   │                                    │
   ▼                                    ▼
E, F, J per config                Validation & test
                                        │
                                        ▼
                                 3,200-atom Langevin MD
                                        │
                                        ▼
                                 ξ, τ mesoscale observables
```

The training dataset is generated by [SpinDFT](https://github.com/isamabdullah88/SpinDFT), a companion repository providing the high-throughput DFT+U → Wannier90 → TB2J pipeline.

---

## Installation

```bash
git clone https://github.com/isamabdullah88/DSpinGNN.git
cd DSpinGNN
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Role |
|---|---|---|
| PyTorch | 2.0+ | Neural network training and inference |
| e3nn | latest | Euclidean equivariant layers |
| ASE | latest | Atomic structure I/O and Langevin dynamics |
| PyTorch Geometric | latest | Graph construction and batching |
| Weights & Biases | latest | Training monitoring and experiment tracking |
| OVITO | latest | Trajectory visualization |

---

## Usage

### 1. Train the E-GNN Structure Model

The E-GNN predicts total energies and atomic forces. Train it first using `main.py` with `--modelname StructureModel`:

```bash
python main.py \
  --modelname    StructureModel \
  --datasetpath  ./DataSets/GNN/RattleGNN.pth \
  --epochs       5000 \
  --batch_size   32 \
  --lr           1e-3 \
  --project      StructureGNN \
  --runname      Run_01 \
  --WANDB_KEY    <your_wandb_key>
```

### 2. Train the Δ-MLP Exchange Model

The Δ-MLP predicts per-bond exchange couplings J(r). Train it independently using `--modelname ExchangeMLP`. The GK analytical block and residual MLP parameter groups use separate learning rates automatically:

```bash
python main.py \
  --modelname    ExchangeMLP \
  --datasetpath  ./DataSets/GNN/RattleGNN.pth \
  --epochs       5000 \
  --batch_size   32 \
  --lr           1e-2 \
  --project      ExchangeMLP \
  --runname      Run_01 \
  --WANDB_KEY    <your_wandb_key>
```

### 3. Fine-Tuning

To resume training from a saved checkpoint:

```bash
python main.py \
  --modelname  ExchangeMLP \
  --finetune \
  --runname    Run_02_Finetune \
  --WANDB_KEY  <your_wandb_key>
```

Checkpoints are loaded from `./checkpoints/latest-model.pt` automatically when `--finetune` is set.

### 4. Running on HPC (SLURM)

A pre-configured SLURM submission script is included:

```bash
bash hpcrun.sh
```

Edit `hpcrun.sh` to set your partition, node count, and module paths before submitting.

### 5. Mesoscale MD Simulation

Once both models are trained, configure the simulation by editing the `SimConfig` block at the bottom of `mdynamics/NVTEnsemble.py`:

```python
config = SimConfig(
    structurepath = "checkpoints/Structural/<run>/Structure-Epoch-XXXX.pt",
    exchangepath  = "checkpoints/Exchange/<run>/Exchange-Epoch-XXXX.pt",
    nx            = 20,           # supercell size (20×20 = 3,200 atoms)
    ny            = 20,
    tmpK          = 5,            # temperature in K
    timesteps     = 1000,         # number of MD steps
    amplitude     = 1.5,          # strain wave amplitude in Å
    strain_type   = "biaxial"     # "biaxial", "Uniaxial_X", or "Uniaxial_Y"
)
```

Then run:

```bash
python -m mdynamics.NVTEnsemble
```

Outputs are written to the configured `target_dir`:
- `trajectory.xyz` — full atomic trajectory (open in OVITO)
- `data.txt` — per-step energy, Cr-I-Cr angles, and exchange coupling values

### 6. Visualization

Post-processing and analysis scripts are in the `visualization/` folder. Trajectory files (`.xyz`) can be opened directly in [OVITO](https://www.ovito.org/) for per-bond exchange coupling map rendering and domain wall analysis.

---

## Repository Structure

```
DSpinGNN/
├── main.py               # Entry point: train StructureModel or ExchangeMLP
├── logger.py             # Dual-output logger (terminal + timestamped file)
├── hpcrun.sh             # SLURM submission script for HPC clusters
├── requirements.txt
│
├── model/                # Model definitions
│   ├── StructureGNN      # E(3)-equivariant GNN (energy + forces)
│   └── ExchangeMLP       # Physics-informed Δ-MLP (exchange coupling)
│
├── train/                # Training utilities
│   └── trainutils.py     # load_checkpoint, MultiTaskLoss, Trainer, etc.
│
├── data/                 # Dataset loading and management
│   └── DatasetManager    # Stratified train/val/test splitting
│
├── graph/                # Graph construction
│   └── CrI3              # CrI₃ unit cell and supercell builder
│
├── mdynamics/            # Molecular dynamics simulation
│   ├── NVTEnsemble.py    # CrI3_Simulator: Langevin MD driver
│   ├── simconfig.py      # SimConfig dataclass
│   ├── dspingnn.py       # DSpinGNNCalculator (ASE-compatible calculator)
│   ├── strains.py        # StrainEngineer: ripple and cell-strain modes
│   └── tracker.py        # MaxForceTracker: per-step force monitoring
│
├── visualization/        # Analysis and plotting scripts
└── server/               # Inference server utilities
```

---

## Tech Stack

**Deep learning** — PyTorch, PyTorch Geometric, e3nn, Weights & Biases

**First-principles** — Quantum ESPRESSO, Wannier90, TB2J (via SpinDFT)

**Simulation & analysis** — ASE, OVITO, Matplotlib, NumPy, SciPy

**Compute** — LUMS HPC cluster (DFT data generation), NVIDIA RTX 4090 (model training), DigitalOcean GPU droplets

---

## Related Work & Applications

DSpinGNN demonstrates a general framework for embedding quantum mechanical exchange rules into equivariant machine learning potentials. Direct extensions include:

- **Full exchange tensor** prediction (including Kitaev and DMI) using time-reversal equivariant representations
- **Spin-lattice back-action** by coupling J into the force calculation for true magnon-phonon dynamics
- **Other 2D van der Waals magnets** (CrBr₃, CrCl₃, CrGeTe₃) using the same pipeline with retraining

---

## Citation

If you use DSpinGNN or SpinDFT in your work, please cite:

```bibtex
@article{balghari2026dspingnn,
  author        = {Isam Abdullah Balghari and Muhammad Faryad and Muhammad Sabieh Anwar},
  title         = {{DSpinGNN}: A Physics-Informed Equivariant Graph Neural Network
                   for Dynamic Magnetic Exchange Prediction in Strain-Deformed
                   Monolayer {CrI}$_3$},
  journal       = {Physical Review Materials},
  year          = {2026},
  note          = {Under review},
  eprint        = {2606.11685},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.11685},
}
```

---

## Authors

**Isam Abdullah Balghari** — *Lead developer and researcher*
Department of Physics, LUMS
✉ isamabdullah88@gmail.com

**Supervisor: Dr. Muhammad Sabieh Anwar**
Department of Physics, LUMS — [Profile](https://physlab.org/muhammad-sabieh-anwar-personal/)

**Co-supervisor: Dr. Muhammad Faryad**
Department of Physics, LUMS — [Profile](https://lums.edu.pk/lums_employee/4010)

---

## License

This project is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for details.
