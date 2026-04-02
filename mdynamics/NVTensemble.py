import torch
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from data.crI3 import CrI3
from .dspingnn import DSpinGNNCalculator
from model import DSpinGNN
from train.trainutils import load_checkpoint

from logging import getLogger
logger = getLogger(__name__)


def runmd(timesteps, tmpK, outfile):
    # Detect Hardware (CUDA vs. MPS vs. CPU)
    if torch.cuda.is_available():
        device = 'cuda'
        use_mps = False
    elif torch.backends.mps.is_available():
        device = 'mps'
        use_mps = True
    else:
        device = 'cpu'
        use_mps = False

    logger.info(f"Hardware detected! Routing tensors to: {device.upper()}")

    # Load the Model
    model = DSpinGNN(mps=use_mps)
    model, _, _, _ = load_checkpoint(model, 'checkpoints/latest-model.pt', device)

    # 3. Generate Pristine CrI3 Lattice & Make Supercell
    logger.info("Generating pristine CrI3 lattice mathematically...")
    cri3_manager = CrI3() 
    patoms = cri3_manager.batoms.copy() 

    # Simulate 5x5 in-plane supercell
    atoms = patoms * (5, 5, 1)
    logger.info(f"Supercell created with {len(atoms)} atoms.")

    # 4. Attach the custom DSpinGNN Calculator
    atoms.calc = DSpinGNNCalculator(model=model, rcut=7.0, device=device)

    # 5. Set initial velocities and KILL global rotation
    logger.info("Thermalizing to 300K...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=tmpK)

    # Prevent the "slow rotating/drifting cluster" effect
    Stationary(atoms)    # Sets center-of-mass velocity to zero
    ZeroRotation(atoms)  # Sets center-of-mass angular momentum to zero

    # Thermostat (NVT)
    # 1.0 fs is a standard safe timestep for heavy transition metals (Cr) and Halogens (I)
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=tmpK, friction=0.01)

    # Setup a custom observer to write an Extended XYZ file
    open(outfile, 'w').close()

    def write_frame():
        # Append the current atomic state to the XYZ file
        write(outfile, atoms, append=True)

    # Attach the observer to run every 5 steps
    dyn.attach(write_frame, interval=5)

    # 8. Run the test
    logger.info("Starting Proof of Life MD...")
    dyn.run(timesteps)
    logger.info("MD Finished! Open '" + outfile + "' in OVITO to view the dynamics.")



if __name__ == "__main__":
    timesteps = 1000
    tmpK = 300
    outfile = "CrI3-MD-100.xyz"
    runmd(timesteps=1000, tmpK=tmpK, outfile=outfile)