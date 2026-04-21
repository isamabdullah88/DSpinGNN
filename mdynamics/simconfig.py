
from dataclasses import dataclass
from pathlib import Path

# =========================================================
# 1. Simulation Configuration Module
# =========================================================
@dataclass
class SimConfig:
    """Strictly defines and validates all simulation parameters."""
    structurepath: str
    exchangepath: str
    nx: int
    ny: int
    tmpK: float
    timesteps: int
    amplitude: float
    strain_type: str
    out_dir: str = "Simulations"

    @property
    def run_name(self) -> str:
        """Generates a clean, consistent folder name for the simulation."""
        return f"{self.strain_type}_{self.nx}x{self.ny}_T{self.tmpK}K_Amp{self.amplitude}A_{self.timesteps}steps"

    @property
    def target_dir(self) -> Path:
        """Builds the full nested path: out_dir / Model_Folder / Run_Name"""
        model_folder_name = Path(self.structurepath).parent.name
        return Path(self.out_dir) / model_folder_name / self.run_name