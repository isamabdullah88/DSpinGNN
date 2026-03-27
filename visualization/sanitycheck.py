
import os
import numpy as np

from ase.db import connect
from data import CrI3, EspressoHubbard


if __name__ == "__main__":
    phase = 'FM'
    stntype = 'Shear_XY'
    wkdir = f"./DataSets/GNN/Rattle-{stntype}-4"
    dbpath = os.path.join(wkdir, f"Rattle-{stntype}-{phase}-4.db")

    strains = np.linspace(-0.15, 0.15, 21)
    numrattles = 10

    hubbardcalc = EspressoHubbard(phase=phase)
    with connect(dbpath) as db:
        for i, strain in enumerate(strains):
            for j in range(numrattles):
                print(f'Writing to db: {i} of {len(strains)} strains, rattle {j+1} of {numrattles}...')
                pwopath = os.path.join(wkdir, f"{phase}", f"Strain_{stntype}_{strain:.4f}_{j}", "espresso.pwo")

                if not os.path.exists(pwopath):
                    print(f"Output file not found for strain {strain:.4f} at {pwopath}. Skipping...")
                    continue


                crI3 = CrI3()
                atoms = crI3.strain_atoms(stntype=stntype, stnvalue=strain)

                atomsout = hubbardcalc.parse(pwopath, atoms)

                energy = atomsout.get_potential_energy()
                # moms = atomsout.get_magnetic_moments()
                forces = atomsout.get_forces()
                # stress = atomsout.get_stress()

                # result = {
                #     'strain': strain,
                #     'id': f"CrI3_{stntype}_{strain:.4f}",
                #     'status': 'SUCCESS',
                #     'energy': energy,
                #     'mag_moments': moms,
                #     'forces': forces,
                #     'stress': stress,
                #     'atoms': atomsout
                # }
                db.write(atomsout)