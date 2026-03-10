from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Define Aspirin via SMILES string
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
mol = Chem.MolFromSmiles(smiles)

# 2. Add Hydrogens (SMILES usually hides them)
mol = Chem.AddHs(mol)

# 3. Generate 3D coordinates using ETKDG method
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

# 4. Optimize the geometry slightly so it's stable
AllChem.MMFFOptimizeMolecule(mol)

# 5. Write to .xyz file
Chem.MolToXYZFile(mol, "aspirin.xyz")

print("aspirin.xyz created successfully!")