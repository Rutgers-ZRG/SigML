import ase.db 
from ase import Atoms
import pickle 
import os
import libfp
import numpy as np
from tqdm import tqdm


def get_fp(cell, positions, symbols, numbers):
    lat = cell[:]
    rxyz = positions 
    syms = symbols
    types_map = {}
    set_syms = list(set(syms))
    for i, s in enumerate(set_syms):
        types_map[s] = i 
    types = []
    for i in range(len(syms)):
        types.append(types_map[syms[i]])

    atomic_numbers  = np.array(list(set(numbers)))
    cell = (np.array(lat), np.array(rxyz), np.array(types), atomic_numbers)
    fp = libfp.get_lfp(cell, cutoff=4.0, log=False, natx=400, orbital='s')
    return fp


con = ase.db.connect("cubic_perovskites.db")


# Print all keys in the database
atoms = []
fps = []
energies = []
print("Database keys:")
for row in tqdm(con.select(), desc="Getting atoms..."):
    if np.abs(row.standard_energy) < 1e-9:
        continue
    atom = Atoms(symbols=row.symbols, positions=row.positions, cell=row.cell, pbc=True)
    atoms.append(atom)
    fp = get_fp(row.cell, row.positions, row.symbols, row.numbers)
    fps.append(fp)
    energies.append(row.standard_energy)



with open("atoms_fps_energies.pkl", "wb") as f:
    pickle.dump((atoms, fps, energies), f)









