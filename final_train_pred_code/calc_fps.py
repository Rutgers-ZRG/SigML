import pickle 
import os
import libfp
import numpy as np
from tqdm import tqdm


def get_fp(atom):
    lat = atom.get_cell()[:]
    rxyz = atom.positions 
    syms = atom.get_chemical_symbols()
    types_map = {}
    set_syms = list(set(syms))
    for i, s in enumerate(set_syms):
        types_map[s] = i 
    types = []
    for i in range(len(syms)):
        types.append(types_map[syms[i]])

    atomic_numbers  = np.array(list(set(atom.numbers)))
    cell = (np.array(lat), np.array(rxyz), np.array(types), atomic_numbers)
    fp = libfp.get_lfp(cell, cutoff=4.0, log=False, natx=100, orbital='s')
    return fp



source_dir = "SAMPLE_CDSE_CALCS/"
patoms = []
psig_texts = []
pefs = []
for pf in os.listdir(source_dir):
# for pf in file_names:
   with open(source_dir + pf, "rb") as f:
       tatoms, tefs, _, _ = pickle.load(f)
   patoms.extend(tatoms)
#    psig_texts.extend(tsig_texts)
   pefs.extend(tefs)

fps = []
for atom in tqdm(patoms, desc="Creating Fingerprints..."):
    fps.append(get_fp(atom))


with open("atoms_fps_energies.pkl","wb") as f:
    pickle.dump([patoms, fps, pefs], f)