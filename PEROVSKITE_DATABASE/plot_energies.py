import matplotlib.pyplot as plt
import numpy as np
import pickle 


with open("atoms_fps_energies.pkl", "rb") as f:
    atoms, fps, energies = pickle.load(f)
energies = np.array(energies)

lens = np.array([len(atom) for atom in atoms])

plt.scatter(np.arange(len(energies)), energies/lens)
plt.show()

