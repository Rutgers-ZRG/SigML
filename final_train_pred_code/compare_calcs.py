from EF_Model import get_standard_ef_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text, \
                  train_test_split, train_full_sig, evaluate_full_sig, train_nequip_ef, \
                  eval_nequip_ef, train_sinf, evaluate_sinf, train_ef
import pickle 
import os 
import numpy as np
from io import StringIO
from tqdm import tqdm
import torch

### just for testing ###
from torch.utils.data import DataLoader
from utils import collate_to_list
from nequip.ase.nequip_calculator import nequip_calculator
from ase import Atoms

import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cpu")



calc1 = nequip_calculator("../nequip_comparison/deployed_model/cdsemodel.pth")


with open("atoms_fps_energies.pkl","rb") as f:
    patoms, pfps, pefs = pickle.load(f)
patoms = patoms[:1000]
pefs = pefs[:1000]
pfps = pfps[:1000]
dataset = build_data(patoms, sig_texts=None, efs=pefs, fps=pfps, device=device)
ave_neighbors = get_average_neighbor_count(dataset)
# n_matsubara = len(dataset[0].iws[0])
train_data, test_data = train_test_split(dataset, train_percent=0.9, seed=34533)
ef_model = get_standard_ef_model(ave_neighbors, cutoff=4.0, weight_path="SAVED_MODELS/ef_model.pth")


acts = []
preds_custom = []
preds_nequip = []

ndiffs = []
cdiffs = []

for i in tqdm(range(len(test_data))):
    pred_custom = ef_model(test_data[i]).cpu().detach().item()
    atom = Atoms(test_data[i].symbol, positions = test_data[i].pos, cell = test_data[i].lattice[0], pbc=True)
    atom.calc = calc1 
    pred_nequip = atom.get_potential_energy()
    custom_diff = np.abs(pred_custom - test_data[i].ef.item()) 
    nequip_diff = np.abs(pred_nequip - test_data[i].ef.item())

    ndiffs.append(nequip_diff)
    cdiffs.append(custom_diff)


fig, axs = plt.subplots(2)

axs[0].hist(cdiffs, bins=50)
axs[1].hist(ndiffs, bins=50)
plt.show()

