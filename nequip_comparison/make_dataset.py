import pickle
from utils import train_test_split
import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

with open("atoms_fps_energies.pkl","rb") as f:
    patoms, pfps, pefs = pickle.load(f)


patoms = patoms[:1000]
pefs = pefs[:1000]
pfps = pfps[:1000]


train_atoms, test_atoms = train_test_split(patoms, train_percent=0.9, seed=34533)
train_atoms, val_atoms = train_test_split(train_atoms, train_percent=0.9, seed = 44123)

train_energies, test_energies = train_test_split(pefs, train_percent=0.9, seed=34533)
train_energies, val_energies = train_test_split(train_energies, train_percent=0.9, seed = 44123)


tr1 = []
v1 = []
ts1 = []

for i in range(len(train_atoms)):
  newatom = train_atoms[i].copy()
  newatom.calc = SinglePointCalculator(atoms = newatom, energy=train_energies[i])
  tr1.append(newatom)


for i in range(len(val_atoms)):
  newatom = val_atoms[i].copy()
  newatom.calc = SinglePointCalculator(atoms = newatom, energy=val_energies[i])
  v1.append(newatom)


for i in range(len(test_atoms)):
  newatom = test_atoms[i].copy()
  newatom.calc = SinglePointCalculator(atoms = newatom, energy=test_energies[i])
  ts1.append(newatom)



write("train_data.extxyz", tr1, format='extxyz')
write("val_data.extxyz", v1, format='extxyz')
write("test_data.extxyz", ts1, format='extxyz')





