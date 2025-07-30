from EF_Model import get_standard_ef_model
from Sinf_Model import get_standard_sinf_model
from Sig_iws_Model import get_standard_full_sig_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text, \
                  train_test_split, train_full_sig, evaluate_full_sig, train_nequip_ef, \
                  eval_nequip_ef, train_sinf, evaluate_sinf, train_ef, evaluate_ef, get_spacegroup_atom
import pickle 
import os 
import numpy as np
from io import StringIO
from tqdm import tqdm
import torch

### just for testing ###
from torch.utils.data import DataLoader
from utils import collate_to_list
#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}\n")



### Testing on previous datasets with already provided self energies ###
# source_dir = "SAMPLE_CDSE_CALCS/"
# patoms = []
# psig_texts = []
# pefs = []
# for pf in os.listdir(source_dir):
# # for pf in file_names:
#    with open(source_dir + pf, "rb") as f:
#        tatoms, tefs, _, _ = pickle.load(f)
#    patoms.extend(tatoms)
# #    psig_texts.extend(tsig_texts)
#    pefs.extend(tefs)

with open("../PEROVSKITE_DATABASE/atoms_fps_energies.pkl","rb") as f:
    patoms, pfps, pefs = pickle.load(f)


# patoms = patoms[18928:]
# pefs = pefs[18928:]
# pfps = pfps[18928:]





# read_data = True
read_data = False
if read_data:
    dataset = pickle.load(open("dataset.pkl", "rb"))
else:
    dataset = build_data(patoms, sig_texts=None, efs=pefs, fps=pfps, device=device)
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
ave_neighbors = get_average_neighbor_count(dataset)
# n_matsubara = len(dataset[0].iws[0])
train_data, test_data = train_test_split(dataset, train_percent=0.9, seed=34533)
print("Training on dataset of length", len(train_data))


### Training for the Sig(iwn) - Sig(iwn -> infty) model ###
# full_sig_model = get_standard_full_sig_model(n_matsubara, ave_neighbors, radial_cutoff=4.0, device=device)
# opt = torch.optim.AdamW(full_sig_model.parameters(), lr=0.01, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.25)
# loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
# save_path = "SAVED_MODELS/full_sig_model.pth"

# full_sig_model.load_state_dict(torch.load(save_path))
# train_full_sig(full_sig_model, opt, train_data, loss_fn, scheduler, save_path = save_path, max_iter=20, val_percent = 0.1, device=device, batch_size=1)
# full_sig_model.load_state_dict(torch.load(save_path))
# for o in range(5):
#     evaluate_full_sig(full_sig_model, test_data, orbital=o, display=False, img_save_dir="output_images")
# exit()
############################################################




### ### Two options for Ef model: Using NequIP or using custom E3NN model ### ###

### Custom E3NN Model ###
batch_size = 16

ef_model = get_standard_ef_model(ave_neighbors, cutoff=4.0)
opt = torch.optim.AdamW(ef_model.parameters(), lr=0.001*batch_size, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
save_path = "SAVED_MODELS/ef_model.pth"

# ef_model.load_state_dict(torch.load(save_path))
train_ef(ef_model, opt, train_data, loss_fn, scheduler, save_path = save_path, max_iter=20, val_percent = 0.1, device=device, batch_size=batch_size)
evaluate_ef(ef_model, test_data, device=device)

### NequIP Model ###
# train_nequip_ef(config_path="./default_config.yaml", dataset=train_data)
# eval_nequip_ef(model_path="SAVED_MODELS/nequip_ef_model.pth", dataset=test_data, display=False, img_save_dir="output_images")
# exit()
####################################################################################


############## Training for the S_infty model ############################
# sinf_model = get_standard_sinf_model(ave_neighbors, device=device)
# opt = torch.optim.AdamW(sinf_model.parameters(), lr=0.01, weight_decay=0.05)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
# loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
# save_path = "SAVED_MODELS/sinf_model.pth"

# # sinf_model.load_state_dict(torch.load(save_path))
# # train_sinf(sinf_model, opt, train_data, loss_fn, scheduler, save_path, max_iter=20)
# sinf_model.load_state_dict(torch.load(save_path))
# evaluate_sinf(sinf_model, test_data, display=False, img_save_dir="output_images", device=device)
# exit()
##########################################################################











