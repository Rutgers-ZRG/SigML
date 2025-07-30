from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import copy
from network import Network
from torch_geometric.data import Batch



class EF_Model(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)     

        self.act1 = nn.PReLU()  


    # def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
    #     data = copy.deepcopy(data_inp)
    #     data.x = F.sigmoid(self.em(data.x))
    #     data.z = F.sigmoid(self.em(data.z))

    #     output = super().forward(data)
    #     # output = torch.tanh(output)
    #     output = self.act1(output)

    #     if self.pool == True:
    #         output = torch.sum(output)

    #     return output

    def forward(self, data_inp: tg.data.Batch) -> list[torch.Tensor]:
        data = copy.deepcopy(data_inp)

        #### For now, equivalent indices are not included since they break the batching #####
        #### TODO: Find a way to integrate equivalent indices into batching (rn they are of different lengths for each grouping, so maybe instead include
        #### index table for each atom that maps to specific equivalent atom)
        if isinstance(data, list):
            batch = Batch.from_data_list(data, exclude_keys=["sig", "iws", "eq_inds"])
        elif isinstance(data, tg.data.Batch):
            batch = data
        else:
            raise TypeError(f"Expected list[Data] or Batch, got {type(data)}")

        ### Calculating how many atoms per graph
        ptr = batch.ptr                

        batch.x = F.sigmoid(self.em(batch.x))
        batch.z = F.sigmoid(self.em(batch.z))

        output = super().forward(batch)
        output = self.act1(output)

        sizes = (ptr[1:] - ptr[:-1]).tolist()   # [n_atoms₀, n_atoms₁, …]
        per_graph = torch.split(output, sizes, dim=0)
        graph_sums = torch.stack([torch.sum(graph, dim=0) for graph in per_graph]).squeeze(-1)

        return graph_sums


def get_standard_ef_model(ave_neighbor_count, cutoff=4.0, weight_path=None):
    out_dim = 1
    em_dim = 32
    model = EF_Model(in_dim = 100,
                    em_dim= em_dim,
                    irreps_in = str(em_dim) + "x0e",
                    irreps_out = str(out_dim) + "x0e",
                    irreps_node_attr = str(em_dim) + "x0e",
                    layers=2,
                    mul=32,
                    lmax=2,
                    max_radius=cutoff,
                    num_neighbors=ave_neighbor_count,
                    reduce_output=True,
                    radial_layers=2,
                    radial_neurons=64)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model




    




