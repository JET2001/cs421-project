from typing import List, Dict
import torch as th
import torch.nn as nn
import copy

class AutoEncoder(nn.Module):
    def __init__(self, hidden: List[int], latent_space_dim: int, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = copy.deepcopy(hidden)
        self.latent_space_dim = latent_space_dim
        
        self.modules : List[nn.Module] = [] 
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.modules.append(nn.Linear(self.input_dim, self.hidden[0]))
        # self.modules.append(nn.ReLU())
        for i in range(len(self.hidden)-1):
            self.modules.append(nn.Linear(self.hidden[i], self.hidden[i+1]))
            # self.modules.append(nn.ReLU())
        self.modules.append(nn.Linear(self.hidden[-1], self.latent_space_dim))
        self.modules.append(nn.ReLU())
        self.modules.append(nn.Linear(self.latent_space_dim, self.hidden[-1]))
        self.modules.append(nn.ReLU())
        # Decoder layer
        for i in range(len(self.hidden)-1, 0, -1):
            if i > 0:
                self.modules.append(nn.Linear(self.hidden[i], self.hidden[i-1]))
                # self.modules.append(nn.ReLU())
        self.modules.append(nn.Linear(self.hidden[0], self.input_dim))
            
        self.model = nn.Sequential(*self.modules).to(self.device)
    
    def forward(self, inputs: th.Tensor):
        return self.model(inputs)