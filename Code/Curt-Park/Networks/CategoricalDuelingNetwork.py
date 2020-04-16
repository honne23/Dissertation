
import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalDuelingNetwork(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(CategoricalDuelingNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
        )
        
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, out_dim*atom_size))
        self.value_layer = nn.Sequential(
            nn.Linear(128,atom_size))
        
        for x in (self.advantage_layer, self.value_layer, self.feature_layer):
            for i in x:
                if isinstance(i, nn.Linear):
                    nn.init.xavier_uniform_(i.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        
        advantage = self.advantage_layer(feature).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(feature).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist