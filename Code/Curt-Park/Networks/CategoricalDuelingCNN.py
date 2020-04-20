import torch
import torch.nn as nn
import torch.nn.functional as F
from Networks.NoisyLinear import NoisyLinear

class CategoricalDuelingCNN(nn.Module):
    def __init__(
        self, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(CategoricalDuelingCNN, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels=32, kernel_size=8, stride = 4), 
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=4, stride = 2), 
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride = 1), 
            nn.ReLU(),
        )
        
        self.advantage_layer = nn.Sequential(
            NoisyLinear(7*7*64, 512),
            nn.ReLU(),
            NoisyLinear(512, out_dim*atom_size))
        
        self.value_layer = nn.Sequential(
            NoisyLinear(7*7*64,512),
            nn.ReLU(),
            NoisyLinear(512, atom_size))
        
        
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = x / 255.
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        
        feature = feature.view(feature.size(0), - 1) #flatten convolution
        
        advantage = self.advantage_layer(feature).view(
            feature.size(0), self.out_dim, self.atom_size
        )
        value = self.value_layer(feature).view(feature.size(0), 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms.view(-1, self.atom_size), dim=1).view(-1, self.out_dim, self.atom_size)
        #dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        for x in (self.advantage_layer, self.value_layer):
            for i in x:
                if isinstance(i, NoisyLinear):
                    i.reset_noise()