import torch
import torch.nn as nn

class DuelingNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(DuelingNetwork, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, out_dim))
        self.value_layer = nn.Sequential(
            nn.Linear(128,1))
        
        for x in (self.feature_layer, self.advantage_layer, self.value_layer):
            for i in x:
                if isinstance(i, nn.Linear):
                    nn.init.xavier_uniform_(i.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.feature_layer(x)
        advantage = self.advantage_layer(output)
        value = self.value_layer(output)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

