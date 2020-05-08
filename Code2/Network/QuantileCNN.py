import torch.nn as nn
from Network.NoisyLinear import NoisyLinear
class QuantileCNN(nn.Module):
    def __init__(self, hidden_size, num_actions, quantiles):
        super(QuantileCNN, self).__init__()
        self.num_actions = num_actions
        self.quantiles = quantiles
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels=32, kernel_size=8, stride = 4), 
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=4, stride = 2), 
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride = 1), 
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.advantage_layer = nn.Sequential(
            nn.Linear(7*7*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions* quantiles)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(7*7*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, quantiles)
        )
    def forward(self, x):
        out = self.feature_layer(x)
        adv = self.advantage_layer(out)
        val = self.value_layer(out)
        adv = adv.view(-1, self.num_actions, self.quantiles)
        val = val.view(-1, 1, self.quantiles)
        return val + adv - adv.mean(dim=1).view(-1, 1, self.quantiles)
    