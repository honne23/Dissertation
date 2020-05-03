import torch
import torch.nn as nn
import torch.nn.functional as F



class IQNetwork(nn.Module):
    
    def __init__(self, num_actions, quantiles):
        super(IQNetwork, self).__init__()
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
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512,num_actions * quantiles)
            )
        
        self.value_layer= nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512,quantiles)
            )
        
    def forward(self, x):
         x = x / 255.
         features = self.feature_layer(x)
         advantage = self.advantage_layer(features)
         advantage = advantage.view(-1, self.num_actions, self.quantiles)
         value = self.advantage_layer(features)
         value = value.view(-1, 1, self.quantiles)
         return value + (advantage - advantage.mean(dim=1).view(-1, 1, self.quantiles))
        