import torch
import torch.nn as nn
from Network.NoisyLinear import NoisyLinear
class ResidueNetwork(nn.Module):
    def __init__(self, in_size : int, hidden_size:int, num_actions : int, quantiles: int):
        super(ResidueNetwork, self).__init__()
        self.num_actions = num_actions
        self.quantiles = quantiles
        self.feature_layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
        )
        
        self.action_embedding_layer = nn.Sequential(
            nn.Linear(num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
            )
        
        self.advantage_layer = nn.Sequential(
            NoisyLinear( hidden_size + 32, 256),
            nn.ReLU(),
            NoisyLinear(256, num_actions* quantiles)
        )
        self.value_layer = nn.Sequential(
            NoisyLinear(hidden_size + 32, 256),
            nn.ReLU(),
            NoisyLinear(256, quantiles)
        )
    def forward(self, x):
        action_distribution = x[:,-self.num_actions:]
        state = x[:,:-self.num_actions]
        
        state_embedding = self.feature_layer(state)
        action_embedding = self.action_embedding_layer(action_distribution)
        features = torch.cat((state_embedding, action_embedding), dim=1)
        adv = self.advantage_layer(features)
        val = self.value_layer(features)
        adv = adv.view(-1, self.num_actions, self.quantiles)
        val = val.view(-1, 1, self.quantiles)
        return val + adv - adv.mean(dim=1).view(-1, 1, self.quantiles)
    
    def reset_noise(self):
        for i in [*self.advantage_layer, *self.value_layer]:
            if isinstance(i, NoisyLinear):
                i.sample_noise()