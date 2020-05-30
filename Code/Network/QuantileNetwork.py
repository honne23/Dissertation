import torch.nn as nn

class QuantileNetwork(nn.Module):
    def __init__(self, in_size,hidden_size, num_actions, quantiles):
        super(QuantileNetwork, self).__init__()
        self.num_actions = num_actions
        self.quantiles = quantiles
        self.feature_layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
        )
        
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions* quantiles)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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