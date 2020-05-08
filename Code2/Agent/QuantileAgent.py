import torch
import numpy as np
import gym
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from Network.QuantileNetwork import QuantileNetwork
from Memory.PrioritisedReplay import PrioritisedReplay
class QuantileAgent:
    
    def __init__(self,
                 env: gym,
                 hidden_size:int = 128,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.01,
                 epsilon_decay: float = (1/2000),
                 mem_size: int = 5000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 num_quantiles: int = 51
                 ):
        self.env = env
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dqn = QuantileNetwork(self.env.observation_space.shape[0], hidden_size, self.env.action_space.n, num_quantiles).to(self.device)
        self.target = QuantileNetwork(self.env.observation_space.shape[0], hidden_size, self.env.action_space.n, num_quantiles).to(self.device)
        self.target.eval()
        self.target.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(params= self.dqn.parameters())
        self.max_epsilon = max_epsilon
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = PrioritisedReplay(mem_size)
        self.gamma = gamma
        self.num_quantiles = num_quantiles
        self.cumulative_density = torch.tensor((2 * np.arange(num_quantiles) + 1) / (2.0 * num_quantiles), device=self.device, dtype=torch.float) 
        self.quantile_weight = 1.0 / self.num_quantiles
        
    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)
    
    def update_network(self) -> float:
        
        idx, weights, samples = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights.astype(float)).to(self.device)
        
        samples = np.vstack(samples)
        state, action, reward, next_state, done = samples.T
        state = torch.FloatTensor(np.vstack(state)).to(self.device)
        next_state = torch.FloatTensor(np.vstack(next_state)).to(self.device)
        action = torch.LongTensor(action.astype(int)).to(self.device)
        reward = torch.FloatTensor(reward.astype(float).reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(done.astype(bool).reshape(-1, 1)).to(self.device)
        
        
        quantiles = self.dqn(state)
        quantiles = quantiles[torch.arange(quantiles.size(0)), action].squeeze(1) #select the quantiles of the actions chosen in each state
        quantiles_next = self.next_distribution(reward, next_state, done)
          
        diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0)

        loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
        loss = loss.transpose(0,1)
        self.memory.update(idx, loss.detach().mean(1).sum(-1).abs().cpu().numpy())
        loss = loss * weights.view(self.batch_size, 1, 1)
        loss = loss.mean(1).sum(-1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)        
        self.optimizer.step()
        
        return loss.item()
    
    def update_epsilon(self):
        self.epsilon = max( \
                    self.min_epsilon, self.epsilon - ( \
                        self.max_epsilon - self.min_epsilon \
                    ) * self.epsilon_decay \
                )
    def update_beta(self, frame_idx:int, num_frames:int):
        fraction = min(frame_idx / num_frames, 1.0)
        beta = self.memory.beta 
        beta = beta + fraction * (1.0 - beta)
        self.memory.beta = beta
    
    def next_distribution(self, reward, next_state, done):
        mask = (1-done).bool().squeeze(1)
        non_final = next_state[mask,:]
        with torch.no_grad():
            quantiles_next = torch.zeros((self.batch_size, self.num_quantiles), device=self.device, dtype=torch.float)
            if not (done.sum().item() == done.size(0)): #if there is at least one non-final next state
                max_next_action = self.get_max_next_state_action(non_final)
                quantiles_next[mask] = self.target(non_final).gather(1, max_next_action).squeeze(dim=1)
                
            quantiles_next = reward +  self.gamma * quantiles_next
        return quantiles_next
   
    def get_max_next_state_action(self, next_states):
        next_dist = self.dqn(next_states) * self.quantile_weight
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_quantiles)
        
        
    def select_action(self, state) -> int:
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                selected_action = (self.dqn(state) * self.quantile_weight).sum(dim=2).max(dim=1)[1].item()
        return selected_action
    
    def step(self, state: np.array) -> tuple:
        action = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        transition = [state, action, reward, next_state, done]
        self.memory.store(transition)
        return next_state, reward, done
    
    def target_update(self):
        self.target.load_state_dict(self.dqn.state_dict())