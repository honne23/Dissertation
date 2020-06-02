import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from Environment.Bravais import Bravais
from Network.QuantileNetwork import QuantileNetwork
from Memory.PrioritisedReplay import PrioritisedReplay
class ResidueAgent(object):
    
    def __init__(self,
                 env: Bravais,
                 idx :int,
                 hidden_size:int = 512,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.01,
                 epsilon_decay: float = (1/2000),
                 mem_size: int = 10000,
                 batch_size: int = 128,
                 gamma: float = 0.99,
                 lr: float = 1e-5,
                 num_quantiles: int = 51,
                 mean_field_beta : float = 0.6,
                 mean_field_tau : float = 0.01
                 ):
        self.env = env
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.dqn = QuantileNetwork(self.env.observation_space_n + self.env.action_space_n, \
                                   hidden_size, self.env.action_space_n, num_quantiles).to(self.device)
            
        self.target = QuantileNetwork(self.env.observation_space_n + self.env.action_space_n, \
                                      hidden_size, self.env.action_space_n, num_quantiles).to(self.device)
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
        self.mean_field_beta = mean_field_beta
        self.mean_field_tau = mean_field_tau
        self.idx=idx
        
    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)
    
    def update_network(self) -> float:
        
        idx, weights, samples = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights.astype(float)).to(self.device)
        
        samples = np.vstack(samples)
        state, action, reward, next_state, done = samples.T
        try:
            state = np.vstack(state)
        except:
            for x in state:
                print(state)
            print(state.size())
        
        neighbour_dists = torch.FloatTensor(state[:,-self.env.action_space_n:]).to(self.device)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(np.vstack(next_state)).to(self.device)
        action = torch.LongTensor(action.astype(int)).to(self.device)
        reward = torch.FloatTensor(reward.astype(float).reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(done.astype(bool).reshape(-1, 1)).to(self.device)
        
        quantiles = self.dqn(state)
        quantiles = quantiles[torch.arange(quantiles.size(0)) , action].squeeze(1) #select the quantiles of the actions chosen in each state
        quantiles_next = self.next_distribution(reward, next_state, done,neighbour_dists)
          
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
        self.dqn.reset_noise()
        self.target.reset_noise()
        return loss.item()
    
    def update_beta(self, frame_idx:int, num_frames:int):
        fraction = min(frame_idx / num_frames, 1.0)
        beta = self.memory.beta 
        beta = beta + fraction * (1.0 - beta)
        self.memory.beta = beta
    
    def next_distribution(self, reward, next_state, done, neighbour_dists):
        mask = (1-done).bool().squeeze(1)
        non_final = next_state[mask,:]
        with torch.no_grad():
            quantiles_next = torch.zeros((self.batch_size, self.num_quantiles), device=self.device, dtype=torch.float)
            if not (done.sum().item() == done.size(0)): #if there is at least one non-final next state
                #max_next_action = self.get_max_next_state_action(non_final)
                #quantiles_next[mask] = self.target(non_final).gather(1, max_next_action).squeeze(dim=1)
                neighbour_dists = neighbour_dists[mask]
                non_final = torch.cat((non_final, neighbour_dists), dim=1)
                target_quantiles = self.target(non_final)
                
                
                target_expected_returns = (target_quantiles * self.quantile_weight).sum(2)
                
                target_probs = F.softmax(-self.mean_field_beta * target_expected_returns, dim=1)
                
                weight = (neighbour_dists * target_probs)
                
                mean_field_quantiles = (target_quantiles * weight.view(weight.size(0), weight.size(1), 1)).sum(2).max(1)[1]
                quantiles_next[mask] = target_quantiles[torch.arange(target_quantiles.size(0)), mean_field_quantiles, :]
                
            quantiles_next = reward +  self.gamma * quantiles_next #edit for mean field
        return quantiles_next
    
    """
    def get_max_next_state_action(self, next_states):
        next_dist = self.dqn(next_states) * self.quantile_weight
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_quantiles)
    """        
        
    def select_action(self, state : np.array,ready : bool) -> int:
        if ready == False:
            selected_action = self.env.sample_action()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                quantiles = self.dqn(state)
                quantile_returns = (quantiles * self.quantile_weight).sum(dim=2)
                action_probs = F.softmax(- self.mean_field_beta * quantile_returns, dim=0) 
                neighbour_distribution = state[-self.env.action_space_n:]
                mean_field_value = (quantiles * neighbour_distribution.view(-1,1) * action_probs.view(-1,1)).sum(dim=2)
                selected_action = mean_field_value.max(dim=1)[1].item()
        return selected_action
    

    
    def target_update(self):
        target_params = self.target.state_dict()
        behaviour_params = self.dqn.state_dict()
        new_params = behaviour_params # temp
        for k,v in behaviour_params.items():
            new_params[k] = self.mean_field_tau * v + (1- self.mean_field_tau) * target_params[k] 
        self.target.load_state_dict(new_params)
