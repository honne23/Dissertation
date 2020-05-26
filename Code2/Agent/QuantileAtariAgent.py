import torch
import numpy as np
import gym
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from Network.QuantileCNN import QuantileCNN
from Memory.PrioritisedReplay import PrioritisedReplay
class QuantileAtariAgent:
    
    def __init__(self,
                 env: gym,
                 #hidden_size:int = 128,
                 hidden_size:int = 512,
                 mem_size: int = 5000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 num_quantiles: int = 51
                 ):
        self.env = env
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dqn = QuantileCNN(hidden_size, self.env.action_space.n, num_quantiles).to(self.device)
        self.target = QuantileCNN(hidden_size, self.env.action_space.n, num_quantiles).to(self.device)
        self.target.eval()
        self.target.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(params= self.dqn.parameters(), lr=lr)
        self.batch_size = batch_size
        self.memory = PrioritisedReplay(mem_size)
        self.gamma = gamma
        self.num_quantiles = num_quantiles
        self.cumulative_density = torch.tensor((2 * np.arange(num_quantiles) + 1) / (2.0 * num_quantiles), device=self.device, dtype=torch.float) 
        self.quantile_weight = 1.0 / self.num_quantiles
        
        self.nsteps = 3
        self.nstep_buffer = []
        
    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)
    
    def update_network(self) -> float:
        
        idx, weights, samples = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights.astype(float)).to(self.device)
        
        samples = np.vstack(samples)
        state, action, reward, next_state, done = samples.T
        
        state = np.vstack(state).reshape(self.batch_size, 4,84,84) 
        next_state = np.vstack(next_state).reshape(self.batch_size,4,84,84)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action.astype(int)).to(self.device)
        reward = torch.FloatTensor(reward.astype(float).reshape(-1, 1)).to(self.device)
        done = torch.LongTensor(done.astype(bool).reshape(-1, 1)).to(self.device)
        
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
        
        self.dqn.reset_noise()
        self.target.reset_noise()
        
        return loss.item()
    
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
        
        
    def select_action(self, state: np.array, ready: bool) -> int:
        if ready:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                selected_action = (self.dqn(state.unsqueeze(0)) * self.quantile_weight).sum(dim=2).max(dim=1)[1].item()
        else:
            selected_action = self.env.action_space.sample()
        return selected_action
    
    def step(self, state: np.array, ready:bool) -> tuple:
        action = self.select_action(state, ready)
        next_state, reward, done, _ = self.env.step(action)
        transition = [state, action, reward, next_state, done]
        #self.memory.store(transition)
        self.append_to_replay(state, action, reward, next_state, done)
        return next_state, reward, done
    
    def target_update(self):
        self.target.load_state_dict(self.dqn.state_dict())
    
    def append_to_replay(self, s, a, r, s_, done):
        self.nstep_buffer.append((s, a, r, s_, done))

        if(len(self.nstep_buffer)<self.nsteps):
            return
        
        R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _, done = self.nstep_buffer.pop(0)
        transition = [s, a, R, s_, done]
        self.memory.store(transition)
    
    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, next_state, done = self.nstep_buffer.pop(0)
            transition = [state, action, R, next_state, done]
            self.memory.store(transition)
