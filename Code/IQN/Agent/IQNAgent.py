
from Memory.PrioritisedBuffer import PrioritizedReplayBuffer
from Network.IQNetwork import IQNetwork
import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class IQNAgent:
    
    def __init__(self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        min_train:int = 1000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # Categorical DQN parameters
        quantiles: int = 51,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        is_test:bool = False):
        
        self.env = env
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.quantiles = quantiles
        self.beta=beta
        obs_dim = [4,84,84]
        action_dim = env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha, pixel=True)
        self.is_test= is_test
        self.dqn = IQNetwork(action_dim, quantiles)
        self.target = IQNetwork(action_dim, quantiles)
        self.target.load_state_dict(self.dqn.state_dict())
        self.cumulative_density = torch.tensor((2 * np.arange(self.quantiles) + 1) / (2.0 * self.quantiles), device=self.device, dtype=torch.float) 
        self.quantile_weight = 1.0 / self.quantiles
        self.min_train = min_train if min_train > memory_size else memory_size
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.preprocess_frame(next_state)
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done
    
    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch(self.beta)
        
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        elementwise_loss = self.compute_loss(samples)
        # PER: update priorities
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()
    
    def huber(self, x):
       cond = (x.abs() < 1.0).float().detach()
       return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)
   
    def preprocess_frame(self,x):
        x = np.array(x)
        return x.reshape(x.shape[2], x.shape[0], x.shape[1])
    
    def select_action(self, state: np.ndarray, frame_idx:int) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if frame_idx> self.min_train:
            if self.epsilon > np.random.random():
                selected_action = self.env.action_space.sample()
            else:
                selected_action = self.dqn(
                    torch.FloatTensor(state).to(self.device),
                ).argmax()
                selected_action = selected_action.detach().cpu().numpy()
        else:
            selected_action = self.env.action_space.sample()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action
    
    def get_max_next_state_action(self, next_states):
        next_dist = self.dqn(next_states) * self.quantile_weight
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.quantiles)
    
    def next_distribution(self, samples):
        device = self.device  # for shortening the following lines
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        non_final_next_states = next_state.gather(0, done)
        
        with torch.no_grad():
            quantiles_next = torch.zeros((self.batch_size, self.quantiles), device=self.device, dtype=torch.float)
            if not done[done<1].size(0) != 0:
                #self.target.sample_noise()
                max_next_action = self.get_max_next_state_action(non_final_next_states) #non_final_next_states
                quantiles_next[done] = self.target(non_final_next_states).gather(1, max_next_action).squeeze(dim=1)

            quantiles_next = reward + ((self.gamma)*quantiles_next)

        return quantiles_next
    
    def compute_loss(self, samples: Dict[str, np.ndarray]):
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        print(action.size())
        batch_action = action.unsqueeze(dim=-1).expand(-1, -1, self.quantiles)

        #self.dqn.sample_noise()
        quantiles = self.model(state)
        quantiles = quantiles.gather(1, batch_action).squeeze(1)

        quantiles_next = self.next_distribution(samples)
          
        diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0)

        loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
        loss = loss.transpose(0,1)
        self.memory.update_priorities(indices, loss.detach().mean(1).sum(-1).abs().cpu().numpy())
        loss = loss * weights.view(self.batch_size, 1, 1)
        loss = loss.mean(1).sum(-1).mean()

        return loss
    

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()