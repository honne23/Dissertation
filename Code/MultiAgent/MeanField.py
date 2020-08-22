
import magent
import math 
import time

import matplotlib.pyplot as plt
import numpy as np

from typing import List
from Agent.ResidueAgent import ResidueAgent
from Environment.Bravais import Bravais

import wandb
wandb.init(project="mean-field")

class GlobalBuffer(object):
    def __init__(self, env : Bravais, num_agents : int):
        self.action_selection = np.zeros(num_agents)
        self.neighbouring_sites = np.zeros(num_agents, dtype=np.object)
        self.env = env

    def get_neighbour_distribution(self, agent_idx : int) -> np.array:
        neighbours = self.neighbouring_sites[agent_idx]
        if len(neighbours) == 0:
            return np.zeros(self.env.action_space_n)
        actions = np.take(self.action_selection, neighbours, axis=0) 
        action_one_hot = np.zeros((actions.shape[0], self.env.action_space_n))
        action_one_hot[np.arange(actions.shape[0]), actions] = 1
        return action_one_hot.sum(axis=0) / neighbours.shape[0]



class MultiAgentRoutine(object):
    def __init__(self, 
            residues : str,
            gamma : float,
            num_epochs : int,
            memory_beta_frames : int,
            target_update : int,
            **kwargs):

        self.env = Bravais(residues=residues, gamma=gamma)
        self.num_epochs = num_epochs
        self.global_buffer = GlobalBuffer(self.env, len(residues))
        self.agents : List[ResidueAgent] = [
                ResidueAgent(env = self.env, idx=i, gamma = gamma, **kwargs) 
                for i in range(self.env.num_residues)] 
        self.arguments = kwargs
        self.memory_beta_frames = memory_beta_frames
        self.target_update = target_update
        self.buffer = np.zeros(self.env.num_residues,dtype=np.object)
        
    def get_action_dist(self, agent_idx : int, state : np.array, init : bool = False) -> np.array:
        if init ==True:
            #Random action distributions are used in initialisation
            num_neighbours = (agent_idx != 0) + (agent_idx != len(self.agents) -1)
            joint_action = np.eye(self.env.action_space_n)[np.random.choice(self.env.action_space_n, num_neighbours)] #random action vector
      
            joint_action_vec = (joint_action.sum(axis=0) / num_neighbours).flatten()
            state = np.concatenate((state, joint_action_vec), axis = 0)
        else:
            action_dist = self.global_buffer.get_neighbour_distribution(agent_idx)
            state = np.concatenate((state, action_dist), axis=0) 
        return state

    def train(self, min_train:int) -> (np.array, np.array):
        state = self.env.reset()
        epoch_scores = []
        epoch_losses = []
        agent_losses = np.zeros(len(self.agents))
        joint_action = np.zeros(len(self.agents), dtype=int)
        mean_score = 0
        for e in range(self.num_epochs):
            
            for agent_idx in range(self.env.num_residues):
                input_state = self.get_action_dist(agent_idx, state, e == 0)
                self.buffer[agent_idx] = input_state
                joint_action[agent_idx] = self.agents[agent_idx].select_action(input_state, e >= min_train)    
      
            next_state, reward, done, info = self.env.step(joint_action)
            if 'neighbours' in info.keys():
                self.global_buffer.neighbouring_sites = info['neighbours']
            self.global_buffer.action_selection = joint_action
            
            for agent_idx in range(self.env.num_residues):
                transition = [self.buffer[agent_idx], joint_action[agent_idx], reward[agent_idx], next_state, done]
                self.agents[agent_idx].memory.store(transition)
                self.agents[agent_idx].update_beta(e, self.memory_beta_frames)
                self.agents[agent_idx].decay_boltzman(e)
            
            state = next_state
            mean_score += (sum(reward) / len(reward) - mean_score) / (e+1) if sum(reward) != 0 else 0
      
            # if episode ends
            if done:
                state = self.env.reset()

            # if training is ready
            if e >= min_train:
                for agent_idx in range(len(self.agents)):
                    loss = self.agents[agent_idx].update_network()
                    agent_losses[agent_idx] = loss
                    # if hard update is needed
                    if e -(min_train + 1) % self.target_update == 0:
                        self.agents[agent_idx].target_update()
                if e % 20 == 0:
                    self.env.render()
                epoch_scores.append(sum(reward))
                epoch_losses.append(sum(agent_losses) / len(agent_losses))
                wandb.log({"Loss": epoch_losses[-1], 'Rewards': epoch_scores[-1], "Boltzman constant": self.agents[0].mean_field_beta })
                if e % 10 == 0:
                    self._plot(e, epoch_scores, epoch_losses)
            
            
        return epoch_scores, epoch_losses
    
    def _plot(
        self,
        frame_idx, 
        scores, 
        average_losses 
        #epsilons,
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(average_losses)
        #plt.subplot(133)
        #plt.title('epsilons')
        #plt.plot(epsilons)
        plt.show()
        
        
experiment = MultiAgentRoutine('GVIDTSAVESAITDGQGDMKAIGGYIVGALVILAVAGLIYSMLRKA', 
                               0.95, 
                               100000, 
                               memory_beta_frames=10000,
                               target_update = 5)
experiment.train(min_train=5000)