import numpy as np

from typing import List
from Agent.ResidueAgent import ResidueAgent
from Environment.Bravais import Bravais


class GlobalBuffer(object):
    def __init__(self, env : Bravais, num_agents : int):
        self.action_selection = np.zeros(num_agents)
        self.neighbouring_sites = np.zeros(num_agents, dtype=np.object)
        self.env = env

    def get_neighbour_distribution(self, agent_idx : int) -> np.array:
        neighbours = self.neighbouring_sites[agent_idx]  
        actions = np.take(self.action_selection, neighbours, axis=0) 
        action_one_hot = np.zeros((actions.shape[0], self.env.action_space_n))
        action_one_hot[np.arange(actions.shape[0]), actions] = 1
        return action_one_hot.sum() / actions.shape[0]



class MultiAgentRoutine(object):
    def __init__(self, 
            residues : str,
            gamma : float,
            num_epochs : int,
            memory_beta_frames : int, 
            **kwargs):

        self.env = Bravais(residues=residues, gamma=gamma)
        self.num_epochs = num_epochs
        self.global_buffer = GlobalBuffer(self.env, len(residues))
        self.agents : List[ResidueAgent] = [
                ResidueAgent(env = self.env, idx=i, gamma = gamma, **kwargs) 
                for i in range(self.env.num_residues)] 
        self.arguments = kwargs
        self.memory_beta_frames = memory_beta_frames
        
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
        print(state.shape)
        return state

    def train(self, min_train:int) -> (np.array, np.array):
        print(self.env.observation_space_n)
        state = self.env.reset()
        frames = np.zeros(len(self.agents),dtype=int)
        epoch_scores = np.zeros(self.num_epochs, dtype=np.object)
        epoch_losses = np.zeros(self.num_epochs, dtype=np.object)
        for e in range(self.num_epochs):
            agent_scores = np.zeros(len(self.agents))
            agent_losses = np.zeros(len(self.agents))
            for agent_idx in range(len(self.agents)):
                choosing_action = True
                steps = 1
                mean_score = 0
                while choosing_action:
                    frames[agent_idx] += 1
                    state = self.get_action_dist(agent_idx,state,e == 0)
                    next_state, reward, done, info = self.agents[agent_idx].step(state, len(self.agents[agent_idx].memory) > min_train)
                    state = next_state
                    mean_score += (reward - mean_score) / steps
                    
                    # PER: increase beta
                    self.agents[agent_idx].update_beta(frames[agent_idx], self.memory_beta_frames)
                    frames[agent_idx] += 1 
                    # if episode ends
                    if done:
                        print('done!')
                        agent_scores[agent_idx] = mean_score
                        choosing_action = done - 0 #invert
                        self.global_buffer.neighbouring_sites[agent_idx] = info['neighbours']
                        self.global_buffer.action_selection[agent_idx] = info['action']

                    # if training is ready
                    if frames[agent_idx] >= min_train:
                        loss = self.agents[agent_idx].update_network()
                        agent_losses[agent_idx] = loss
                        # if hard update is needed
                        if frames[agent_idx] -(min_train + 1) % self.arguments['target_update'] == 0:
                            self.agents[agent_idx].target_update()
                    steps += 1
            epoch_scores[e] = agent_scores
            epoch_losses[e] = agent_losses
        return epoch_scores, epoch_losses


experiment = MultiAgentRoutine('GVIDTSAVESAITDGQGDMKAIGGYIVGALVILAVAGLIYSMLRKA', 0.99, 100000, memory_beta_frames=10000)
experiment.train(10000)