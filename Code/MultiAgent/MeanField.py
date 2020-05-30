import numpy as np

from typings import List
from Agent.ResidueAgent import ResidueAgent
from Environment.Bravais import Bravais


class GlobalBuffer(object):
    def __init__(self, env : Bravais, num_agents : int):
        self.action_selection = np.zeros(num_agents)
        self.neighbouring_sites = np.zeros(num_agents, dtype=np.object)
        self.env = env

    def get_neighbour_distribution(self, agent_idx : int) -> np.array:
        neighbours = self.neighbouring_sites[agent_idx] 
        actions = np.take(self.action_selection, neighbours)
        action_one_hot = np.zeros((actions.shape(0), self.env.action_space_n))
        action_one_hot[np.arange(actions.shape(0)), actions] = 1
        return action_one_hot.sum() / actions.shape(0)



class MultiAgentRoutine(object):
    def __init__(self, 
            residues : str,
            gamma : float,
            num_frames : int,
            **kwargs):

        self.env = Bravais(residues=residues, gamma=gamma)
        self.num_frames = num_frames
        self.global_buffer = GlobalBuffer(self.env, len(residues))
        self.agents : List[ResidueAgent] = [
                ResidueAgent(env = self.env, **kwargs) 
                for _ in range(self.env.num_residues)] 
        self.arguments = kwargs
        
        def get_action_dist(self, agent_idx : int, state : np.array, init : bool = False) -> np.array:
            if init:
                num_neighbours = (agent_idx != 0) + (agent_idx != len(self.agents) -1)
                joint_action = np.eye(self.env.action_space_n)[np.random.choice(self.env.action_space_n, num_neighbours)] #random action vector
                joint_action_vec = (joint_action.sum(axis=1) / num_neighbours).flatten()
                state = np.concatenate(state, joint_action_vec)
            else:
                action_dist = self.global_buffer.get_neighbour_distribution(agent_idx)
                state = np.concatenate(state, action_dist)
            return state

        def train(self, min_train:int, epochs : int) -> (np.array, np.array):
            state = self.env.reset()
            frames = np.zeros(len(self.agents))
            epoch_scores = np.zeros(epochs, dtype=np.object)
            epoch_losses = np.zeros(epochs, dtype=np.object)
            for e in range(epochs):
                agent_scores = np.zeros(len(self.agents))
                agent_losses = np.zeros(len(self.agents))
                for agent_idx in range(self.agents):
                    choosing_action = True
                    steps = 1
                    mean_score = 0
                    while choosing_action:
                        frames[agent_idx] += 1
                        state = self.get_action_dist(agent_idx,state,(frames[agent_idx], e) == (1,1))
                        next_state, reward, done, info = self.agents[agent_idx].step(state, len(self.agents[agent_idx].memory) > min_train)
                        state = next_state
                        mean_score += (reward - mean_score) / steps
                        
                        # PER: increase beta
                        self.agents[agent_idx].update_beta(frames[agent_idx], self.arguments['memory_beta_frames'])
                        frames[agent_idx] += 1 
                        # if episode ends
                        if done:
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
