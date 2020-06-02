import numpy as np
from itertools import product
from numpy.linalg import inv, norm
from math import pi
from typing import List
from functools import reduce

class Bravais(object):
    
    def __init__(self, residues : str, gamma: float, init_positions : bool = True, limit : int = 150):
        
        self.e = np.array([
            [0., 0.5, 0.5],
            [0.5, 0., 0.5],
            [0.5, 0.5, 0.]
            ])
        self.rewards = np.array([
            [-2,4,0,0,0],
            [4,3,0,0,0],
            [0,0,-1,1,0],
            [0,0,1,-1,0],
            [0,0,0,0,0]
            ])
        
        self.alpha = -0.4
        self.gamma = gamma
        self.encoding =  ['AV','GILMFPW', 'RHK','DE','NCQSTY'] #hHPNX
        self.actions = np.array(list(product([-1,0,1], repeat=3)), dtype=int)
        self.num_residues = len(residues)
        self.coordination_number = 12 
        self.init_positions = init_positions
        self.position_buffer = self.reset().reshape(self.num_residues, 3)
        self.observation_space_n = self.position_buffer.reshape(-1,1).shape[0]
        self.action_space_n = self.actions.shape[0]
        self.phi_current = np.zeros(self.num_residues)
        self.types = self.init_types(residues)
        
        self.site_potentials = self.get_sites(self.position_buffer)[0]
        self.global_reward = sum(self.site_potentials)
        self.count_down = 0
        self.limit = limit
        
    
    def init_types(self, residues : str) -> List[int]:
        types = []
        for i in residues:
            for j in range(len(self.encoding)):
                if i in self.encoding[j]:
                    types.append(j)
        return np.array(types)
    
    def step(self, joint_action : np.array) -> tuple:
        """
        Step in the environment
        returns state, reward, info

        """
        
        action_vectors = np.take(self.actions, joint_action, axis=0)
        
        movement_vectors = self.e.dot(action_vectors.T).T
        new_pos = self.position_buffer + movement_vectors
        
        self_avoiding = np.array([self.check_self_avoiding(new_pos,i) for i in range(new_pos.shape[0])], dtype= np.bool)
        
        reward, new_local, new_g, neighbours = self.calc_reward(new_pos)
        reward[~self_avoiding] += -10
    
        #print(reward)
        done = False
        if sum(self_avoiding) == 0:
            self.count_down = 0
            reward = np.zeros(self.num_residues)
            done = True
            return self.position_buffer.flatten(), reward, done, {'neighbours' : neighbours, 'self-avoiding': sum(self_avoiding)}
        
        if sum(reward) < 0 :
            self.count_down += 1
            
        if self.count_down % self.limit == 0 and self.count_down != 0:
            self.count_down = 0
            reward = np.zeros(self.num_residues)
            done = True
            return self.position_buffer.flatten(), reward, done, {'neighbours' : neighbours, 'self-avoiding': sum(self_avoiding)}
        #self.position_buffer = proposed
        self.site_potentials = new_local
        self.global_reward = new_g
        self.position_buffer = new_pos
        return self.position_buffer.flatten(), reward, done, {'neighbours' : neighbours, 'self-avoiding': sum(self_avoiding)}
    
    def check_self_avoiding(self, new_pos : np.array, index : int) -> bool:
        """
        Measure the distance between previous index and next index to maintain the backbone
        """
        last_index = norm(new_pos[index] -new_pos[(index-1) % new_pos.shape[0]]) in [np.sqrt(0.5),1.]
        next_index = norm(new_pos[index] - new_pos[(index+1) % new_pos.shape[0]]) in [np.sqrt(0.5),1.]
        overlap = sum((norm(new_pos - new_pos[index], axis=1) == 0).astype(int)) == 1
        if index == 0:
            return next_index and overlap
        elif index == self.num_residues - 1:
            return last_index and overlap
        else:
            return next_index and last_index and overlap
        
    def reset(self) -> np.array:
        """
        Denature the protein
        """
        if self.init_positions == True:
            x1 = np.array([1,0,0])
            position_buffer = np.vstack([(i * x1) for i in range(self.num_residues)])
        else:
            position_buffer = np.zeros(self.num_residues)
        self.position_buffer = position_buffer
        self.site_potentials = self.get_sites(self.position_buffer)[0]
        self.global_reward = sum(self.site_potentials)
        return position_buffer.flatten()
    
    def find_neighbours(self, conformation : np.array, index : int ) -> np.array:
        """
        Find neighbours of prospective position, including any overlapping sites
        """
        distances = norm(conformation - conformation[index], axis=1)
        neighbours = np.nonzero((distances == 1.) | (distances == np.sqrt(0.5)) | (distances == 0.)) #indexes of neighbours
        neighbour_indexes = neighbours[0][neighbours[0] != index] #cannot be our own neighbours
        neighbour_positions = np.take(conformation, neighbour_indexes, axis=0)
        return  index, neighbour_indexes, neighbour_positions
    
    def calc_desireability(self, 
                           positions:np.array, 
                           num_neighbours: int,
                           mean_pos: np.array,
                           covar: np.array,
                           offset: np.array) -> float:
        """
        Calculate the desirebiilty of a prospective point according to new agent density)
        """
        try:
            numerator = np.exp(-offset.T.dot(inv(covar).dot(offset))).item()
        except:
            return 0
        coeff = (2 * pi * np.sqrt(norm(covar)))**-1
        denom = (1 + (num_neighbours/self.coordination_number))**-self.alpha #Max number of neighbours = 12 for FCC lattice
        
        return coeff * numerator * denom
    
    def get_sites(self, new_positions: np.array) -> List[int]:
        """
        Calculate the sum of the rewards of occupying a particular area without using additional memory
        """
        
        indices, neighbours, sites = zip(*[self.find_neighbours(new_positions, i) for i in range(new_positions.shape[0])])
        rewards = np.zeros(new_positions.shape[0])
        for residue in range(len(sites)):
            rewards[indices[residue]] += self.rewards[self.types[residue], self.types[neighbours[residue]]].sum()
        return (rewards, neighbours) # memory efficient reward calculation
    
    def global_difference_reward(self, new_local_rewards : int) -> int:
        old_global_prime = self.global_reward - self.site_potentials
        new_global = old_global_prime + new_local_rewards
        return new_global - old_global_prime, new_global
    
    def calc_reward(self, new_pos: np.array) -> float:
        """
        Shaped reward from: http://web.engr.oregonstate.edu/~ktumer/publications/files/tumer-devlin_aamas14.pdf
        """
        new_local_rewards, neighbours = self.get_sites(new_pos)
        #print(new_local_rewards)
        mean_pos = np.mean(new_pos, axis=0)
        covar = np.cov(new_pos.T) #Covariance 
        offset = new_pos - mean_pos
        
        phi_next = np.array([self.calc_desireability(new_pos[i], len(neighbours[i]), mean_pos, covar, offset)  \
                             for i in range(len(neighbours))], dtype=np.float32)
        
        g_diff, new_g = self.global_difference_reward(new_local_rewards) 
        shaped_reward = g_diff + self.gamma * phi_next - self.phi_current
        self.phi_current = phi_next
        return shaped_reward, new_local_rewards, new_g, neighbours
    
    def sample_action(self):
        return np.random.randint(self.action_space_n)
