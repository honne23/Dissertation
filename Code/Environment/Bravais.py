import numpy as np
from itertools import product
from numpy.linalg import inv, norm
from math import pi
from typing import List
from functools import reduce

class Bravais(object):
    
    def __init__(self, residues : str, gamma: float, init_positions : bool = True):
        
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
        
        self.types = self.init_types(residues)
        
        self.site_potentials = [self.get_sites(v[0],v[1],i)[0] \
                                for i,v in enumerate(zip(self.types, self.position_buffer)) ]
        self.global_reward = sum(self.site_potentials)
        self.done_flag = False
        
    
    def init_types(self, residues : str) -> List[int]:
        types = []
        for i in residues:
            for j in range(len(self.encoding)):
                if i in self.encoding[j]:
                    types.append(j)
        return types
    
    def step(self, action : int, index : int) -> tuple:
        """
        Step in the environment
        returns state, reward, info

        """
        if self.done_flag:
            #Transition into terminal state with reward = 0
            self.done_flag = False
            return np.zeros(self.position_buffer.shape[0]), 0, True 
        old_pos = self.position_buffer[index,:].reshape(-1,1)
        new_pos = old_pos + self.e.dot(self.actions[action,:].reshape(-1,1))
        reward, new_local, new_g, neighbours = self.calc_reward(index,old_pos,new_pos)
        self_avoiding = self.check_self_avoiding(index, new_pos)
        if reward == -10 or not self_avoiding:
            self.done_flag = False
        else:
            self.done_flag = True
            self.position_buffer[index] = new_pos.reshape(-1,1)
            self.site_potentials[index] = new_local
            self.global_reward = new_g
        return self.position_buffer.flatten(), reward, False, {'neighbours' : neighbours}
    
    def check_self_avoiding(self, index:int, new_pos : np.array) -> bool:
        """
        Measure the distance between previous index and next index to maintain the backbone
        """
        last_index = norm(new_pos-self.position_buffer[(index-1) % len(self.position_buffer)]) in [np.sqrt(0.5),1.]
        next_index = norm(new_pos - self.position_buffer[(index+1) % len(self.position_buffer)]) in [np.sqrt(0.5),1.]
        if index == 0:
            return next_index
        elif index == self.num_residues - 1:
            return last_index
        else:
            return next_index and last_index
        
    def reset(self) -> np.array:
        """
        Denature the protein
        """
        if self.init_positions == True:
            x1 = np.array([1,0,0])
            position_buffer = np.vstack([(i * x1) for i in range(self.num_residues)])
        else:
            position_buffer = np.zeros(self.num_residues)
        return position_buffer.flatten()
    
    def find_neighbours(self, position : np.array, index : int = -1) -> np.array:
        """
        Find neighbours of prospective position, including any overlapping sites
        """
        distances = norm(self.position_buffer - position, axis=1)
        neighbours = np.nonzero((distances == 1.) | (distances == np.sqrt(0.5)) | (distances == 0.)) #indexes of neighbours
        if index != -1:
            return neighbours[0][neighbours[0] != index]
        return neighbours[0]
    
    def calc_desireability(self, position:np.array, num_neighbours: int) -> float:
        """
        Calculate the desirebiilty of a prospective point according to agent density)
        """
        if num_neighbours == 0:
            return 0
        position = position
        mean_pos = np.mean(self.position_buffer, axis=0)
        Sigma = np.cov(self.position_buffer.T) #Covariance 
        offset = position - mean_pos
        try:
            numerator = np.exp(-offset.T.dot(inv(Sigma).dot(offset))).item()
        except:
            return 0
        coeff = (2 * pi * np.sqrt(norm(Sigma)))**-1
        denom = (1 + (num_neighbours/self.coordination_number))**-self.alpha #Max number of neighbours = 12 for FCC lattice
        
        return coeff * numerator * denom
    
    def get_sites(self, residue_type:int, position: np.array, index : int = -1) -> List[int]:
        """
        Calculate the sum of the rewards of occupying a particular area without using additional memory
        """
        neighbours = self.find_neighbours(position, index)
        sites = np.take(self.position_buffer, neighbours, axis=0)
        if sum((sites[:]==position).all(1)) > 1: 
            return -10 #if the site is overlapping or chain is broken
        site_potential = reduce(lambda x,y : x + self.rewards[residue_type,self.types[y]], np.insert(neighbours,0,0))
        return (site_potential, neighbours) # memory efficient reward calculation
    
    def global_difference_reward(self, index: int, new_local_rewards : int) -> int:
        old_global_prime = self.global_reward - self.site_potentials[index]
        new_global = old_global_prime + new_local_rewards
        return new_global - old_global_prime, new_global
    
    def calc_reward(self, index: int, old_pos: np.array, new_pos: np.array) -> float:
        """
        Shaped reward from: http://web.engr.oregonstate.edu/~ktumer/publications/files/tumer-devlin_aamas14.pdf
        """
        new_pos = new_pos.reshape(1,3)
        old_pos = old_pos.reshape(1,3)
        new_local_rewards, neighbours = self.get_sites(self.types[index],new_pos, index)
        phi_next = self.calc_desireability(new_pos, len(neighbours))
        phi_current = self.calc_desireability(old_pos, self.find_neighbours(self.position_buffer[index]).shape[0])
        g_diff, new_g = self.global_difference_reward(index, new_local_rewards) 
        shaped_reward = g_diff + self.gamma * phi_next - phi_current
        return shaped_reward, new_local_rewards, new_g, neighbours
    
    def sample_action(self):
        return np.random.randint(self.action_space_n)
