
import numpy as np
from PER.SumTree import SumTree
import random

class PrioritizedMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def store(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return np.asarray(batch), idxs, is_weight

    def batch_update(self, idx, error):
        p = self._get_priority(error)
        for ti, p in zip(idx, p):
            self.tree.update(ti, p)
"""
class PrioritizedMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
   
    #Source: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
    
    #/////////
    #This SumTree code is modified version and the original code is from:
    #https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
   
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
      
        #Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        #And also a data array
        #We don't use deque because it means that at each timestep our experiences change index by one.
        #We prefer to use a simple array and to overwrite when the memory is full.
       
        self.tree = SumTree(capacity)
  
    #Store a new experience in our tree
    #Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
   
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

        
    #- First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    #- Then a value is uniformly sampled from each range
    #- We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    #- Then, we calculate IS weights for each minibatch element

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
      
        # Calculating the max_weight
        tree = self.tree.tree[-self.tree.capacity:]
        p_min = np.min(tree) / self.tree.total_priority
        
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            #A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            #Experience that correspond to each value is retrieved
   
            index, priority, data = self.tree.get_leaf(value)
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        return b_idx, np.array(memory_b).squeeze(1), b_ISWeights
    
    #Update the priorities on the tree
   
    def __len__(self):
        return len(self.tree)
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors = abs_errors + self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            
            
"""