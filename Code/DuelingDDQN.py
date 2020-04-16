#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:24:16 2020

@author: adrian
"""


import gym

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import random
import  time
import sys
import os
import plotly.graph_objects as go
import numpy as np
import plotly
import plotly.graph_objs
import torch.nn.functional as F
from torch.autograd import Variable
from PER.PrioReplay import PrioritizedMemory


clear = lambda: os.system('cls') 

Tensor = torch.Tensor
LongTensor = torch.LongTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make('CartPole-v0')


seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
learning_rate = 0.001
num_episodes = 170
gamma = 0.9999

hidden_layer = 64

replay_mem_size = 5000
batch_size = 64

update_target_frequency = 500

double_dqn = True

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 500

report_interval = 10
score_to_solve = 195

clip_error = True

number_of_inputs = env.observation_space.shape[0]

number_of_outputs = env.action_space.n


def calculate_epsilon(steps_done):
    return egreedy_final + (egreedy-egreedy_final) * \
        math.exp(-1. * steps_done/egreedy_decay)
 
    
class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,hidden_layer)
        self.relu = nn.Tanh()
        self.advantageLayer = nn.Linear(hidden_layer, number_of_outputs)
        self.valueLayer = nn.Linear(hidden_layer, 1)
        
    def forward(self,x):
        output = self.linear1(x)
        output = self.relu(output)
        value = self.valueLayer(output)
        advantage = self.advantageLayer(output)
        return value + (advantage - advantage.mean())
    
    
class QNet_Agent(object):
       
    def __init__(self, memory):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)
        self.initWeights = None
        self.loss_func = nn.MSELoss()
        self.memory = memory
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.update_target_counter = 0
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()        
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self, frames):
        ## CER Sampling
        if (frames < batch_size):
            return
        
        tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
        state, action, reward, new_state, done = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
       
        state = Tensor(list(map(lambda x: x.tolist(), state))).to(device)
        new_state = Tensor(list(map(lambda x: x.tolist(), new_state))).to(device)
        reward = Tensor(reward.astype(float)).to(device)
        action = LongTensor(action.astype(np.long)).to(device)
        done = Tensor(done.astype(int)).to(device)
        

        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]  
            
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
        
        
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
  
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        target_value = Variable(target_value)

        #errors = torch.abs(target_value-predicted_value).data.cpu().numpy()
        
            
        element_wise_loss = F.smooth_l1_loss(predicted_value, target_value, reduction='none') 
        loss = (torch.FloatTensor(ISWeights_mb).to(device) * element_wise_loss.unsqueeze(0)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.memory.batch_update(tree_idx, element_wise_loss.detach().cpu().numpy())
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        self.update_target_counter += 1

memory = PrioritizedMemory(replay_mem_size)
qnet_agent = QNet_Agent(memory)
frames_total = 0
rewards_total = []
episode_losses = []
steps_total =[]
frames_total = 0 
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):
    
    state = env.reset()
    rewardE = 0
    step = 0
    #for step in range(100):
    while True:
        
        step += 1
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total)
        
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)
        
        new_state, reward, done, info = env.step(action)
        rewardE += reward
        qnet_agent.memory.store((state, action, reward, new_state, done))
        qnet_agent.optimize(frames_total)
        
        state = new_state
        if done:
            steps_total.append(step)
            rewards_total.append(rewardE)
            mean_reward_100 = sum(steps_total[-100:])/100
            
            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After %i episodes " % i_episode)
                solved_after = i_episode
                solved = True
            
            if (i_episode % report_interval == 0):
                
                
                
                print("\n*** Episode %i *** \
                      \nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f \
                      \nepsilon: %.2f, frames_total: %i" 
                  % 
                  ( i_episode,
                    report_interval,
                    sum(steps_total[-report_interval:])/report_interval,
                    mean_reward_100,
                    sum(steps_total)/len(steps_total),
                    epsilon,
                    frames_total
                          ) 
                  )
                  
                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            break
        
        
env.close()
env.env.close()

plotly.offline.plot({
"data": [
    plotly.graph_objs.Scatter(    x=list(range(len(rewards_total))),
    y=rewards_total, mode='lines',
    marker=dict(
        size=[40, 60, 80, 100]))],
"layout": plotly.graph_objs.Layout(showlegend=False,
    height=700,
    width=1800,
)
})

plt.figure(figsize=(12,5))
plt.title('Loss')
plt.bar(torch.arange(len(episode_losses)), episode_losses, alpha=0.6, color='blue')


plt.figure(figsize=(12,5))
plt.title('Steps by episode')
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red')