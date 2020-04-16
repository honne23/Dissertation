#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:54:33 2020

@author: adrian
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import random
import time
import sys
import gym
import plotly.graph_objects as go
import numpy as np
import plotly
import plotly.graph_objs
import torch.nn.functional as F
from torch.autograd import Variable
from PER.PrioReplay import PrioritizedMemory
from torch.nn.utils import clip_grad_norm_

from atari_wrappers import make_atari, wrap_deepmind
"""
Source https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""


clear = lambda: os.system('cls') 

Tensor = torch.Tensor
LongTensor = torch.LongTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env, frame_stack=True)
directory= ('./PongVideos')
env = gym.wrappers.Monitor(env, directory, video_callable=lambda idx : idx % 20 == 0, force=True )


seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
learning_rate = 0.0001
num_episodes = 300
gamma = 0.9999

hidden_layer = 512

replay_mem_size = 100000
batch_size = 64

update_target_frequency = 50

atom_size = 51
v_min = 0.0
v_max= 200.0


egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 10000

report_interval = 10
score_to_solve = 195
support = torch.linspace(v_min, v_max, atom_size).to(device)
####################

double_dqn = True
clip_error = True

number_of_inputs = env.observation_space.shape[0]

number_of_outputs = env.action_space.n


def calculate_epsilon(steps_done):
    return egreedy_final + (egreedy-egreedy_final) * \
        math.exp(-1. * steps_done/egreedy_decay)

def preprocess_frame(frame):
    frame = np.asarray(frame)
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame)
    return frame.to(device, dtype=torch.float32).unsqueeze(0) 
    
class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            )
        self.advantage_hidden_layer = nn.Linear(7*7*64, hidden_layer)
        self.advantage_layer = nn.Linear(hidden_layer, number_of_outputs * atom_size)
        self.value_hidden_layer = nn.Linear(7*7*64, hidden_layer)
        self.value_layer = nn.Linear(hidden_layer, atom_size)
        
        #Xavier uniform weight initialisation
        for i in self.feature_layer:
            if isinstance(i, nn.Conv2d):
                torch.nn.init.xavier_uniform_(i.weight)
        torch.nn.init.xavier_uniform_(self.advantage_hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.advantage_layer.weight)
        torch.nn.init.xavier_uniform_(self.value_hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.value_layer.weight)
        
    def forward(self, x):
        output = self.dist(x)
        return torch.sum(output * support, dim=2)
    
    def dist(self, x):
        feature = self.feature_layer(x)
        feature = feature.view(feature.size(0), -1)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        advantage = self.advantage_layer(adv_hid).view(
            -1, number_of_outputs, atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, atom_size)
        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist
        
        
    
    
class QNet_Agent(object):
       
    def __init__(self, memory):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)
        self.memory = memory
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.update_target_counter = 0
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        if random_for_egreedy > epsilon: 
            state = preprocess_frame(state)
            selected_action = torch.max(self.nn(state),1)[1]
            selected_action = selected_action.detach().cpu().numpy() 
            
        else:
            selected_action  = env.action_space.sample()
        
        return selected_action 
    
    def optimize(self, frames, obv):
        ## CER + PER sampling
        if (frames < batch_size):
            return
        
        batch, tree_idx, ISWeights_mb = memory.sample(batch_size)
        if len(batch.shape) ==1:
            state, action, reward, new_state, done = [], [], [], [], []
            for x in batch:
                if x == 0:
                    return
                state.append(x[0])
                action.append(x[1])
                reward.append(x[2])
                new_state.append(x[3])
                done.append(x[4])
        else:
            state = batch[:,0]
            action = batch[:,1]
            reward = batch[:,2]
            new_state = batch[:,3]
            done = batch[:,4]
        state = torch.cat(list(map(lambda x: preprocess_frame(x), state))).to(device)
        action = LongTensor(action.astype(np.long))
        reward = Tensor(reward.astype(float)).reshape(-1,1).to(device)
        new_state = torch.cat(list(map(lambda x: preprocess_frame(x), new_state))).to(device)
        done = Tensor(done.astype(int)).reshape(-1,1).to(device)
        
        delta_z = float(v_max - v_min) / (atom_size - 1)
        
        with torch.no_grad():
            next_action = torch.max(self.nn(new_state),1)[1]
            next_dist = self.target_nn.dist(new_state)
            next_dist = next_dist[range(batch_size), next_action]
            target_value =  reward + ( 1 - done ) * gamma * support
            target_value = target_value.clamp(min= v_min, max = v_max)
            
            b = ((target_value - v_min) / delta_z).to(device)
            l = b.floor().long().to(device)
            u = b.ceil().long().to(device)
            
            l[(u > 0) * (l == u)] -= 1
            u[(l < (atom_size - 1)) * (l == u)] += 1
            offset = (
                    torch.linspace(
                        0, (batch_size - 1) * atom_size, batch_size
                    ).long()
                    .unsqueeze(1)
                    .expand(batch_size, atom_size)
                    .to(device)
                )
            
            proj_dist = torch.zeros(next_dist.size(), device=device)
            
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u - b)).view(-1)
            )
            
            proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )   
            
            
        dist = self.nn.dist(state)
        log_p = torch.log(dist[range(batch_size), action]).to(device)
        element_wise_loss = -(proj_dist * log_p).sum(1)
       
        loss = torch.mean(torch.FloatTensor(ISWeights_mb).to(device) * element_wise_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.nn.parameters(), 10.0)
        self.optimizer.step()
        
        self.memory.batch_update(tree_idx, element_wise_loss.detach().cpu().numpy())
        
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
        target = qnet_agent.nn(Variable(preprocess_frame(state))).data
        old_val = target[0][action]
        target_val = qnet_agent.target_nn.dist(Variable(preprocess_frame(new_state))).data
        target[0][action] = reward + (1-done)* gamma * torch.max(target_val)

        error = abs(old_val - target[0][action])
        error = error.cpu().numpy()
        qnet_agent.memory.store(error, [state, action, reward, new_state, done])
        qnet_agent.optimize(frames_total, (state, action, reward, new_state, done))
        
        state = new_state
        if done:
            rewards_total.append(rewardE)
            mean_reward_100 = sum(rewards_total[-100:])/100
            
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
                    sum(rewards_total[-report_interval:])/report_interval,
                    mean_reward_100,
                    sum(rewards_total)/len(rewards_total),
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