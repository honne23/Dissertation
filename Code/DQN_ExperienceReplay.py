#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:44:47 2020

@author: adrian

LOTTERY TICKET TARGET
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
num_episodes = 500
gamma = 0.9999

hidden_layer = 64

replay_mem_size = 500
batch_size = 32

update_target_frequency = 500

double_dqn = True

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 500

report_interval = 10
score_to_solve = 195

clip_error = False

number_of_inputs = env.observation_space.shape[0]

number_of_outputs = env.action_space.n


def calculate_epsilon(steps_done):
    return egreedy_final + (egreedy-egreedy_final) * \
        math.exp(-1. * steps_done/egreedy_decay)
        
class ExperienceReplay(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state,action, new_state, reward, done):
        transition = (state,action, new_state, reward, done)
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))
    
    def __len__(self):
        return len(self.memory)
    
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
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)
        self.initWeights = None
        self.loss_func = nn.MSELoss()
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
    
    def optimize(self, c_state, c_action, c_new_state, c_reward, c_done):
        ## CER Sampling
        if (len(memory) < batch_size):
            return
        
        state, action, new_state, reward, done = memory.sample(batch_size)
        state = list(state)
        state.append(c_state)
        action = list(action)
        action.append(c_action)
        new_state = list(new_state)
        new_state.append(c_new_state)
        reward = list(reward)
        reward.append(c_reward)
        done = list(done)
        done.append(c_done)
        
        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


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
        
        loss = self.loss_func(predicted_value, target_value)
    
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        self.update_target_counter += 1

memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()
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
        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize(state, action, new_state, reward, done)
        
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