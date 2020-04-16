#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:04:45 2020

@author: adrian
"""

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import  time
import sys
import os
clear = lambda: os.system('cls') #on Windows System

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('CartPole-v0')

seed_value = 23

env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

num_episodes = 2000
learning_rate = 0.02
gamma = 0.99
egreedy = 0.9
egreedy_final = 0
egreedy_decay = 500
steps_total = []

number_of_inputs = env.observation_space.shape[0]

number_of_outputs = env.action_space.n


def calculate_epsilon(steps_done):
    return egreedy_final + (egreedy-egreedy_final) * \
        math.exp(-1. * steps_done/egreedy_decay)
 
class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,64).float()
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,64).float()
        self.linear3 = nn.Linear(64, number_of_outputs).float()
        
    def forward(self,x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        return output
    
    
class QNetAgent(object):
    def __init__(self):
        self.nn = NeuralNetwork()
        self.nn.to(device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr = learning_rate)
        
    def select_action(self, state, epsilon):
        ran = torch.rand(1).item()
        
        if ran > epsilon:
            with torch.no_grad():
                state = torch.tensor(state).to(device).float()
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            action = env.action_space.sample()
        return action

    def optimize(self, state, action, new_state, reward, done):
        
        state = torch.Tensor(state).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = torch.Tensor([reward]).to(device)
        
        if done:
            target_value = reward
        else:
            new_state_values = self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values)
            target_value = reward + gamma * max_new_state_values
            
        predicted_value = self.nn(state)[action]
        loss = self.loss_func(predicted_value, target_value)
        ##Main steps
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

qnet_agent = QNetAgent()

frames_total = 0
rewards_total = []
episode_losses = []
start_time = time.time()
for e in range(num_episodes):
    state = env.reset()
    step = 0
    rewardE = 0
    lossE = 0
    while True:
        step += 1
        frames_total += 1
        epsilon = calculate_epsilon(e)
        action = qnet_agent.select_action(state, epsilon)
        
        new_state, reward, done, info = env.step(action)
        
        lossE += qnet_agent.optimize(state,action, new_state, reward, done)
        rewardE += reward
        state = new_state
        if(e > num_episodes - 200):
            time.sleep(0.03)
            env.render()
        if done:
            steps_total.append(step)
            rewards_total.append(rewardE)
            episode_losses.append(lossE)
            if e % 10 == 0:
                sys.stdout.write(" \n*** Episode %i *** \
                      \n Av.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f \
                     \nepsilon: %.2f, frames_total: %i \n"
                      % ( e,
                         10,
                         sum(steps_total[-10:])/10,
                         sum(steps_total[-100:])/100,
                         sum(steps_total) /len(steps_total), 
                         epsilon,
                         frames_total
                          ))
                elapsed_time = time.time() - start_time
                sys.stdout.write("Elapsed time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            break
        
        
env.close()
env.env.close()

plt.figure(figsize=(12,5))
plt.title('Rewards')
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green')

plt.figure(figsize=(12,5))
plt.title('Loss')
plt.bar(torch.arange(len(episode_losses)), episode_losses, alpha=0.6, color='blue')


plt.figure(figsize=(12,5))
plt.title('Steps by episode')
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red')