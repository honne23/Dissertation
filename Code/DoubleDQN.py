#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 08:09:36 2020

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

clear = lambda: os.system('cls') #on Windows System

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('CartPole-v0')

seed_value = 23

env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

####Params####

num_episodes = 500
learning_rate = 0.001
gamma = 0.9999
egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 500
steps_total = []

capacity = 50000
memory_batch_size = 32
update_target_frequency = 500

##############

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
        
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
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
        self.linear1 = nn.Linear(number_of_inputs,64)
        self.tan= nn.Tanh()
        self.linear2 = nn.Linear(64, number_of_outputs)
        
    def forward(self,x):
        output1 = self.linear1(x)
        output1 = self.tan(output1)
        output2 = self.linear2(output1)
        return output2
    
    
class QNetAgent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.targetNN = NeuralNetwork().to(device)
        self.initWeights = None
        self.update_target_counter = 0
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr = learning_rate)
        
    def select_action(self, state, epsilon):
        ran = torch.rand(1)[0]
        
        if ran > epsilon:
            with torch.no_grad():
                state = torch.tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            action = env.action_space.sample()
        return action

    def optimize(self):
        if len(memory) < memory_batch_size:
            return
        state,action, new_state, reward, done = memory.sample(memory_batch_size)
        
        state = torch.Tensor(state).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = torch.Tensor(reward).to(device)
        action = torch.Tensor(action).long().to(device)
        done = torch.Tensor(done).to(device)
        
        new_state_indexes = self.nn(new_state).detach()
        max_new_state_indexes = torch.max(new_state_indexes, 1)[1]
        
        new_state_values = self.targetNN(new_state).detach()       
        max_new_state_values = new_state_values.gather(1,max_new_state_indexes.unsqueeze(1)).squeeze(1)
         
        target_value = reward + (1 - done) * gamma * max_new_state_values
        predicted_value = self.nn(state).gather(1,action.unsqueeze(1)).squeeze(1)
        loss = self.loss_func(predicted_value, target_value)
        ##Main steps
        self.optimizer.zero_grad()
        loss.backward()
        
        #for param in self.nn.parameters():
        #    param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.update_target_counter % update_target_frequency == 0:
            """
            lotteryDict = None
            newStateDict = {}
            for key in self.nn.state_dict().keys():
                if 'weight' in key:
                    lotteryDict = self.nn.state_dict()[key]
                    for i in lotteryDict:
                        for j in i:
                            if -0.4 < j < 0.4:
                                j = 0
                            else:
                                j = 1
                    newStateDict[key] = self.initWeights[key] * lotteryDict
                newStateDict[key] = self.initWeights[key]
            """
            self.targetNN.load_state_dict(self.nn.state_dict())
        self.update_target_counter += 1
        return loss

memory = ExperienceReplay(memory_batch_size)
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
        
        memory.push(state,action, new_state, reward, done)
        
        qnet_agent.optimize()
        
        rewardE += reward
        state = new_state
        #if(e > num_episodes - 10):
        #    time.sleep(0.03)
        #    env.render()
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
plt.title('Steps by episode')
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red')