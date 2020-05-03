
from Agents.CategoricalDuelingPixelAgent import CDPAgent

import gym
import torch
import random
import numpy as np

ATARI = True

if ATARI == False:
   env_id = "CartPole-v0"
   env = gym.make(env_id) 
else:
    from atari_wrappers import make_atari, wrap_deepmind
    env_id = 'PongNoFrameskip-v4'
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True)
    directory= ('./PongVideos')
    env = gym.wrappers.Monitor(env, directory, video_callable=lambda idx : idx % 10 == 0, force=True )


seed = 42

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)
env.seed(seed)

# parameters
num_frames = 600000
memory_size = 10**6
batch_size = 32
target_update = 1000
epsilon_decay = 1 / 10**5   

# train
agent = CDPAgent(env, memory_size, batch_size, target_update, epsilon_decay, min_train=50000) #swap out the agent

agent.train(num_frames, plotting_interval=1000)