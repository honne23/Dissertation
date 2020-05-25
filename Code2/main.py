import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from Agent.QuantileAtariAgent import QuantileAtariAgent
from atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_id = 'PongNoFrameskip-v4'
env = make_atari(env_id)
env = wrap_deepmind(env, frame_stack=True)
env = wrap_pytorch(env)
directory= ('./PongVideos')
env = gym.wrappers.Monitor(env, directory, video_callable=lambda idx : idx % 10 == 0, force=True )

seed = 42

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)
env.seed(seed)

# parameters
num_frames = 600000
memory_size = 30000
batch_size = 32
target_update = 1000


num_frames = 500000
plotting_interval = 1000

agent = QuantileAtariAgent( 
    env=env, 
    gamma=0.99, 
    mem_size = memory_size,
    batch_size=batch_size)

"""Train the agent."""


state = env.reset()
update_cnt = 0
epsilons = []
losses = []
scores = []
score = 0


def _plot(
        frame_idx, 
        scores, 
        losses 
        #epsilons,
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        #plt.subplot(133)
        #plt.title('epsilons')
        #plt.plot(epsilons)
        plt.show()

min_train = 10000
for frame_idx in range(1, num_frames + 1):
    next_state, reward, done = agent.step(state, frame_idx > min_train)
    state = next_state
    score += reward
    
    # PER: increase beta
    agent.update_beta(frame_idx, 100000)
    
    # if episode ends
    if done:
        state = env.reset()
        scores.append(score)
        score = 0
        agent.finish_nstep()
    # if training is ready
    if frame_idx >= min_train:
        loss = agent.update_network()
        losses.append(loss)
        update_cnt += 1
        
        # if hard update is needed
        if update_cnt % target_update == 0:
            agent.target_update()

    # plotting
    if frame_idx % plotting_interval == 0:
        _plot(frame_idx, scores, losses) #epsilons
        
env.close()
