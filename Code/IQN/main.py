import gym
import torch
import random
import numpy as np
from Agent.IQNAgent import IQNAgent
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
memory_size = 10**5
batch_size = 32
target_update = 1000
epsilon_decay = 1 / 10**5   

# train
agent = IQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, min_train=50000)



is_test = False

state = env.reset()


update_cnt = 0
epsilons = []
losses = []
scores = []
score = 0
plotting_interval: int = 200
state = agent.preprocess_frame(state)
for frame_idx in range(1, num_frames + 1):
    action = agent.select_action(state, frame_idx)
    next_state, reward, done = agent.step(action)
    
    state = next_state
    score += reward
    
    # PER: increase beta
    fraction = min(frame_idx / num_frames, 1.0)
    agent.beta = agent.beta + fraction * (1.0 - agent.beta)

    # if episode ends
    if done:
        state = env.reset()
        scores.append(score)
        score = 0

    # if training is ready
    if len(agent.memory) >= batch_size:
        loss = agent.update_model()
        losses.append(loss)
        update_cnt += 1
        
        # linearly decrease epsilon
        agent.epsilon = max(
            agent.min_epsilon, agent.epsilon - (
                agent.max_epsilon - agent.min_epsilon
            ) * agent.epsilon_decay
        )
        epsilons.append(agent.epsilon)
        
        # if hard update is needed
        if update_cnt % agent.target_update == 0:
            agent._target_hard_update()

    # plotting
    if frame_idx % plotting_interval == 0:
        agent._plot(frame_idx, scores, losses, epsilons)
        
env.close()