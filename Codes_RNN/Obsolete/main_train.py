from gym_MC.envs import MC
from visualize import visualize
from MC_RNN import DQN
import torch
import matplotlib.pyplot as plt

env = MC(render_mode="rgb_array")

is_restart = False
batch_size = 1
eps_start = 0.1
if is_restart:
    dqn = DQN(env, batch_size = batch_size)
else:
    dqn = torch.load('MC_trained.pt')
    print('model loaded')

dqn.train(1000000, n_log = 5000, eps_start = eps_start)
torch.save(dqn.dqn_policy.state_dict(), 'MC_trained_weights.torch')
torch.save(dqn,'MC_trained.pt')




