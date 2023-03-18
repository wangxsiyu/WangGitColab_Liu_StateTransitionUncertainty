from gym_MC.envs import MC
from visualize import visualize
from MC_RNN import DQN
import torch
import matplotlib.pyplot as plt

env = MC(render_mode="rgb_array")

is_record = False

dqn = torch.load('MC_trained_v1.pt')
print('model loaded')

rmode = "rgb_array" if is_record else "human"
venv = MC(render_mode=rmode)
visualize(venv, dqn, is_record = is_record)



