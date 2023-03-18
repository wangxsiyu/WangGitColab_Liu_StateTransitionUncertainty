from task_Goal_Action import task_Goal_Action
from W_Gym import W_env_simulator
from Wang_Actor_Critic import Actor_Critic, train_Actor_Critic
import numpy as np
import torch

is_model = False

render_mode = "human"
n_trialperblock = 100
n_maxtrials = 100
env = task_Goal_Action(render_mode = render_mode, n_trial = n_trialperblock, n_maxtrials = n_maxtrials)
player = W_env_simulator(env)

if is_model:
    savename = '.\\models\\v1.pt'
    a2c = Actor_Critic(np.prod(env.observation_space.shape), env.action_space.n, 20, 20, device = "cpu")
    a2c.load_state_dict(torch.load(savename))
    player.play("model", a2c)
else:
    player.play("human")


