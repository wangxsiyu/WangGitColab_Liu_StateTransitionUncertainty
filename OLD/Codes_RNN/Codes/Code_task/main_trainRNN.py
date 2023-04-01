from task_Goal_Action import task_Goal_Action
from Wang_Actor_Critic import Actor_Critic, train_Actor_Critic
import numpy as np
import torch 
import os
savename = '.\\models\\v1.pt'
n_trialperblock = 100
n_maxtrials = 100

env = task_Goal_Action(render_mode="rgb_array")
env.setup_reward({'R_advance':1, 'R_error': -1, 'R_reward':100})
a2c = Actor_Critic(np.prod(env.observation_space.shape), env.action_space.n, 20, 20, device = "cpu")
if os.path.isfile(savename):
    print(f'load {savename}')
    a2c.load_state_dict(torch.load(savename))
lr = 0.004
weight_decay = 0
gamma = 0.99
trainer = train_Actor_Critic(a2c, lr, weight_decay, gamma)

n_iters = 10000

reward_history = np.zeros(n_iters)
for i in range(n_iters):
    hidden_policy = None
    hidden_value = None
    tmemory, treward = a2c.play(env, hidden_policy=hidden_policy, hidden_value=hidden_value)
    trainer.update_episode(tmemory.memory['obs'], tmemory.memory['action'], tmemory.memory['reward'])
    reward_history[i] = treward
    if i % 100 == 0:
        print(f"iter {i}: r = {treward}, len = {len(tmemory.memory['action'])}")
    if i % 1000 == 0:
        trainer.model.save(savename)
