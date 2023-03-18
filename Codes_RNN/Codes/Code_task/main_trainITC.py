from Wang_Actor_Critic import Actor_Critic, train_Actor_Critic
import numpy as np
import torch 
import os
from W_Env.W_Env import W_Env

render_mode = None
n_maxT = 1000
env = W_Env('WV', render_mode = render_mode, \
                        n_maxT = n_maxT, is_ITI = True)
a2c = Actor_Critic(env._len_observation(), env._len_actions(), 20, 20, device = "cpu")

savename = '.\\models\\v1.pt'
n_trialperblock = 100
n_maxtrials = 100
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
    if i % 10 == 0:
        print(f"iter {i}: r = {treward}, len = {len(tmemory.memory['action'])}")
    if i % 100 == 0:
        trainer.model.save(savename)
