from envs.task_evidence_accumulation import Evidence_accumulation
from Wang_Actor_Critic import Actor_Critic, train_Actor_Critic
import numpy as np
import torch 
savename = '.\\models\\v0.pt'

env = Evidence_accumulation(render_mode="rgb_array")
a2c = Actor_Critic(env.num_observations(), env.num_actions(), 20, 20, device = "cuda")
a2c.load_state_dict(torch.load(savename))
lr = 0.004
weight_decay = 0
gamma = 1
trainer = train_Actor_Critic(a2c, lr, weight_decay, gamma)

n_iters = 20000

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
