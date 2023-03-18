from envs.task_evidence_accumulation import Evidence_accumulation
from Wang_Actor_Critic import Actor_Critic, train_Actor_Critic
import numpy as np
import torch 
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

savename = '.\\models\\v0.pt'

env = Evidence_accumulation(render_mode="rgb_array")
a2c = Actor_Critic(env.num_observations(), env.num_actions(), 20, 20, device = "cuda")
a2c.load_state_dict(torch.load(savename))

n_iters = 1000
actions = np.zeros(n_iters)
dLR = np.zeros(n_iters)
for i in range(n_iters):
    hidden_policy = None
    hidden_value = None
    tmemory, treward = a2c.play(env, hidden_policy=hidden_policy, hidden_value=hidden_value)
    tstim = torch.stack(tmemory.memory['obs']).cpu().numpy()
    tstim = tstim[tstim[:,0] == 1]
    tLR = np.sum(tstim[:,range(1,3)], axis= 0)
    tas = torch.stack(tmemory.memory['action']).squeeze().cpu().numpy()
    tas = tas[tas>0]
    tas = np.unique(tas)
    dLR[i] = tLR[1] - tLR[0]
    if len(tas) == 1:
        actions[i] = tas
    if i % 100 == 0:
        print(f"iter {i}")

id_valid = actions > 0
actions = actions[id_valid]
dLR = dLR[id_valid]
pR, edges, _ = binned_statistic(dLR, actions == 2, 'mean', bins= np.linspace(-20.5,20.5,42))
plt.figure()
plt.plot(np.linspace(-20,20, 41), pR, 'b.', label='raw data')
#plt.hlines(pR, edges[:-1], edges[1:], colors='g', lw=5,
#           label='binned statistic of data')
plt.xlabel('$\Delta$ Right - Left')
plt.ylabel('p(choose right)')
plt.legend()
plt.show()