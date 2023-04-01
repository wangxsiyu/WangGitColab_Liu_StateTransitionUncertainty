import numpy as np
import torch 
import os
from W_Env.W_Env import W_Env
from W_A2C import W_A2C
from W_LSTM_AC import W_LSTM_AC
from W_trainer import W_Env_trainer

render_mode = None
n_maxTrials = 100
env = W_Env('TwoStep_simple', render_mode = render_mode, \
                        n_maxTrials = n_maxTrials, is_ITI = False)

a2c = W_LSTM_AC(env._len_observation(), 48, env._len_actions())
trainer = W_A2C(a2c)
simulator = W_Env_trainer(env, trainer)
simulator.train(n_iters = 100000, savename = ".\\models\\mymodel2")








# savename = '.\\models\\v1.pt'
# if os.path.isfile(savename):
#     print(f'load {savename}')

#     a2c.load_state_dict(torch.load(savename))
