import torch
import os
import torch
from W_RNN.W_RNN import W_RNN_Head_ActorCritic
from W_RNN.W_Trainer import W_Trainer, W_Worker
import yaml
import argparse
from W_Env.W_Env import W_Env
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
modeltype = "LSTM"# 'vanilla'
render_mode = None
n_maxTrials = 100
env = W_Env("TwoStep_Confidence_2frame", render_mode = render_mode, \
                        n_maxTrials = n_maxTrials)
with open('task.yaml', 'r', encoding="utf-8") as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)
device = torch.device("cpu")
model = W_RNN_Head_ActorCritic(env.observation_space_size() + env.action_space.n + 1,\
    config['a2c']['mem-units'],env.action_space.n,modeltype,device = device)
wk = W_Worker(env, model, capacity = 1000,device = device, mode_sample = "last")

fd = "test_flip"
fdsave = "simu"
tfd = f"data/{fdsave}"
if not os.path.exists(tfd):
    os.mkdir(tfd)
# os.mkdir(f"data/{fd}")
n_seeds = 8
its = (np.ones(n_seeds)*20000).astype('int')
for i in range(n_seeds):
    modeldata = torch.load(f"./{fd}/v_{i+1}/train_iter_{its[i]}.pt")
    wk.model.load_state_dict(modeldata['state_dict'])
    wk.run_worker(100)

    lenm = [len(x.done) for x in wk.memory.memory]
    tid = np.where(lenm == np.max(lenm))[0].astype(int)
    wk.memory.memory = [wk.memory.memory[x] for x in tid]
    buffer = wk.memory.sample(len(lenm))
    (obs,action,reward,timestep,done) = buffer

    action = torch.matmul(action, torch.tensor([1,2,3,4,5], dtype = torch.float)).numpy()
   
    nepisode, ntrial = action.shape
    episodeID = np.linspace(np.ones(ntrial),np.ones(ntrial)*nepisode,nepisode)
    planet = torch.matmul(obs[:,:,:3], torch.tensor([1,2,3], dtype = torch.float)).numpy()
    spaceship = torch.matmul(obs[:,:,3:5], torch.tensor([1,2], dtype = torch.float)).numpy()
    displayreward = obs[:,:,5].numpy()
    question = torch.matmul(obs[:,:,6:9], torch.tensor([1,2,3], dtype = torch.float)).numpy()
    dataset = pd.DataFrame({'episodeID': episodeID.flatten(), 'stepID':timestep.flatten(), \
                            'action': action.flatten(), \
                            'planet':planet.flatten(), 'spaceship':spaceship.flatten(), \
                            'displayreward':displayreward.flatten(), 'question':question.flatten(), \
                            'reward':reward.numpy().flatten()})
    path = f"data/{fdsave}/simu_seed{i}.csv"
    dataset.to_csv(path)