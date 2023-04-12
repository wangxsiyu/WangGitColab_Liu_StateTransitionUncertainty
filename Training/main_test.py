import yaml
import torch
from W_RNN.W_RNN import W_RNN_Head_ActorCritic
from W_RNN.W_Trainer import W_Worker
from W_Env.W_Env import W_Env
from W_Python import W_tools as W
import numpy as np
import re
import os
import sys
import itertools
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
from multiprocessing import Process, Pool, freeze_support, RLock, Lock

def test_2frame(subID):
    tlt = "v" + f"_{subID}"
    with open('task.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    device = torch.device("cpu")
    env = W_Env("TwoStep_Confidence_2frame")
    model = W_RNN_Head_ActorCritic(env.observation_space_size() + env.action_space.n + 1,\
                config['a2c']['mem-units'], env.action_space.n, 'LSTM', device = device)
    wk = W_Worker(None, model, capacity = 100, mode_sample = "all", \
                    device = device)
    model_path = "model/step7_mixed/" + tlt
    wk.loaddict_folder(currentfolder = model_path)
    params = dict(
        ps_high_state = [0.5,0.6,0.7,0.8,0.9,1.0],\
        ps_common_trans = [0.5,0.6,0.7,0.8,0.9,1.0],\
        ps_ambiguity = [0.5,0.6,0.7,0.8,0.9,1.0],\
        )
    p_switch_transition = [True, 0]    
    n_maxTrials = 30
    cb = W.W_counter_balance(params)
    # set save folder
    exp_path = W.W_mkdir('data')
    exp_path = W.W_mkdir(os.path.join(exp_path, tlt))
    for i in tqdm(reversed(range(cb['n'])), position = 1):
        env = W_Env("TwoStep_Confidence_2frame", \
                ps_high_state = cb['ps_high_state'][i], \
                ps_common_trans = cb['ps_common_trans'][i], \
                ps_ambiguity = cb['ps_ambiguity'][i], \
                is_random_common0 = p_switch_transition[0], \
                p_switch_transition = p_switch_transition[1], \
                render_mode = None, \
                n_maxTrials = n_maxTrials)
        wk.env = env
        out_path = os.path.join(exp_path, f"data_{env.get_versionname()}.csv")
        wk.run_worker(100, is_test = True, savename = out_path, showprogress = True)

if __name__ == "__main__":
    n_seeds = 8
    print(n_seeds)
    freeze_support()
    proc = []
    for i in range(1, n_seeds + 1):
        p = Process(target = test_2frame, args = (i, ))
        p.start()
        proc.append(p)

    for p in proc:
        p.join()
