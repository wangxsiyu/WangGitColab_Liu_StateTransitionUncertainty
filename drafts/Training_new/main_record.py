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

def test_2frame(subID, tvar, cb, loopi):
    tlt = tvar + "_v" + f"_{subID}"
    with open('task.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    device = torch.device("cpu")
    env = W_Env("TwoStep")
    model = W_RNN_Head_ActorCritic(env.env.get_n_obs(),\
                config['a2c']['mem-units'], env.env.get_n_actions(), 'LSTM', device = device)
    wk = W_Worker(None, model, capacity = 100, mode_sample = "all", \
                    device = device)
    model_path = "model/" + tvar + "/" + f"v_{subID}"
    wk.loaddict_folder(currentfolder = model_path)
    # params = dict(
    #     ps_high_state = [0.75], \
    #     ps_common_trans = [0.8],\
    #     ps_ambiguity =  [0,0.4],\
    #     )
    # cb2 = W.W_counter_balance(params)
    # params = dict(
    #     ps_high_state = [0.75,1.0], \
    #     ps_common_trans = [0.8],\
    #     ps_ambiguity = [0],\
    #     )
    # cb3 = W.W_counter_balance(params)
    # cb = {key:cb1[key]+cb2[key]+cb3[key] for key in cb1}
    # p_switch_transition = [True, 0]    
    n_maxTrials_reset = 300
    # set save folder
    exp_path = W.W_mkdir('data')
    exp_path = W.W_mkdir(os.path.join(exp_path, tlt))
    # for i in tqdm(reversed(range(cb['n'])), position = 0):
    env = W_Env("TwoStep", \
            ps_high_state = cb['ps_high_state'][loopi], \
            ps_common_trans = cb['ps_common_trans'][loopi], \
            ps_ambiguity = cb['ps_ambiguity'][loopi], \
            is_random_common0 = True, \
            p_switch_reward = 0.02, \
            p_switch_transition = 0, \
            render_mode = None, \
            n_maxTrials_reset = n_maxTrials_reset, \
            block_n_maxTrials = 300)
    wk.env = env
    out_path = os.path.join(exp_path, f"data_{env.env.get_savename()}.csv")
    wk.env.record(wk.model, n_episode = 100, savename = out_path, showprogress = True)

if __name__ == "__main__":
    # with open('param.yaml', 'r', encoding="utf-8") as fin:
    #     config = yaml.load(fin, Loader=yaml.FullLoader)
    # test_2frame(1, 'Ambiguity2')
    freeze_support()
    params = dict(
        ps_high_state = [0.8], \
        ps_common_trans = [0.8],\
        ps_ambiguity = [0, 0.1, 0.2, 0.5, 1.0],\
        )
    cb1 = W.W_counter_balance(params)
    params = dict(
        ps_high_state = [0.8], \
        ps_common_trans = [0.6,0.7,0.8,0.9,1.0],\
        ps_ambiguity = [0],\
        )
    cb2 = W.W_counter_balance(params)
    cb = cb1 # {key:cb1[key]+cb2[key] for key in cb1}
    proc = []
    for seed_idx in range(1, 5):
        for veri in range(3, 4):
            for loopi in range(cb['n']):
                if veri == 1:
                    tvar = 'Ambiguity'
                elif veri == 2:
                    tvar = 'Transition'
                elif veri == 3:
                    tvar = 'MBMF'
                # keys = config[tvar]
                # test_2frame(seed_idx, tvar, cb, loopi)
                p = Process(target = test_2frame, args = (seed_idx, tvar, cb, loopi))
                p.start()
                proc.append(p)

    for p in proc:
        p.join()
