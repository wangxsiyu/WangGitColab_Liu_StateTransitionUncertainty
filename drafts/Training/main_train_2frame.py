import yaml
import torch
from W_RNN.W_RNN import W_RNN_Head_ActorCritic
from W_RNN.W_Trainer import W_Trainer
from W_Env.W_Env import W_Env
from W_Python import W_tools as W
import numpy as np
import re
import os
# from old_task import task_TwoStep_Confidence_2frame
import sys
np.set_printoptions(threshold=sys.maxsize)

def train_2frame(seed_idx, key = None, lastver = None, verbose = False):
    device = torch.device("cpu")
    with open('task.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    if key is not None:
        env = W_Env("TwoStep_Confidence_2frame", \
                ps_high_state = key['ps_high_state'], \
                ps_common_trans = key['ps_common_trans'], \
                ps_ambiguity = key['ps_ambiguity'], \
                is_random_common0 = key['is_random_common0'], \
                p_switch_transition = key['p_switch_transition'], \
                render_mode = None, \
                n_maxTrials = config['task']['n_maxtrials'])
    else:
        env = W_Env("TwoStep_Confidence_2frame", \
                render_mode = None, \
                n_maxTrials = config['task']['n_maxtrials'])
        key = {'nstep': 10000, 'ver': 'test'}

    seed = 1995 * (seed_idx + 1)
    model = W_RNN_Head_ActorCritic(env.observation_space_size() + env.action_space.n + 1,\
                config['a2c']['mem-units'], env.action_space.n, 'LSTM', device = device)
    loss = dict(name = 'A2C', params = dict(gamma = config['a2c']['gamma'], \
                                            coef_valueloss = config['a2c']['value-loss-weight'], \
                                            coef_entropyloss = config['a2c']['entropy-loss-weight']))
    optim = dict(name = 'RMSprop', lr  = config['a2c']['lr'])
    wk = W_Trainer(env, model, loss, optim, capacity = 1000, mode_sample = "last", \
                device = device, gradientclipping=config['a2c']['max-grad-norm'], \
                seed = seed)

    # set save folder
    savefolder = W.W_mkdir(config['save-path'])
    exp_path = W.W_mkdir(os.path.join(savefolder, key['ver']))
    tlt = "v" + f"_{seed_idx}"
    exp_path = W.W_mkdir(os.path.join(exp_path, tlt))
    out_path = os.path.join(exp_path, f"train_iter")
    with open(out_path + ".yaml", 'w') as fout:
        yaml.dump(config, fout)
    wk.logger.set(save_path= out_path, save_interval= config['save-interval'])
    wk.loaddict_main(lastver, currentfolder = exp_path, isresume = config['resume'])
    wk.train(key['nstep'], batch_size = 1, tqdmpos = seed_idx)
    if verbose:
        print(f'Last saved version: {wk.logger.last_saved_version}')
    return wk.logger.last_saved_version

if __name__ == "__main__":
    train_2frame(0, verbose= True)
