import yaml
import torch
from W_RNN.W_RNN import W_RNN_Head_ActorCritic
from W_RNN.W_Trainer import W_Trainer
from W_RNN.W_Logger import W_Logger 
from W_Env.W_Env import W_Env
from W_Python import W_tools as W
import numpy as np
import re
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
from multiprocessing import Process, Pool, freeze_support, RLock, Lock

class W_logger1(W_Logger):
    def __init__(self):
        super().__init__()
    def _init(self):
        self.info['pamb'] = np.zeros_like(self.info['rewards'])
    def _update(self, infogame):
        self.info['pamb'][self.episode] = infogame[0]['params']['p_ambiguity']
    def _getdescription(self, str):
        episode = self.episode
        pa = self.info['pamb'][max(0, episode-self.smooth_interval):(episode+1)]
        r = self.info['rewards'][max(0, episode-self.smooth_interval):(episode+1)]
        if episode > 100:
            av = [r[pa < 0.1].mean(),r[pa ==0.5].mean(),r[pa >= 0.9].mean()]
            str = f"R0 = {av[0]}, R50 = {av[1]}, R100 = {av[2]}"
        else:
            str = str
        return str

def train_2frame(seed_idx, veri, key = None, lastver = None, verbose = False):
    device = torch.device("cpu")
    with open('task.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    if key is None:
        raise Exception('define the key')
    env = W_Env("TwoStep_Ambiguity_1frame", \
            p_switch_reward = key['p_switch_reward'], \
            ps_high_state = key['ps_high_state'], \
            ps_common_trans = key['ps_common_trans'], \
            ps_ambiguity = [0,0.5,1], \
            is_random_common0 = key['is_random_common0'], \
            p_switch_transition = key['p_switch_transition'], \
            render_mode = None, \
            n_maxTrials = config['task']['n_maxtrials'])


    seed = 1995 * (seed_idx + 1)
    model = W_RNN_Head_ActorCritic(env.observation_space_size() + env.action_space.n + 1,\
                config['a2c']['mem-units'], env.action_space.n, 'LSTM', device = device)
    loss = dict(name = 'A2C', params = dict(gamma = config['a2c']['gamma'], \
                                            coef_valueloss = config['a2c']['value-loss-weight'], \
                                            coef_entropyloss = config['a2c']['entropy-loss-weight']))
    optim = dict(name = 'RMSprop', lr  = config['a2c']['lr'])
    logger = W_logger1()
    wk = W_Trainer(env, model, loss, optim, logger = logger, capacity = 1000, mode_sample = "last", \
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
    strver = 'Transition' if veri == 2 else 'Ambiguity'
    wk.train(key['nstep'], batch_size = 1, tqdmpos = seed_idx+ veri*4, tqdmstr= strver + f"_{seed_idx}")
    if verbose:
        print(f'Last saved version: {wk.logger.last_saved_version}')
    return wk.logger.last_saved_version

def trainver(seed_idx, config, veri):
    if veri == 1:
        keys = config['Ambiguity']
    elif veri == 2:
        keys = config['Transition']
    train_2frame(seed_idx,  veri, key = keys, verbose= True)



if __name__ == "__main__":
    with open('param.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    trainver(1, config, 2)
    # freeze_support()
    # proc = []
    # for seed_idx in range(1, 5):
    #     for veri in range(2, 3):
    #         p = Process(target = trainver, args = (seed_idx, config, veri))
    #         p.start()
    #         proc.append(p)

    # for p in proc:
    #     p.join()
