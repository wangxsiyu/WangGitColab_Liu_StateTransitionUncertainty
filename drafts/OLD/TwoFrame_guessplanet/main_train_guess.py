import torch
from W_RNN.W_RNN import W_RNN_Head_ActorCritic
# from W_RNN.W_Trainer import W_Trainer
import torch.nn as nn
import yaml
import argparse
from W_Env.W_Env import W_Env
from task_TwoStep_Confidence_2frame_guessplanet import task_TwoStep_Confidence_2frame_guessplanet
from task_safebet import task_safebet
import numpy as np
import re
# from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Pool, freeze_support, RLock, Lock
# from multiprocessing.pool import ThreadPool
import tqdm
from trainer import trainer


class mymodel(W_RNN_Head_ActorCritic):
    def __init__(self, input_len, hidden_len, action_len, gatetype="vanilla", inittype=None, device=None, *arg, **kwarg):
        super().__init__(input_len, hidden_len, action_len, gatetype, inittype, device, *arg, **kwarg)

    def _forward(self, obs, hidden_state = None):
        if hidden_state is None:
            batch_size = obs.shape[0]
            hidden_state = self.get_h0_c0(batch_size)   
        obs = obs.permute((1,0,2))     
        h, hidden_state = self.RNN(obs, hidden_state)
        policy_vector = self.actor(h)
        value = self.critic(h)
        policy_safebet = self.conf(h)
        return policy_vector, value, hidden_state, policy_safebet


class W_Trainer_WV(trainer):
    def __init__(self, env, model, param_loss, param_optim, logger=None, device=None, gradientclipping=None, seed=None, position_tqdm=0, *arg, **kwarg):
        super().__init__(env, model, param_loss, param_optim, logger, device, gradientclipping, seed, position_tqdm, *arg, **kwarg)
        self.last_checkpoint = 0

    # def _train_special(self, episode, totalreward, totalrewardsmooth):
    #     if episode > self.last_checkpoint + 100 and \
    #         totalreward[episode-1] > 0 and \
    #         totalrewardsmooth[episode-1] > totalrewardsmooth[self.last_checkpoint]:
    #         self.env.n_maxT += 100
    #         self.last_checkpoint = episode - 50

def mytrain(seed_idx):
    device = "cpu"
    device = torch.device(device)
    print(f"enabling {device}")
    position_tqdm = seed_idx
    with open('task.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    render_mode = None
    n_maxTrials = 100
    env = task_safebet(is_fixed = [2, 2], is_flip_trans = True, is_flip = False, render_mode = render_mode, \
                            n_maxTrials = n_maxTrials, is_guess = True)
    
    # env = W_Env("TwoStep_Confidence_2frame", is_fixed = [1, 1], is_flip_trans = True, is_flip = False, render_mode = render_mode, \
    #                         n_maxTrials = n_maxTrials)
    tseed = 515 * seed_idx
    tlt = "v" + f"_{seed_idx}"
    
    import os
    savefolder = "guess"
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    exp_path = os.path.join(savefolder, tlt)
    if not os.path.isdir(exp_path): 
        os.mkdir(exp_path)
    # print("running" + f"_{seed_idx}")
    noise_scale = 0 #0.05 if np.random.rand() < 0.5 else 0
    # print(f"noise_scale = {noise_scale}")
    config['noise_scale'] = noise_scale    
    out_path = os.path.join(exp_path, f"train_iter")
    with open(out_path + ".yaml", 'w') as fout:
        yaml.dump(config, fout)

    model = mymodel(env.observation_space_size() + env.action_space.n + 1,\
                            config['a2c']['mem-units'],env.action_space.n,'LSTM',noise_scale = noise_scale, device = device)
    
    model.conf = nn.Linear(model.RNN.hidden_size, 6)
    file_trained_list = os.listdir(os.path.dirname(
        out_path))
    
    loss = dict(name = 'A2C', params = dict(gamma = config['a2c']['gamma'], \
                                            coef_valueloss = config['a2c']['value-loss-weight'], \
                                            coef_entropyloss = config['a2c']['entropy-loss-weight']))
    
    basenames = [os.path.basename(x) for x in file_trained_list]
    res = [re.search("train_iter_(.*).pt", x) for x in basenames]
    res = [x for x in res if x is not None]
    if not len(res) == 0:
        res = [x.group(1) for x in res]
        maxiter = np.max([int(x) for x in res])
        loadname = f"{out_path}_{maxiter}.pt"
        print(f"load model data: {loadname}")
        modeldata = torch.load(loadname)
        model.load_state_dict(modeldata['state_dict'])
        last_episode = modeldata['last_episode'] + 1
        print(f"start episode: {last_episode}")
    else:
        last_episode = 0
    
    for param in model.RNN.parameters():
        param.requires_grad = False
    for param in model.critic.parameters():
        param.requires_grad = False
    for param in model.actor.parameters():
        param.requires_grad = False
    model.h0.requires_grad = False
    model.c0.requires_grad = False
    
    optim = dict(name = 'RMSprop', lr  = config['a2c']['lr'])
    wk = W_Trainer_WV(env, model, loss, optim, capacity = 1000, mode_sample = "last", \
                   device = device, gradientclipping=config['a2c']['max-grad-norm'], seed = tseed, position_tqdm = position_tqdm)
    
    wk.train(150000, batch_size = 1, save_path= out_path, save_interval= 500, last_episode=last_episode)





if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="parameters")
    # parser.add_argument('-c','--config', type = str, default = 'task.yaml')

    # args = parser.parse_args()

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    freeze_support()

    n_seeds = 7

    # with ProcessPoolExecutor(max_workers=n_seeds) as executor:
    #     executor.map(mytrain, range(n_seeds))

    # mytrain(3)
    # mytrain(0)
    # p = Pool(n_seeds)
    # for seed_idx in range(n_seeds):
    #     p.apply_async(mytrain, seed_idx)
    # p.close()
    # p.join()

    proc = []
    for seed_idx in range(1, n_seeds + 1):
        p = Process(target = mytrain, args = (seed_idx, ))
        p.start()
        proc.append(p)

    for p in proc:
        p.join()


