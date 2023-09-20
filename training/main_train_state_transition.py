from W_Env.W_Env import W_Env
from W_RNN.W_RNN_ActorCritic import W_RNN_ActorCritic
from W_Trainer.W_Trainer import W_Trainer
from W_Python.W import W
import torch
import numpy as np
import yaml
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    seed = 0
    device = "cpu" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('setup.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    with open('param_task.yaml', 'r', encoding="utf-8") as fin:
        config_task = yaml.load(fin, Loader=yaml.FullLoader)
    key = "ConsTrans"
    env = W_Env("TwoStep", \
            param_task = config_task[key], \
            render_mode = None, \
            n_maxTrials = config_task['n_maxtrials'])
    ninput = env.get_n_obs()
    model = W_RNN_ActorCritic(ninput, config['model']['mem-units'], \
                              gatetype = "LSTM", \
                              actionlayer = env.get_n_actions(), device=device)
    trainer = W_Trainer(env, model, param_loss = config['param_loss'], param_optim = config['param_optim'], \
                        param_logger = config['param_logger'], param_buffer = config['param_buffer'], \
                            gradientclipping = config['trainer']['max-grad-norm'], \
                            save_path = config['trainer']['save_path'], \
                            device = device, \
                            seed = seed)

    trainer.train(max_episodes= config['trainer']['max_episodes'], batch_size=1, train_mode="RL", is_online=True)
    # # set save folder
    # savefolder = W.W_mkdir(config['save-path'])
    # exp_path = W.W_mkdir(os.path.join(savefolder, key['ver']))
    # tlt = "v" + f"_{seed_idx}"
    # exp_path = W.W_mkdir(os.path.join(exp_path, tlt))
    # out_path = os.path.join(exp_path, f"train_iter")
    # with open(out_path + ".yaml", 'w') as fout:
    #     yaml.dump(config, fout)
    # wk.logger.set(save_path= out_path, save_interval= config['save-interval'])
    # wk.loaddict_main(lastver, currentfolder = exp_path, isresume = config['resume'])
    # strver = 'ConsTrans' if veri == 1 else 'RandTrans'
    # wk.train(key['nstep'], batch_size = 1, tqdmpos = seed_idx+ veri*4, tqdmstr= strver + f"_{seed_idx}")
    # if verbose:
    #     print(f'Last saved version: {wk.logger.last_saved_version}')
    # return wk.logger.last_saved_version



        
    
