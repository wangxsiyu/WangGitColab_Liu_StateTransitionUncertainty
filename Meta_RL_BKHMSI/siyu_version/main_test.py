import os
import yaml
import pickle
import argparse
import datetime
import scipy.signal

import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer: 
    def __init__(self, config):
        self.device = 'cpu'
        self.seed = config["seed"]

        T.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        T.random.manual_seed(config["seed"])


        self.val_coeff = config["a2c"]["value-loss-weight"]
        self.entropy_coeff = config["a2c"]["entropy-weight"]
        self.max_grad_norm = config["a2c"]["max-grad-norm"]
        self.switch_p = config["task"]["swtich-prob"]
        self.start_episode = 0

        self.writer = SummaryWriter(log_dir=os.path.join("logs", config["run-title"]))
        self.save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")

        if config["resume"]:
            print("> Loading Checkpoint")
            self.start_episode = config["start-episode"]
            self.agent.load_state_dict(T.load(self.save_path.format(epi=self.start_episode) + ".pt")["state_dict"])




    def train(self, max_episodes, gamma, save_interval):


        for episode in progress:


            self.optim.zero_grad()
            loss = self.a2c_loss(buffer, gamma) 
            loss.backward()
            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optim.step()


            avg_reward_10 = total_rewards[max(0, episode-10):(episode+1)].mean()
            avg_reward_100 = total_rewards[max(0, episode-100):(episode+1)].mean()
            self.writer.add_scalar("perf/reward_t", reward, episode)
            self.writer.add_scalar("perf/avg_reward_10", avg_reward_10, episode)
            self.writer.add_scalar("perf/avg_reward_100", avg_reward_100, episode)
            self.writer.add_scalar("losses/total_loss", loss.item(), episode)
            if self.max_grad_norm > 0:
                self.writer.add_scalar("losses/grad_norm", grad_norm, episode)

           


    def test(self, num_episodes):
        progress = tqdm(range(num_episodes))
        self.env.reset_transition_count()
        self.agent.eval()
        total_rewards = np.zeros(num_episodes)
        for episode in progress:
            reward, _ = self.run_episode(episode)
            total_rewards[episode] = reward
            avg_reward = total_rewards[max(0, episode-10):(episode+1)].mean()            
            progress.set_description(f"Episode {episode}/{num_episodes} | Reward: {reward} | Last 10: {avg_reward:.4f}")

        self.env.plot(self.save_path.format(epi=self.seed))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="two_step.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    n_seeds = 8
    base_seed = config["seed"]
    base_run_title = config["run-title"]
    for seed_idx in range(1, n_seeds + 1):
        config["run-title"] = base_run_title + f"_{seed_idx}"
        config["seed"] = base_seed * seed_idx
        
        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.mkdir(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        print(f"> Running {config['run-title']}")
        trainer = Trainer(config)
        if config["train"]:
            trainer.train(config["task"]["train-episodes"], config["a2c"]["gamma"], config["save-interval"])
        if config["test"]:
            trainer.test(config["task"]["test-episodes"])