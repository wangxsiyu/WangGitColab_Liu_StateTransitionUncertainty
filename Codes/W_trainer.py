import torch
import numpy as np
import pickle

class W_Env_trainer():
    def __init__(self, env, trainer, *arg, **kwarg):
        self.trainer = trainer
        self.env = env

    def record_behavior(self):
        env = self.env
        model = self.trainer.model
        obs = env.reset()
        obs = torch.Tensor(obs)
        done = False
        tot_reward = 0
        h = None
        c = None
        observations = [obs]
        actions = []
        rewards = []
        info = []
        while not done:
            action_prob, value, h, c = model.forward(obs, h, c)
            action = self.choose_action(action_prob.detach())
            next_obs, reward, done, _, tinfo = env.step(action.numpy()[0])
            tot_reward += reward
            obs = torch.from_numpy(next_obs).to(torch.float32)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            info.append(tinfo)
        return observations, actions, rewards, tot_reward, info
    
    def choose_action(self, action_prob, deterministic = False):
        # if deterministic:
        #     action = torch.argmax(action_prob)
        # else:
        dist = torch.distributions.Categorical(probs = action_prob)
        action = dist.sample()
        return action
    
    def train(self, n_iters, savename):
        reward_history = []
        loss_history = []
        for i in range(n_iters):
            obs, actions, rewards, treward_history, info = self.record_behavior()
            reward_history.append(treward_history)
            obs = torch.stack(obs)
            actions = torch.stack(actions, dim = 0)
            rewards = torch.tensor(rewards).to(torch.float32).unsqueeze(1)
            if info is not None:    
                info = torch.tensor(info)
            tloss = self.trainer.train_episodes(obs, actions, rewards, info)
            loss_history.append(tloss)
            if i % 10 == 0:
                ob_idx = torch.matmul(obs[0:-1,0:3], torch.tensor([1,2,3.])).numpy()
                ob_idx = ob_idx.astype('int')
                tr = rewards.numpy()
                ratio1 =np.sum(tr[ob_idx == 2] == 1)/(np.sum(tr[ob_idx == 2] >= 0))
                ratio2 =np.sum(tr[ob_idx == 3] == 1)/(np.sum(tr[ob_idx == 3] >= 0))
                better = (ratio2 > ratio1).astype(int) + 1
                ta = actions.squeeze().numpy()
                p_better = np.sum(ta[ob_idx == 1] ==  better)/np.sum(ta[ob_idx == 1] != 0)
                print(f"iter {i}: r = {reward_history[i]}, loss = {loss_history[i]}, err1 = {np.mean(rewards[ob_idx == 1].numpy() == -1)}, err2 = {np.mean(rewards[ob_idx != 1].numpy() == -1)}")
                print(f"better = {better}, p(better) = {p_better}")
                 
                with open(f"{savename}.pkl", "wb") as f: 
                    pickle.dump([reward_history, loss_history], f)
            if i % 200 == 0:
                self.save(f"{savename}_iter{i}.pt")

    def save(self, savename):
        torch.save(self.trainer.model.state_dict(), f'{savename}')
