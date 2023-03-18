import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Wang_RNN import RNN, MemoryEpisode

class PolicyNet(nn.Module):
    def __init__(self, observation_size, hidden_size, action_size, activationfunc, device) -> None:
        super().__init__()
        self.policynet = RNN(input_size=observation_size, hidden_size=hidden_size, output_size=action_size,\
            activationfunc=activationfunc, device=device)
    def forward(self, observation, hidden):
        actionvalue, hidden = self.policynet.forward(observation, hidden)
        action_dist = nn.functional.softmax(actionvalue, dim = 1)
        return action_dist, hidden

class ValueNet(nn.Module):
    def __init__(self, hidden_policy_size, hidden_value_size, action_size, activationfunc, device) -> None:
        super().__init__()
        self.action_size = action_size
        self.valuenet = RNN(input_size=hidden_policy_size + action_size, hidden_size=hidden_value_size, output_size=1,\
            activationfunc=activationfunc, device=device)
    def forward(self, policynet, action, hidden):
        onehot_action = nn.functional.one_hot(action, self.action_size)
        combined = torch.cat((policynet, onehot_action), 1)
        value, hidden = self.valuenet.forward(combined, hidden)
        return value, hidden

class Actor_Critic(nn.Module):
    def __init__(self, observation_size, action_size, hidden_value_size, hidden_policy_size, activationfunc = "ReLU", device = "cuda") -> None:
        super().__init__()
        self.device = device
        self.is_playmode = False
        self.valuenet = ValueNet(hidden_policy_size, hidden_value_size, action_size, activationfunc, device)
        self.policynet = PolicyNet(observation_size, hidden_policy_size, action_size, activationfunc, device)
    def forward(self, observation, hidden_policy, hidden_value, action = None, deterministic = False):
        action_dist, hidden_policy = self.policynet.forward(observation, hidden_policy)
        if action is None:
            if deterministic:
                action = torch.argmax(action_dist)
            else:
                dist = torch.distributions.Categorical(probs = action_dist)
                action = dist.sample()
        value, hidden_value = self.valuenet.forward(hidden_policy, action, hidden_value)
        return action, value, hidden_policy, hidden_value, action_dist
    def playmode(self):
        self.is_playmode = True
        self.hidden_policy = None
        self.hidden_value = None
    def justaction(self, obs):
        obs = torch.from_numpy(obs)
        obs = obs.flatten()
        obs = obs.to(torch.float)
        action_dist, self.hidden_policy = self.policynet.forward(obs, self.hidden_policy)
        action = torch.argmax(action_dist)
        action = action.unsqueeze(0)
        value, self.hidden_value = self.valuenet.forward(self.hidden_policy, action, self.hidden_value)
        return action
    def play(self, env, hidden_policy, hidden_value):
        obs = env.reset()
        obs = torch.Tensor(obs.flatten()).to(self.device)
        done = False
        tot_reward = 0
        memory = MemoryEpisode()
        memory.append(obs)
        while not done:
            action, value, hidden_policy, hidden_value, action_dist = self.forward(obs, hidden_policy, hidden_value)
            next_obs, reward, done, _, _ = env.step(np.array(action[0].detach().cpu()))
            next_obs = torch.Tensor(next_obs.flatten()).to(self.device)
            tot_reward += reward
            memory.append(next_obs, reward, action.detach())
            obs = next_obs
        return memory, tot_reward
    def save(self, savename):
        torch.save(self.state_dict(), f'{savename}')
        
class train_Actor_Critic():
    def __init__(self, model, lr, weight_decay, gamma) -> None:
        self.optimizer_valuenet = optim.AdamW(model.valuenet.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_policynet = optim.AdamW(model.policynet.parameters(), lr=lr, weight_decay=weight_decay)
        self.gamma = gamma
        self.model = model
    def update_episode(self, obs, actions, rewards):
        nstep = len(actions)
        # rerun model
        hidden_policy = None
        hidden_value = None
        value = torch.zeros(nstep)
        action_prob = torch.zeros(nstep)
        for i in range(nstep):
            taction = actions[i]
            _, tvalue, hidden_policy, hidden_value, taction_dist = self.model.forward(obs[i], hidden_policy, hidden_value, taction)
            value[i] = tvalue
            action_prob[i] = torch.log(taction_dist[0][taction])

        returns = np.zeros((1, nstep))
        treturn = 0.0
        for i in range(nstep):
            i_step = nstep - i - 1
            treturn = rewards[i_step] + self.gamma * treturn
            returns[0][i_step] = treturn
        advantage = torch.Tensor(returns) - value

        self.optimizer_valuenet.zero_grad()
        self.optimizer_policynet.zero_grad()

        nn.utils.clip_grad_norm_(self.model.valuenet.parameters(), max_norm=1)
        nn.utils.clip_grad_norm_(self.model.policynet.parameters(), max_norm=1)   # arbitrary value 

        critic_loss = advantage.pow(2).mean()
        actor_loss = (-action_prob*advantage.detach()).mean()
        
        loss = critic_loss + actor_loss
        loss.backward()
        self.optimizer_valuenet.step()
        self.optimizer_policynet.step()
