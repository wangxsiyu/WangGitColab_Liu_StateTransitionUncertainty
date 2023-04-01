import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import torch.optim as optim
from itertools import count
import math
import numpy as np
import pickle

device = "cpu"# torch.device("cuda") if torch.cuda.is_available() else "cpu"

def save_obj(obj, name ):
    with open('temp/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def push(self, episode):
        """Save a transition"""
        self.memory.append(episode)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_network(nn.Module):
    def __init__(self, input_dim = 9, n_action = 9, lstm_hidden_dim = 10, lstm_n_layer = 1, batch_size = 1):
        super(DQN_network, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_n_layer = lstm_n_layer
        self.lstm_layer = nn.LSTM(input_dim, lstm_hidden_dim, lstm_n_layer, batch_first = True).to(device)
        self.out_layer = nn.Linear(lstm_hidden_dim, n_action).to(device)

    def initialize_hidden_units(self, batch_size):
        hidden_state = torch.zeros(self.lstm_n_layer, batch_size, self.lstm_hidden_dim).float().to(device)
        cell_state = torch.zeros(self.lstm_n_layer, batch_size, self.lstm_hidden_dim).float().to(device)
        return hidden_state.to(device), cell_state.to(device)

    def forward(self, x, hidden_state, cell_state, batch_size = 1):
        if len(x.shape) < 3:
            nseq = 1
            x = x.view(batch_size, nseq, len(x))
        hidden = (hidden_state, cell_state)
        [out, (hidden_state, cell_state)] = self.lstm_layer(x, hidden)
        return self.out_layer(out), hidden_state, cell_state

class DQN():
    def __init__(self, env, in_num = 9, batch_size = 1):
        self.env = env
        self.in_num = in_num
        self.dqn_policy = DQN_network(in_num, env.num_actions())
        self.dqn_target = DQN_network(in_num, env.num_actions())
        self.dqn_policy.to(device)
        self.dqn_target.to(device)
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_target.eval()
        
        self.loss_stat = []
        self.reward_stat = []
        self.memory = ReplayMemory(3000)
        self.batch_size = batch_size
        self.target_update = 5000
        self.policy_update = 1
        self.performance_save_interval = 5000
        self.gamma = torch.tensor(1, device = device)
        self.reset_eps()
        self.optimizer = optim.Adam(self.dqn_policy.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.n_episode = 0
        self.n_training = 0
        self.fill_memory(1)
        self.eval_hidden_state = None
        self.eval_cell_state = None

    def reset_eps(self, num_episodes = 10000, eps_start = 1.00):
        self.eps_end = 0.05
        self.eps_start = eps_start
        self.eps_decay = num_episodes

    def fill_memory(self, fillsize = None):
        if fillsize is None:
            fillsize = self.memory.capacity
        for i in range(fillsize):
            self.simulate(randomchoice = True)
        print('Populated with %d episodes'%self.memory.capacity)

    def predict(self, prev_state, hidden_state = None, cell_state = None, deterministic = False, is_eval = False):
        if is_eval:
            if self.eval_hidden_state is None:
                self.eval_hidden_state, self.eval_cell_state = self.dqn_policy.initialize_hidden_units(batch_size=1)
            hidden_state = self.eval_hidden_state
            cell_state = self.eval_cell_state
        torch_x = torch.from_numpy(prev_state).float().to(device)
        model_out, hidden_state, cell_state = self.dqn_policy.forward(torch_x, hidden_state=hidden_state, cell_state=cell_state, batch_size = self.batch_size)
        eps_threshold = self.eps_start + (self.eps_end - self.eps_start) * self.n_episode/self.eps_decay
        if np.random.rand(1) > eps_threshold or deterministic:
            action = int(torch.argmax(model_out))
        else:
            action = np.random.randint(0, self.env.num_actions())
        if is_eval:
            self.eval_cell_state = cell_state
            self.eval_hidden_state = hidden_state
            return action
        else:
            return action, hidden_state, cell_state

    def simulate(self, randomchoice = False):
        tot_reward = 0
        env = self.env
        prev_state = env.reset()
        local_memory = []
        local_action = []
        local_observation = []
        local_reward = []
        if not randomchoice:
            (hidden_state, cell_state) = self.dqn_policy.initialize_hidden_units(batch_size = 1)
        done = False
        while not done:
            # Select and perform an action
            if randomchoice:
                action = np.random.randint(0, env.num_actions())
            else:
                action, hidden_state, cell_state = self.predict(prev_state,hidden_state, cell_state)
            next_state, reward, done, _ = env.step(action)
            local_observation.append(prev_state)
            local_reward.append(reward)
            local_action.append(action)
            # Move to the next state
            prev_state = next_state
            tot_reward += reward
            # Perform one step of the optimization (on the policy network)
            if done:
                break
        local_memory = (local_observation, local_action, local_reward)
        self.memory.push(local_memory)
        return tot_reward

    def train(self, num_episodes = 1, n_log = 10, eps_start = 1.00):
        self.n_episode = 0
        self.reset_eps(num_episodes=num_episodes, eps_start = eps_start)
        for i_episode in range(num_episodes):
            self.n_episode = i_episode + 1
            if i_episode % n_log == 0:
                print(f"@episode = {i_episode}")
            tot_reward = self.simulate()
            self.reward_stat.append(tot_reward)
            if self.n_episode % self.policy_update == 0:
                self.step_train()

            if self.n_episode % self.performance_save_interval == 0:
                perf = {}
                perf['loss'] = self.loss_stat
                perf['total_reward'] = self.reward_stat   
                save_obj(name = "LSTM_perf", obj = perf)

                torch.save(self.dqn_policy.state_dict(), 'MC_trained_weights.torch')
                print('weights saved')
        print('Complete')
        # return self.logs

    def step_train(self):
        hidden_batch, cell_batch = self.dqn_policy.initialize_hidden_units(batch_size=self.batch_size)
        batch = self.memory.sample(batch_size = self.batch_size)
        
        observation = []
        acts = []
        rewards = []            
        for b in batch:
            observation.append(b[0])
            acts.append(b[1])
            rewards.append(b[2])
            
        observation = np.array(observation)
        acts = np.array(acts)
        rewards = np.array(rewards)
        
        torch_observation = torch.from_numpy(observation).float().to(device)
        torch_acts = torch.from_numpy(acts).long().to(device)
        torch_rewards = torch.from_numpy(rewards).float().to(device)       
        
        Q_t, _, _= self.dqn_target.forward(torch_observation,hidden_state=hidden_batch,cell_state=cell_batch, batch_size=self.batch_size)
        Q_t_max,__ = Q_t.detach().max(dim=2)
        Q_end = torch.zeros((self.batch_size,1)).to(device)
        Q_max = torch.cat((Q_t_max[:,:-1], Q_end), dim = 1)

        target_values = torch_rewards + (self.gamma * Q_max)
            
        Q_p, _, _= self.dqn_policy.forward(torch_observation,hidden_state=hidden_batch,cell_state=cell_batch, batch_size=self.batch_size)   
        Q_a = Q_p.gather(dim=2,index=torch_acts.unsqueeze(2)).squeeze(2)
            
        # Compute Huber loss
        loss = self.criterion(Q_a,target_values)
        self.loss_stat.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.dqn_policy.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.n_training % self.target_update == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())