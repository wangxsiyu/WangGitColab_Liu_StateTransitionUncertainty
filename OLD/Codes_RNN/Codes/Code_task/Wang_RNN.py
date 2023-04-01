import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


class MemoryEpisode():
    def __init__(self) -> None:
        self.memory = defaultdict(list)
    def get_memory(self):
        return self.memory
    def append(self, obs, reward = None, action = None):
        self.memory['obs'].append(obs)
        if reward is not None:
            self.memory['reward'].append(reward)
        if action is not None:
            self.memory['action'].append(action)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activationfunc, device = None, batch_size = 1) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.layer_encode = nn.Linear(input_size + hidden_size, hidden_size, device = self.device)
        self.layer_action = nn.Linear(input_size + hidden_size, output_size, device = self.device)
        self.activation_encode = getattr(nn, activationfunc)()
        self.hidden0 = nn.Parameter(self.init_hidden(batch_size))
    
    def forward(self, input, hidden = None):
        if hidden is None:
            hidden = self.hidden0
        input = input.to(self.device)
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
        noise = self.get_noise()
        hidden = hidden + noise
        combined = torch.cat((input, hidden), 1)
        hidden = self.activation_encode(self.layer_encode(combined))
        action = self.layer_action(combined)
        return action, hidden
        
    def init_hidden(self, batch_size = 1):
        return torch.zeros([batch_size, self.hidden_size], requires_grad=True).to(self.device)

    
    def get_noise(self, batch_size = 1):
        """get Gaussian noise"""
        noise = torch.normal(mean=0, std=1, size=[batch_size, 
                                                  self.hidden_size])
        return np.sqrt(2/1) * 1 * noise