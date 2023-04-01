import torch
import torch.nn as nn
import torch.nn.functional as F

from W_NN_initialize import W_NN_initialize as W_init

class AC_LSTM(nn.Module):
    def __init__(self, input_len, hidden_len, action_len):
        # intialize actor and critic weights
        
    def forward(self, data):
        state, p_action, p_reward, timestep, hidden_state = data 
        p_input = torch.cat((state, p_action, p_reward, timestep), dim=-1)

    
        h_t, hidden_state = self.working_memory(p_input.unsqueeze(1), hidden_state)

        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value_estimate = self.critic(h_t)

        return action_dist, value_estimate, hidden_state
