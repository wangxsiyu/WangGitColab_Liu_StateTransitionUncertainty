import torch
import torch.nn as nn
from W_LSTM_cell import W_LSTM_cell

class W_LSTM_AC(nn.Module):
    def __init__(self, input_len, hidden_len, action_len, \
                 batch_size = 1):
        super().__init__()
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.action_len = action_len
        self.batch_size = batch_size
        self.lstm = W_LSTM_cell(input_len = input_len, hidden_len = hidden_len)
        self.layer_value = nn.Linear(hidden_len, 1)
        self.layer_action = nn.Sequential(
            nn.Linear(hidden_len, action_len),
            nn.Softmax()
        )
        hidden0 = torch.zeros([batch_size, hidden_len])
        cell0 = torch.zeros([batch_size, hidden_len])
        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)
        self.cell0 = nn.Parameter(cell0, requires_grad=True)

    def forward(self, input, hidden = None, cell = None):
        if hidden is None:
            hidden = self.hidden0
        if cell is None:
            cell = self.cell0
        h_n, c_n = self.lstm(input, (hidden, cell))
        # combined = torch.hstack((h_n, c_n))
        action = self.layer_action(h_n)
        value = self.layer_value(h_n)
        return action, value, hidden, cell

    def forward_unrolled(self, input, hidden = None, cell = None):
        # input: time x batch x obs
        if hidden is None:
            hidden = self.hidden0
        if cell is None:
            cell = self.cell0
        ntime = input.shape[0]
        h = None
        c = None
        a = torch.zeros([ntime, self.batch_size, self.action_len])
        v = torch.zeros([ntime, self.batch_size])
        for i in range(ntime):
            ta,tv,h,c = self.forward(input[i], h, c)
            a[i] = ta
            v[i] = tv
        return a, v
