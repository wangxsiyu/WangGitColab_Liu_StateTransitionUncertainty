import torch
from torch import nn

class W_LSTM_cell(nn.Module):
    def __init__(self, input_len, hidden_len):
        super().__init__()
        self.input_len = input_len
        self.hidden_len = hidden_len

        # forget gate components
        self.linear_forget_x = nn.Linear(self.input_len, self.hidden_len, bias=True)
        self.linear_forget_h = nn.Linear(self.hidden_len, self.hidden_len, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components
        self.linear_input_x = nn.Linear(self.input_len, self.hidden_len, bias=True)
        self.linear_input_h = nn.Linear(self.hidden_len, self.hidden_len, bias=False)
        self.sigmoid_input = nn.Sigmoid()

        # cell memory components
        self.linear_newmemory_x = nn.Linear(self.input_len, self.hidden_len, bias=True)
        self.linear_newmemory_h = nn.Linear(self.hidden_len, self.hidden_len, bias=False)
        self.activation_newmemory = nn.Tanh()

        # out gate components
        self.linear_output_x = nn.Linear(self.input_len, self.hidden_len, bias=True)
        self.linear_output_h = nn.Linear(self.hidden_len, self.hidden_len, bias=False)
        self.sigmoid_output = nn.Sigmoid()

        self.activation_cell2hidden = nn.Tanh()

    def forget_gate(self, x, h):
        x = self.linear_forget_x(x)
        h = self.linear_forget_h(h)
        return self.sigmoid_forget(x + h)

    def input_gate(self, x, h):
        x = self.linear_input_x(x)
        h = self.linear_input_h(h)
        return self.sigmoid_input(x + h)
        
    def new_memory(self, i, f, x, h, c_prev):
        x = self.linear_newmemory_x(x)
        h = self.linear_newmemory_h(h)
        # new information part that will be injected in the new context
        k = self.activation_newmemory(x + h)
        # forget old context/cell info and learn new context/cell info
        c_next = f * c_prev + i * k
        return c_next

    def out_gate(self, x, h):
        x = self.linear_output_x(x)
        h = self.linear_output_h(h)
        return self.sigmoid_output(x + h)

    def forward(self, x, tuple_in):
        (h, c_prev) = tuple_in
        # Equation 1. input gate
        i = self.input_gate(x, h)

        # Equation 2. forget gate
        f = self.forget_gate(x, h)

        # Equation 3. updating the cell memory
        c_next = self.new_memory(i, f, x, h, c_prev)

        # Equation 4. calculate the main output gate
        o = self.out_gate(x, h)

        # Equation 5. produce next hidden output
        h_next = o * self.activation_cell2hidden(c_next)
        return h_next, c_next