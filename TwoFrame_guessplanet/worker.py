from W_RNN.W_Worker import W_Worker
import torch

class worker(W_Worker):
    def __init__(self, env, model, device=None, *arg, **kwarg):
        super().__init__(env, model, device, *arg, **kwarg)

    def select_action(self, action_vector, mode_action):
        if mode_action == "softmax":
            action_dist = torch.nn.functional.softmax(action_vector, dim = -1)
            action_cat = torch.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
        return action