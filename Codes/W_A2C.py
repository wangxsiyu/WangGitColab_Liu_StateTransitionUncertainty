import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class W_A2C():
    def __init__(self, model, discount_factor = 0.9, beta_v = 0.05, beta_entropy = 0.05):
        self.model = model
        self.optimizer = optim.RMSprop(model.parameters(), lr = 0.0007)
        self.discount_factor = discount_factor
        self.beta_v = beta_v
        self.beta_entropy = beta_entropy
        # self.criterion_entropy = HLoss()

    # def calculate_return(self, rewards, v, info):
    #     # torch.set_printoptions(profile="full")
    #     R = torch.zeros_like(rewards)
    #     nT = R.shape[0]
    #     R[nT-1] = rewards[nT-1]
    #     for i in reversed(range(nT-1)):
    #         if info[i,1] == 0:
    #             R[i] = rewards[i]
    #         else:
    #             R[i] = self.discount_factor * R[i+1] + rewards[i]
    #     return R

    def discount(self, x, gamma):
        out = x.clone()
        for i in reversed(range(x.size()[0]-1)):
            out[i] = out[i] + gamma * out[i+1]
        return out
    
    def _select_action_prob(self, a, actions):
        return torch.gather(a, 1, actions)

    def train_episodes(self, obs, actions, rewards, info):
        nstep = len(actions)
        # run model
        a0, v = self.model.forward_unrolled(obs, None, None)
        a = a0.squeeze()[0:-1,]
        action_prob = self._select_action_prob(a, actions) # select action_prob based on a and actions
        # compute bootstrapped return
        # r_bootstrapped = self.calculate_re
        # turn(rewards, v.detach(), info)
        rewards_discounted = self.discount(rewards, self.discount_factor)
        advantages = rewards + self.discount_factor * v[1:,] - v[:-1,]
        advantages = self.discount(advantages, self.discount_factor)
        self.optimizer.zero_grad()

        critic_loss = 0.5 * (rewards_discounted - v[:-1,]).pow(2).sum()

        actor_loss = (-torch.log(action_prob + 1e-7)*advantages.detach()).sum()
        entropy_loss = - torch.sum(torch.log(a + 1e-7) * a)
        loss = actor_loss + self.beta_v * critic_loss - self.beta_entropy * entropy_loss
        # loss = -loss
        loss.backward()
        self.optimizer.step()
        return loss.detach()


# class HLoss(nn.Module):
#     def __init__(self):
#         super(HLoss, self).__init__()

#     def forward(self, x):
#         x = x.squeeze()
#         b = F.softmax(x, dim=1) * F.log_softmax(x + 1e-7, dim=1)
#         b = -1.0 * b.sum()
#         return b
        