from W_Trainer import W_Trainer, W_loss
from W_Worker import W_Buffer
import torch
from collections import namedtuple 
import numpy as np
class W_loss2(W_loss):
    loss_name = None
    def __init__(self, loss, device):
        super().__init__(loss, device)  

    def loss_A2C(self, buffer, trainingbuffer):
        gamma = self.params['gamma']
        (_, _, _, reward, _, done,_,_,_) = buffer
        (action_dist, guess_dist, value, action_likelihood, guess_likelihood, l1, l2, l3, d1, d2, d3) = trainingbuffer
        rguess = reward[:,:,1]
        rsb1 = reward[:,:,2]
        rsb2 = reward[:,:,3]
        rsb0 = reward[:,:,4]
        reward = reward[:,:,0]
        # bootstrap discounted returns with final value estimates
        nbatch = done.shape[0]
        nstep = done.shape[1]
        returns = torch.zeros((nbatch,)).to(self.device)
        advantages = torch.zeros((nbatch,)).to(self.device)
        last_value = torch.zeros((nbatch,)).to(self.device)
        all_returns = torch.zeros((nbatch, nstep)).to(self.device)
        all_advantages = torch.zeros((nbatch, nstep)).to(self.device)
        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(nstep)):
            mask = 1 - done[:,t]
            returns = reward[:,t] + returns * gamma * mask
            deltas = reward[:,t] + last_value * gamma * mask - value[:,t].data
            
            advantages = advantages * gamma * mask + deltas

            all_returns[:,t] = returns 
            all_advantages[:,t] = advantages
            last_value = value[:,t].data

        logll = torch.log(action_likelihood)
        policy_loss = -(logll * all_advantages).mean()
        value_loss = 0.5 * (all_returns - value).pow(2).mean()
        entropy_reg = -(action_dist * torch.log(action_dist)).mean()

        loss_actor = policy_loss - self.params['coef_entropyloss'] * entropy_reg
        loss_critic = self.params['coef_valueloss'] * value_loss  

        # loss for guess
        guessll = torch.log(guess_likelihood)
        loss_guess = -(guessll * rguess).mean()

        # loss for safebet
        loss_sb1 = -(torch.log(l1) * rsb0).mean()
        loss_sb2 = -(torch.log(l2) * rsb1).mean()
        loss_sb3 = -(torch.log(l3) * rsb2).mean()


        loss = loss_actor + loss_critic + loss_guess + loss_sb1 + loss_sb2  + loss_sb3

        if torch.isinf(loss):
            print('check')
        return loss

class trainer(W_Trainer):
    def __init__(self, env, model, param_loss, param_optim, logger=None, device=None, gradientclipping=None, seed=None, position_tqdm=0, *arg, **kwarg):
        super().__init__(env, model, param_loss, param_optim, logger, device, gradientclipping, seed, position_tqdm, *arg, **kwarg)
        self.loss = W_loss2(param_loss, device = device)
        
        Memory = namedtuple('Memory', ('obs', 'guess', 'action', 'reward', 'timestep', 'done', \
                                       'sb1', 'sb2', 'sb3'))
        self.memory = W_Buffer(Memory, device = device, *arg, **kwarg)

    def run_episode_outputlayer(self, buffer):
        self.model.train()
        (obs, guess, action, _,_,_, sb1, sb2, sb3) = buffer
        action = action.to(self.device)
        guess = guess.to(self.device)
        mem_state = None
        action_vec, val_estimate, mem_state, v = self.model(obs.to(self.device), mem_state)
        action_vec = action_vec.permute((1,0,2))
        v = v.permute((1,0,2))
        guess_dist = torch.nn.functional.softmax(action_vec[:,:,2:4], dim = -1)
        action_dist = torch.nn.functional.softmax(action_vec[:,:,(0,1,4)], dim = -1)


        dist1 = torch.nn.functional.softmax(v[:,:,0:2], dim = -1)
        dist2 = torch.nn.functional.softmax(v[:,:,2:4], dim = -1)
        dist3 = torch.nn.functional.softmax(v[:,:,4:6], dim = -1)

        eps = 1e-4
        guess_dist = guess_dist.clamp(eps, 1-eps)
        action_dist = action_dist.clamp(eps, 1-eps)
        dist1 = dist1.clamp(eps, 1-eps)
        dist2 = dist2.clamp(eps, 1-eps)
        dist3 = dist3.clamp(eps, 1-eps)

        val_estimate = val_estimate.permute((1,0,2)).squeeze(2)
        action_likelihood = (action_dist * action).sum(-1)
        guess_likelihood = (guess_dist * guess).sum(-1)
        l1 = (dist1 *sb1).sum(-1)
        l2 = (dist2 *sb2).sum(-1)
        l3 = (dist3 *sb3).sum(-1)

        tb = namedtuple('TrainingBuffer', ("action_dist","guess_dist", "value", \
                                           "action_likelihood","guess_likelihood", \
                                            "l1","l2","l3","d1","d2","d3"))
        return tb(action_dist, guess_dist, val_estimate, action_likelihood, guess_likelihood, \
                  l1, l2, l3, dist1, dist2, dist3)
    
    def select_action(self, action_vector, mode_action):
        if mode_action == "softmax":
            guess_vector = action_vector[:,:,2:4]
            action_vector = action_vector[:,:,(0,1,4)]
            guess_dist = torch.nn.functional.softmax(guess_vector, dim = -1)
            action_dist = torch.nn.functional.softmax(action_vector, dim = -1)
            guess_cat = torch.distributions.Categorical(guess_dist.squeeze())
            action_cat = torch.distributions.Categorical(action_dist.squeeze())
            guess = guess_cat.sample()
            action = action_cat.sample()
        return action, guess
    
    
    def select_safebet(self, v):
        dist1 = torch.nn.functional.softmax(v[:,:,0:2], dim = -1)
        dist2 = torch.nn.functional.softmax(v[:,:,2:4], dim = -1)
        dist3 = torch.nn.functional.softmax(v[:,:,4:6], dim = -1)
        
        cat1 = torch.distributions.Categorical(dist1.squeeze())
        cat2 = torch.distributions.Categorical(dist2.squeeze())
        cat3 = torch.distributions.Categorical(dist3.squeeze())

        sb1 = cat1.sample()
        sb2 = cat2.sample()
        sb3 = cat3.sample()
        return sb1,sb2,sb3
    
    def run_episode(self, mode_action = "softmax"):
        self.model.eval()
        done = False
        total_reward = 0
        obs = self.env.reset()
        mem_state = None
        self.memory.clear()
        while not done:
            # take actions
            obs = torch.from_numpy(obs).unsqueeze(0).float()
            action_vector, val_estimate, mem_state_new, conf_vector  = self.model(obs.unsqueeze(0).to(self.device), mem_state)
            action, guess = self.select_action(action_vector, mode_action)
            sb1, sb2, sb3 = self.select_safebet(conf_vector)
            obs_new, reward, done, timestep, _ = self.env.step(action.item(), guess.item(), sb1.item(), sb2.item(), sb3.item())
            # reward = float(reward)
            # rguess = float(rguess)
            action_onehot = torch.nn.functional.one_hot(action, 3)
            action_onehot = action_onehot.unsqueeze(0).float()
            guess_onehot = torch.nn.functional.one_hot(guess, 2)
            guess_onehot = guess_onehot.unsqueeze(0).float()

            
            sb1_onehot = torch.nn.functional.one_hot(sb1, 2)
            sb1_onehot = sb1_onehot.unsqueeze(0).float().to('cpu').numpy()
            sb2_onehot = torch.nn.functional.one_hot(sb2, 2)
            sb2_onehot = sb2_onehot.unsqueeze(0).float().to('cpu').numpy()
            sb3_onehot = torch.nn.functional.one_hot(sb3, 2)
            sb3_onehot = sb3_onehot.unsqueeze(0).float().to('cpu').numpy()
            reward = np.expand_dims(reward,0)

            self.memory.add(obs.to('cpu').numpy(), guess_onehot.to('cpu').numpy(), \
                             action_onehot.to('cpu').numpy(), reward, [timestep], [done], \
                                sb1_onehot, sb2_onehot, sb3_onehot)
            
            obs = obs_new
            mem_state = mem_state_new
            total_reward += reward.sum()

        self.memory.push()
        return total_reward
    















