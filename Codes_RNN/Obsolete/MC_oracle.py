import numpy as np

class MC_oracle():
    stage = 0
    goal = None
    actionlists = None
    def __init__(self):
        self.reset()
    def reset(self):
        self.stage = 0
        self.goal = None
        self.actionlists = None
    def predict(self, observation, deterministic = True, is_eval = True):
        action = 4
        cccs = np.array([0,2,6,8])
        ccs = np.array([3,1,5,7])
        if self.stage == 0:
            obs = observation[cccs]
            if any(obs > 0):
                self.goal = cccs[obs == 1]
                self.actionlists = self.get_actionlists(self.goal)
                self.stage = 1
        elif self.stage == 1:
            obs = observation[ccs]
            if any(obs > 0):
                self.actionlists = np.intersect1d(self.actionlists, ccs[np.where(obs == 1)[0]])
                self.stage = 2
        elif self.stage == 2:
            if observation[4] == 0: # center dot goes away
                action = np.random.choice(self.actionlists,1)[0]
            if any(observation[cccs]) > 0:
                self.stage = 3
        elif self.stage == 3:
            action = self.goal[0]
            self.stage = 3
            if all(observation == 0):
                self.reset()
                # print('new trial, reset')
        # print(self.stage, action)


        return(action)


    def get_actionlists(self, goal):
        goal = np.where(goal == np.array([0,2,8,6]))[0]
        te = np.array([goal[0], goal[0]+1])
        te[te == 4] = 0
        ccs = np.array([3,1,5,7])
        return ccs[te]