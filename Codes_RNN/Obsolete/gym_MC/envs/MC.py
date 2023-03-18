import gym
from gym import spaces
import numpy as np
import math
# from gym.utils.renderer import Renderer
import pygame

class MC(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array", "single_rgb_array"], "render_fps": 20}
    params = dict()
    def __init__(self, render_mode = None, window_size = (512, 512), render_fps = 20, preward = 1):
        print(f'render mode: {render_mode}')
        self.window_size = window_size # PyGame window size
        self.metadata['render_fps'] = render_fps
        self.timestep = 1000/render_fps
        self.preward = preward
        self.reset_oracle() # change better image based on block, for now, oracle is always image 1
        # observation: 9 locations
        self.rect_middle = np.array([0.03, 0.03]) * window_size
        self.radius_image = np.mean(np.array([0.05]) * window_size)
        self.position = np.array([[0.1, 0.1],[0.5, 0.1],[0.9, 0.1],
            [0.1, 0.5],[0.5, 0.5],[0.9, 0.5],
            [0.1, 0.9],[0.5, 0.9],[0.9, 0.9]]) * window_size
        low = np.array(np.zeros(9)).astype(np.float32)
        high = np.array(np.ones(9) * 4).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        self.n_actions = 9
        self.action_space = spaces.Discrete(self.n_actions)
        self.setup_rendermode(render_mode)

    def num_actions(self):
        return self.n_actions

    def reset(self, seed = None, return_info = False):
        super().reset(seed = seed)
        self.reset_trial()
        # clean the render collection and add the initial frame
        # self.renderer.reset()
        # self.renderer.render_step()
        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()
        info = self._get_info()
        return obs if not return_info else (obs, info)

    def reset_oracle(self):
        self.oracle = 1 #np.random.randint(1,3)

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.
        import pygame # avoid global pygame dependency. This method is not called with no-render.
    
        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 0, 0))

        for i in range(9):
            if self.observation[i] > 0:
                if i % 2 == 1: # middle point observations
                    pygame.draw.rect(canvas,
                        (255,255,255),
                        np.concatenate((-self.rect_middle + self.position[i],self.rect_middle * 2), axis = None),
                        0)
                elif i == 4: # fixation point observations
                    pygame.draw.circle(
                            canvas,
                            (255, 255, 255),
                            self.position[i],
                            self.radius_image/5,
                        )
                else: # images observations
                    if self.observation[i] == 2:
                        tcol = (0, 0, 255)
                    elif self.observation[i] == 1:
                        tcol = (0, 255, 0)
                    else:
                        tcol = (100,100,100)
                    pygame.draw.circle(
                            canvas,
                            tcol,
                            self.position[i],
                            self.radius_image,
                        )

        if (self.action is not None) and (self.action != 0):
            # print(self.position, self.action)
            pygame.draw.circle(
                    canvas,
                    (255, 0, 0),
                    self.position[self.action],
                    self.radius_image/7,
                )

        if mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            # self.window.blit(canvas_text, (1,1))
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def advance_stage(self):
        reward_advance = 10
        self.stage = self.stage + 1
        self.timer = 0
        return reward_advance

    def reset_trial(self):
        self.reward = 0
        self.observation = np.zeros(9)
        self.observation[4] = 1
        self.action = None
        self.stage = 0
        self.timer = 0
        cccs = np.array([0,2,8,6])
        ccs = np.array([3,1,5,7])
        self.ccc_id = np.random.choice(4,2, replace = False)
        self.ccc_pos = cccs[self.ccc_id]
        self.cc_id = np.array([0,2]) + np.random.choice(2,1)
        self.cc_pos = ccs[self.cc_id]
        self.options3 = None
        self.recorded_choice_cc = None
        self.recorded_choice_ccc = None
        self.images3 = None
        self.iti = 500
        # print(cccs, self.ccc_pos)
        # print(ccs, self.cc_pos)
    def step(self, action):
        reward = 0
        is_error = 0
        self.action = action
        done = False
        self.timer += self.timestep
        if self.stage == 0: # acquire gaze 
            if action == 4:
                reward += self.advance_stage()
            if self.timer > 100.0:
                is_error = 1
        elif self.stage == 1: # fixation
            if action == 4:
                if self.timer >= 150.0:
                    reward += self.advance_stage()
            else:
                is_error = 1
        elif self.stage == 2: # flash target
            if action == 4:
                if self.timer <= 50.00:
                    self.observation[4] = 1
                    self.observation[self.ccc_pos[0]] = 1
                    self.observation[self.ccc_pos[1]] = 2
                else:
                    reward += self.advance_stage()
                    self.observation = np.zeros(9)
                    self.observation[4] = 1
            else:
                is_error = 1
        elif self.stage == 3: # flash 1 after
            if action == 4:
                if self.timer >= 300.00:
                    reward += self.advance_stage()
            else:
                is_error = 1
        elif self.stage == 4: # flash direction
            if action == 4:
                if self.timer <= 50.00:
                    self.observation[self.cc_pos[0]] = 1
                    self.observation[self.cc_pos[1]] = 1
                else:
                    reward += self.advance_stage()
                    self.observation = np.zeros(9)
                    self.observation[4] = 1
            else:
                is_error = 1
        elif self.stage == 5: # flash 2 after
            if action == 4:
                if self.timer >= 400.00:
                    reward += self.advance_stage()
                    self.observation = np.zeros(9)
                    self.observation[1] = 1
                    self.observation[3] = 1
                    self.observation[5] = 1
                    self.observation[7] = 1
            else:
                is_error = 1
        elif self.stage == 6: # choice 1
            if self.recorded_choice_cc is None:
                if self.timer > 500.0:
                    is_error = 1
                elif (action in self.cc_pos):
                    self.recorded_choice_cc = action
                    self.observation = np.zeros(9)
                    self.observation[action] = 1
            else:
                if action != self.recorded_choice_cc:
                    is_error = 1
                else:
                    if self.timer > 600.0:
                        reward += self.advance_stage()
                        self.observation = np.zeros(9)
                        self.options3 = self.get_valid_option3(action)
                        self.observation[self.options3[0]] = 3
                        self.observation[self.options3[1]] = 3
                        self.images3 = self.get_images3(self.options3)
        elif self.stage == 7: # choice 2
            if self.recorded_choice_ccc is None:
                if self.timer > 700.0:
                    is_error = 1
                elif (action in self.options3):
                    reward += 1
                    self.recorded_choice_ccc = action
                    self.observation = np.zeros(9)
                    self.observation[action] = self.images3[np.where(self.options3 == action)[0]]                    
            else:
                if action != self.recorded_choice_ccc:
                    is_error = 1
                else:
                    if self.timer > 800.0:
                        reward +=  self.advance_stage()
                        if action == self.ccc_pos[0]:
                            reward += 1000
                        elif action == self.ccc_pos[1]:
                            reward += 0
                        else:
                            print('choose neither target, should not happen, check!')
        elif self.stage == 8:
            self.observation = np.zeros(9)
            if self.timer > self.iti:
                done = True
        
        if is_error:
            reward -= 10
            done = True
        else:
            reward += 1
            
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = self._get_info()
        self.reward = self.reward + reward
        return obs, reward, done, info
    
    def get_images3(self, options3):
        images3 = np.zeros(len(options3))
        for i in range(len(images3)):
            if options3[i] in self.ccc_pos:
                images3[i] = np.where(self.ccc_pos == options3[i])[0] + 1
            else:
                images3[i] = 3
        return images3

    def get_valid_option3(self, action):
        action = np.where(action == np.array([3,1,5,7]))[0][0]
        te = np.array([action, action-1])
        te[te == -1] = 3
        cccs = np.array([0,2,8,6])
        return cccs[te]
        
    def _get_obs(self):
        return self.observation

    def _get_info(self):
        return {'total-reward': self.reward}

    def setup_rendermode(self, render_mode = None):
        self.render_mode = render_mode
        if self.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            # pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None
        # self.renderer = Renderer(self.render_mode, self._render_frame)

    def render(self):
        return self._render_frame(self.render_mode)
        # Just return the list of render frames collected by the Renderer.
        # return self.renderer.get_renders()

    def close(self):
        if self.window is not None:
            import pygame 
            pygame.display.quit()
            pygame.quit()