import gym
from gym import spaces
import numpy as np
from scipy.stats import bernoulli
import math
import pygame

class Evidence_accumulation(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array", "single_rgb_array"], "render_fps": 20}
    params = dict()
    def __init__(self, render_mode = None, window_size = (512, 512), render_fps = 20):
        print(f'render mode: {render_mode}')
        self.window_size = window_size # PyGame window size
        self.metadata['render_fps'] = render_fps
        self.timestep = 1000/render_fps
        # observation: 3 locations (L, F, R)
        self.radius_dot = np.mean(np.array([0.05]) * window_size)
        self.position = np.array([[0.5, 0.5],[0.1, 0.5],[0.9, 0.5]]) * window_size
        low = np.array(np.zeros(3)).astype(np.float32)
        high = np.array(np.ones(3)).astype(np.float32)
        self.n_observations = 3
        self.observation_space = spaces.Box(low, high)
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)
        self.setup_rendermode(render_mode)

    def num_actions(self):
        return self.n_actions

    def num_observations(self):
        return self.n_observations

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

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.
        import pygame # avoid global pygame dependency. This method is not called with no-render.
    
        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 0, 0))

        for i in range(3):
            if self.observation[i] > 0:
                tcol = (100,100,100)
                pygame.draw.circle(
                        canvas,
                        tcol,
                        self.position[i],
                        self.radius_dot,
                    )
        pygame.draw.circle(
                canvas,
                (255, 0, 0),
                self.position[self.action],
                self.radius_dot/7,
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

    def step(self, action):
        reward = 0
        is_error = 0
        self.action = action
        done = False
        self.timer += self.timestep

        if self.timer < self.fix_period:
            if action > 0:
                is_error = True
        elif self.timer < self.stimulus_period:
            if action > 0:
                is_error = True
            sL = bernoulli.rvs(self.pL, 1)-1
            sR = 1-sL
            self.observation = np.array([1,sL,sR])
        elif self.timer < self.decision_period:
            self.observation = np.array([0,0,0])
            if action > 0:
                self.timer = self.decision_period
                self.chosen_action = action
                # print(self.correctside)
        elif self.timer < self.decision_period + 0:
            self.observation = np.array([0,self.chosen_action==1, self.chosen_action ==2])
            if action != self.chosen_action:
                is_error = 1
        elif self.chosen_action != 0:
            done = True
            if self.chosen_action == self.correctside:
                reward += 100
            else:
                reward += 0
        else:
            is_error = 1

        if is_error:
            reward -= 1
            done = True
        else:
            reward += 1
            
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = self._get_info()
        self.reward = self.reward + reward
        return obs, reward, done, info
    
    def reset_trial(self):
        self.observation = np.array([1,0,0])
        self.pL = np.random.random(1)
        # self.pR = 1 - self.pL
        self.correctside = (self.pL < 0.5) + 1
        self.fix_period = 50
        self.timer = 0
        self.stimulus_period = 1000 + self.fix_period
        self.decision_period = 500 + self.stimulus_period
        self.action = 0
        self.reward = 0
        self.chosen_action = 0

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