from gym.envs.registration import register

register(
    id='gym_MC/MC-v0',
    entry_point='gym_MC.envs:MC',
    max_episode_steps = None,
)