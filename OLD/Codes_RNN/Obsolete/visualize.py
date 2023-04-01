import numpy as np
from gym.wrappers import RecordVideo

def wrap_env(env):
    log_dir = './video'
    env = RecordVideo(env, log_dir)
    # print(env.recorded_frames)
    return env

def visualize(tenv, model = None, is_record = False, n_rep = 1):
    if is_record:
        tenv = wrap_env(tenv)
    observation, info = tenv.reset(return_info = True)
    # n_frames = 100
    total_reward = 0
    # i = 0
    while True:
        # i = i + 1
        tenv.render()
        # action = {"gaze_dir": np.random.random_integers(0,4), 
        #           "gaze_step": np.random.random_integers(0,3), 
        #           "hit": np.random.random_integers(0,2)}
        if model is None:
            action = np.random.randint(0, tenv.num_actions())
        else:
            action = model.predict(observation, deterministic=True, is_eval = True)
        observation, reward, done, info = tenv.step(action)
        total_reward += reward
        # print(observation, reward, done, action)
        # print(f"reward:{reward}, tot:{total_reward}")
        if done:
            n_rep = n_rep - 1
            tenv.reset()
            if n_rep == 0:
                break
    print(total_reward)
    # print(info["total-reward"])
    tenv.close()