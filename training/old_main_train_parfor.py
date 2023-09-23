# from multiprocessing import Process, Pool, freeze_support, RLock, Lock

# def trainver(seed_idx, config, veri):
#     if veri == 1:
#         keys = config['ConsTrans']
#     elif veri == 2:
#         keys = config['RandTrans']
#     elif veri == 3:
#         keys = config['FlipTrans']
#     train_2frame(seed_idx,  veri, key = keys, verbose= True)
# class W_logger1(W_Logger):
#     def __init__(self):
#         super().__init__()
#     def _init(self):
#         self.info['pamb'] = np.zeros_like(self.info['rewards'])
#     def _update(self, infogame):
#         self.info['pamb'][self.episode] = infogame[0]['params']['p_ambiguity']
#     def _getdescription(self, str):
#         episode = self.episode
#         pa = self.info['pamb'][max(0, episode-self.smooth_interval):(episode+1)]
#         r = self.info['rewards'][max(0, episode-self.smooth_interval):(episode+1)]
#         if episode > 100:
#             av = [r[pa < 0.1].mean(),r[pa ==0.5].mean(),r[pa >= 0.9].mean()]
#             str = f"R0 = {av[0]/300:.2f}, R50 = {av[1]/300:.2f}, R100 = {av[2]/300:.2f}"
#         else:
#             str = str
#         return str
#     # trainver(1, config, 1)
#     freeze_support()
#     proc = []
#     for seed_idx in range(1, 5):
#         for veri in range(1, 3):
#             p = Process(target = trainver, args = (seed_idx, config, veri))
#             p.start()
#             proc.append(p)

#     for p in proc:
#         p.join()
