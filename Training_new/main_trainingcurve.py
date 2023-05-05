import torch
import numpy as np
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import resample

if __name__ == "__main__":
    d = []
    for seed_idx in tqdm(range(1, 5)):
        model_path = "model/" + "PLOS" + "/" + f"v_{seed_idx}"
        file_trained_list = os.listdir(model_path)
        fs = [re.search("(.*)_(.*).pt", x) for x in file_trained_list]
        fs = [x for x in fs if x is not None]
        its = [x.group(2) for x in fs]
        tid = np.argmax([int(x) for x in its])
        model_file = os.path.join(model_path, fs[tid].group(0))
        modeldata = torch.load(model_file)
        td = modeldata['training_info']['rewardrate_smooth']
        d.append(td)
    d = np.vstack(d)
    d = d[:, 200000:800001]
    n = d.shape[1]
    e = d[:,np.arange(0,n, 1000)]
    # print(d)
    plt.figure()
    plt.plot(e.T)
    plt.xlabel('training episode')
    plt.ylabel('average reward rate')
    plt.show()


