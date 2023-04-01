from multiprocessing import Process, Pool, freeze_support, RLock, Lock
from main_train_2frame_curriculum import train_2frame_curriculum
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument('-c','--nseed', type = str, default = '8')
    args = parser.parse_args()
    n_seeds = int(args.nseed)
    print(n_seeds)
    freeze_support()
    proc = []
    for seed_idx in range(1, n_seeds + 1):
        p = Process(target = train_2frame_curriculum, args = (seed_idx, ))
        p.start()
        proc.append(p)

    for p in proc:
        p.join()
