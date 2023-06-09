import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from tqdm import tqdm 
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator

'''plotting functions'''

def anova_2way(base_path, mode, n_seeds=10, base_seed=1111):
    data = []
    for seed_idx in range(1, n_seeds + 1):

        basename = os.path.basename(base_path)
        subpath = f"{basename}_{seed_idx}_{base_seed*seed_idx}.npy"
        if mode is not None: 
            subpath = f"{basename}_{seed_idx}_{base_seed*seed_idx}_{mode}.npy"
        
        path = os.path.join(base_path, basename+f"_{seed_idx}", subpath)
        stay_probs = np.load(path)
        
        for i in range(stay_probs.shape[0]):
            for j in range(stay_probs.shape[1]):
                data += [{
                    'reward': 1-i,
                    'common': 1-j,
                    'prob': stay_probs[i,j,0]
                }]

    df = pd.DataFrame(data)

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    #perform two-way ANOVA
    model = ols('prob ~ C(reward) + C(common) + C(reward):C(common)', data=df).fit()
    result = sm.stats.anova_lm(model, typ=2)
    print(result)

def plot_seeds(save_path, 
            load_path, 
            mode,
            y_lim=0.5,
            n_seeds=8, 
            base_seed=1111,
            title="Two-Step Task"):

    _, ax = plt.subplots()
    common_sum = np.array([0.,0.])
    uncommon_sum = np.array([0.,0.])
    dir_base_path = os.path.basename(load_path)

    for seed_idx in range(1, n_seeds + 1):

        ax.set_ylim([y_lim, 1.0])
        ax.set_ylabel('Stay Probability')
        
        base_path = f"{os.path.basename(load_path)}_{seed_idx}_{base_seed*seed_idx}.npy"
        if mode is not None: 
            base_path = f"{os.path.basename(load_path)}_{seed_idx}_{base_seed*seed_idx:04d}_{mode}.npy"
        path = os.path.join(load_path, dir_base_path+f"_{seed_idx}", base_path)
        stay_probs = np.load(path)

        common = [stay_probs[0,0,0], stay_probs[1,0,0]]
        uncommon = [stay_probs[0,1,0], stay_probs[1,1,0]]
        
        common_sum += np.array(common)
        uncommon_sum += np.array(uncommon)

        ax.set_xticks([1.5,3.5])
        ax.set_xticklabels(['Rewarded', 'Unrewarded'])

        plt.plot([1,3], common, 'o', color='black')
        plt.plot([2,4], uncommon, 'o', color='black')
        
    c  = plt.bar([1,3], (1. / n_seeds) * common_sum, color='b', width=0.5)
    uc = plt.bar([2,4], (1. / n_seeds) * uncommon_sum, color='r', width=0.5)
    ax.legend( (c[0], uc[0]), ('Common', 'Uncommon') )
    ax.set_title(title)
    plt.show()
    # plt.savefig(save_path)

def compare_rewards(load_path_mrl, load_path_emrl, save_path, title):

    mrl_cued   = np.load(os.path.join(load_path_mrl, "mrl_reward_cued.npy"))
    mrl_uncued = np.load(os.path.join(load_path_mrl, "mrl_reward_uncued.npy"))

    emrl_cued   = np.load(os.path.join(load_path_emrl, "emrl_reward_cued.npy"))
    emrl_uncued = np.load(os.path.join(load_path_emrl, "emrl_reward_uncued.npy"))

    t_state, p_val = stats.ttest_ind(mrl_cued, emrl_cued) 
    print(f"Cued --> P-Value: {p_val} | T-Statistic: {t_state}")

    t_state, p_val = stats.ttest_ind(mrl_uncued, emrl_uncued) 
    print(f"Uncued --> P-Value: {p_val} | T-Statistic: {t_state}")

    t_state, p_val = stats.ttest_ind(np.stack([mrl_cued, mrl_uncued]).mean(axis=0), np.stack([emrl_cued, emrl_uncued]).mean(axis=0)) 
    print(f"Total --> P-Value: {p_val} | T-Statistic: {t_state}")

    _, ax = plt.subplots()
    ax.set_ylim([0.5, 0.8])
    ax.set_xticks([1.5,3.5])
    ax.set_xticklabels(['Uncued', 'Cued'])

    mrl = plt.bar([1.2,3.2], 
        [mrl_uncued.mean(), mrl_cued.mean()], 
        yerr=[mrl_uncued.std(), mrl_cued.std()], 
        color='orange', 
        width=0.5
    )
    
    emrl = plt.bar([1.8,3.8], 
        [emrl_uncued.mean(), emrl_cued.mean()], 
        yerr=[emrl_uncued.std(), emrl_cued.std()], 
        color='gray', 
        width=0.5
    )
    ax.legend((mrl[0], emrl[0]), ('MRL', 'EMRL'))
    # ax.set_title(title)
    ax.set_ylabel(title)
    plt.show()

    print(f"MRL: Cued {mrl_cued.mean()} | Uncued {mrl_uncued.mean()}")
    print(f"EMRL: Cued {emrl_cued.mean()} | Uncued {emrl_uncued.mean()}")

def read_data(load_dir, tag="perf/avg_reward_10"):

    events = os.listdir(load_dir)
    for event in events:
        path = os.path.join(load_dir, event)
        ea = event_accumulator.EventAccumulator(path, size_guidance={ 
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 10_000,
                event_accumulator.HISTOGRAMS: 0,
        })
        
        ea.Reload()
        tags = ea.Tags()

        if tag not in tags["scalars"]: continue

        if len(ea.Scalars(tag)) == 10_000:
            return np.array([s.value for s in ea.Scalars(tag)])

    return None 

def plot_rewards_curve(save_path, load_path_epi, load_path_inc, n_seeds=10):

    epi_data = np.zeros((n_seeds, 10_000))
    inc_data = np.zeros((n_seeds, 10_000))

    for seed_idx in tqdm(range(n_seeds)):
        epi_event = read_data(load_dir=load_path_epi+f"_{seed_idx+1}")
        inc_event = read_data(load_dir=load_path_inc+f"_{seed_idx+1}")
        if epi_event is None or inc_event is None: 
            raise ValueError()
        epi_data[seed_idx] = epi_event 
        inc_data[seed_idx] = inc_event

    epi_mean = epi_data.mean(axis=0)
    inc_mean = inc_data.mean(axis=0)

    plt.plot(epi_mean)
    plt.plot(inc_mean)
    plt.legend(["Episodic", "Incremental"])
    plt.title("Episodic vs Incremental Training Curves")
    plt.savefig(save_path)


if __name__ == "__main__":

    #### Two-Way ANOVA ####
    # anova_2way("ckpt/TwoStepEp_12", mode="episodic")
    # anova_2way(".\\ckpt\\TwoStep_71", mode=None, n_seeds=8)

    #### Episodic Plot ####
    plot_seeds(
        save_path="TwoStep_71", 
        load_path="TwoStep_71",
        mode=None,
        y_lim=0, 
        n_seeds=8, 
        base_seed=1995,
        title="Episodic"
    )

    #### Compare Training Curves ####
    # plot_rewards_curve(
    #     save_path="./assets/epi_inc_rewards.png",
    #     load_path_epi="./logs/TwoStep_71",
    #     load_path_inc="./logs/TwoStep_71",
    #     n_seeds=8
    # )

    #### Compare Rewards ####
    # compare_rewards(
    #     load_path_mrl="./ckpt/TwoStep_71",
    #     load_path_emrl="./ckpt/TwoStep_71",
    #     save_path=None,
    #     title="Performance" 
    # )
    