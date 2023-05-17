from main_train_2frame import train_2frame
import yaml

def train_2frame_curriculum(seed_idx):
    with open('curriculum.yaml', 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    keys = config.keys()
    lastver = None
    for k in keys:
        key = config[k]
        lastver = train_2frame(seed_idx, key, lastver, verbose= False)

if __name__ == "__main__":
    train_2frame_curriculum(1)