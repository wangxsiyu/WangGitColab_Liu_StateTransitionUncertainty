import pickle
import sys

def func(a):
    with open(f'savename_{a}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([a], f)

if __name__ == "__main__":
    func(sys.argv[1])