import pickle

def func(a):
    with open('savename.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([a], f)