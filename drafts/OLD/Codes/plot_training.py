import matplotlib.pyplot as plt
import pickle
with open("mymodel2.pkl", "rb") as f: 
    r, l = pickle.load(f)
fig, ax = plt.subplots(2)
ax[0].plot(r)
ax[0].set_ylim([0,100])
ax[1].plot(l)


plt.show()

