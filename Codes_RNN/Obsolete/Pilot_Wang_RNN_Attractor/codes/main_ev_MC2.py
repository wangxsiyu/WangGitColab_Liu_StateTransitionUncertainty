from envs.MC2 import MC2
from visualize import visualize
from oracle import oracle_MC2
md = oracle_MC2()
env = MC2(render_mode="human")
visualize(env, None, n_rep=1)


