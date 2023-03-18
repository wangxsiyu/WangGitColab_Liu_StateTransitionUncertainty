from gym_MC.envs import MC
from visualize import visualize
from MC_oracle import MC_oracle

env = MC(render_mode="human")
md = MC_oracle()
visualize(env, md, n_rep=1)

