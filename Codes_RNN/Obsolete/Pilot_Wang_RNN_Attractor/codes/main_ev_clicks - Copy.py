from envs.task_evidence_accumulation import Evidence_accumulation
from visualize import visualize
from oracle import oracle
md = oracle()
env = Evidence_accumulation(render_mode="human")
visualize(env, md, n_rep=1)


