from policies_disc.agents.DDQN  import DDQN
from policies_disc.networks.dueling import Dueling


class DuelingDDQN(DDQN):
    def __init__(self, obs, num_actions, args):
        obs_dim         = obs.shape[0]
        self.network_fn = lambda obs: Dueling(num_actions, args.network, args.network_groups)(obs)
        self.initialize_all(obs, args)
