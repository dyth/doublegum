import jax
from policies_disc.agents.DQN       import DQN
from policies_disc.networks.network import Network


class DDQN(DQN):
    def __init__(self, obs, num_actions, args):
        obs_dim         = obs.shape[0]
        self.network_fn = lambda obs: Network(num_actions, args.network, args.network_groups)(obs)
        self.initialize_all(obs, args)


    def network_loss(self, params, state, batch, seed):
        state        = jax.lax.stop_gradient(state)
        seed1, seed2 = jax.random.split(seed)

        logits      = self.network_model_apply(state.network.params, state.network.sn_state, seed1, batch['next_obs'])[0].pop('q')
        next_action = logits.argmax(-1)

        target_info = self.network_model_apply(state.target.network, state.network.sn_state, seed1, batch['next_obs'])[0]
        target_Q    = target_info.pop('q')
        target_Q    = target_Q[self.index, next_action]
        discount    = self.args.discount ** self.args.nstep
        done        = 1. - batch['done'].squeeze()
        target_Q    = done * discount * target_Q
        target_Q    = batch['rew'].squeeze() + target_Q

        online_info, sn_state = self.network_model_apply(params, state.network.sn_state, seed2, batch['obs'], update_stats=True)
        online_Q              = online_info.pop('q')
        action                = batch['act'].squeeze().astype(int)
        online_Q              = online_Q[self.index, action]

        td_loss   = online_Q - target_Q
        loss      = td_loss ** 2.
        loss_mean = loss.mean()
        loss_std  = loss.std()

        aux = {
            'td_loss'    : td_loss.mean(),
            'td_loss_std': td_loss.std(),
            'sn_state'   : sn_state,
            'loss'       : loss_mean,
            'loss_std'   : loss_std,

            'target_Q_mean': target_Q.mean(),
            'online_Q'     : online_Q.mean(),
            'target_Q_std' : target_Q.std(),
            'online_Q_std' : online_Q.std(),
        }
        return loss_mean, aux
