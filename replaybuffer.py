'''
With inspiration from
https://github.com/facebookresearch/drqv2/pull/17/files
https://github.com/ikostrikov/jaxrl/blob/haiku/replay_buffer.py
'''
import numpy as np
# from jax import numpy as np

class ReplayBuffer(object):
    def __init__(self, obs_dim, act_dim, max_size=int(1e6), discount=.99, nstep=1):
        self.index  = 0
        self.size   = 0
        self.length = 0

        self.max_size  = max_size
        self.discounts = np.power(discount, np.arange(nstep))
        self.nstep     = nstep

        self.obs  = np.zeros((max_size, obs_dim))
        self.act  = np.zeros((max_size, act_dim))
        self.nobs = np.zeros((max_size, obs_dim))
        self.rew  = np.zeros((2*max_size, 1))
        self.done = np.zeros((2*max_size, 1))        # done is 0 when not finished and 1 when done


    def add(self, obs, act, rew, next_obs, done):
        # self.obs.at[self.index].set(obs)
        # self.act.at[self.index].set(act)
        # self.rew.at[self.index].set(rew)
        # self.nobs.at[self.index].set(next_obs)
        # self.done.at[self.index].set(done)
        self.obs[self.index]  = obs
        self.act[self.index]  = act
        self.rew[self.index]  = rew
        self.nobs[self.index] = next_obs
        self.done[self.index] = done

        self.index   = (self.index + 1) % self.max_size
        self.size    = min(self.size + 1, self.max_size)
        self.length += 1
        
        
    # def sample(self, batch_size):
    #     index = np.random.randint(0, self.size-self.nstep, size=batch_size)
    #     # print(self.obs[index].shape)
    #     # print(self.act[index].shape)
    #     # print(self.rew[index].shape)
    #     # print(self.nobs[index].shape)
    #     # print(self.done[index].shape)
    #     # print(c)
    #     return {
    #         'obs'     : self.obs[index],
    #         'act'     : self.act[index],
    #         'rew'     : self.rew[index],
    #         'next_obs': self.nobs[index],
    #         'done'    : self.done[index]
    #     }


    def sample(self, batch_size):
        # half-cheetah, nstep3, this : ~ 200
        # half-cheetah, nstep3, cpprb: ~ 320
        first = np.random.randint(0, self.size-self.nstep, size=batch_size)
        last  = np.minimum(first + self.nstep, self.size)

        mask                                    = np.ones((batch_size, self.nstep+1))
        mask[np.arange(batch_size), last-first] = 0
        mask                                    = np.cumprod(mask, 1)
        mask                                    = mask[:, :self.nstep]

        def discounted_sum(f, l):
            rews      = self.rew[f:l]
            discounts = self.discounts[:l-f]
            return np.sum(rews * discounts) # this is not correct because it needs to account for the sequence

        # adapted from https://stackoverflow.com/a/45152908
        dones = self.done[first[:, None] + np.arange(self.nstep)]
        print(dones.shape, mask.shape)
        dones *= mask[:, :, None]
        dones = 1 - (1 - dones).prod(1)



        print(self.obs[first].shape)
        print(self.act[first].shape)
        print(np.vectorize(discounted_sum)(first, last)[:, None].shape)
        print(self.nobs[last].shape)
        print(dones.shape)
        print(c)

        return {
            'obs'     : self.obs[first],
            'act'     : self.act[first],
            'rew'     : np.vectorize(discounted_sum)(first, last)[:, None],
            'next_obs': self.nobs[last],
            'done'    : dones
        }
