import numpy as np
import gym
from random import randint
from functools import reduce

import gym.spaces as spaces

def space_tuple(tpl):
    return tuple([spaces.Discrete(n) for n in tpl])

def counts(dim_list):
    return [reduce(lambda a, x: a*x, dim_list[i:]) for i in range(len(dim_list))]
        
class DiscreteTuple(spaces.Discrete):

    def __init__(self, ntuple):
        self.counts = counts(ntuple)
        spaces.Discrete.__init__(self, self.counts[0])

    def interpret(self, n):
        l = []
        q = n
        for p in self.counts[1:]:
            l.append(q // p)
            q = q % p
        l.append(q)

        return l

"""
for _ in range(100000):
    action = env.action_space.sample()

    res_old = env.res
    mul_old = env.mul
    bw_old = env.bw_fun(res_old, mul_old)
    gain_old = env.gain_fun(res_old, mul_old)
    bias_old = env.bias_fun(res_old, mul_old)
    reward_old = env._reward()

    _, r, _, dd = env.step(action)

    res_new = env.res
    mul_new = env.mul
    bw_new = env.bw_fun(res_new, mul_new)
    gain_new = env.gain_fun(res_new, mul_new)
    bias_new = env.bias_fun(res_new, mul_new)
    reward_new = env._reward()

    if (r > 20):
        print(dd)
        print("reward diff {}".format(r))

        print("max_bias {}".format(env.max_bias/1e-3))
        print("min_bw {}".format(env.min_bw/1e9))
        print("min_gain {}".format(env.min_gain/1))

        print("old:")
        print("{} [res] / {} [mul]".format(res_old, mul_old))
        print("{} [bw] / {} [gain] / {} [bias]".format(bw_old, gain_old, bias_old))
        print("reward {}".format(reward_old))

        print("new:")
        print("{} [res] / {} [mul] /".format(res_new, mul_new))
        print("{} [bw] / {} [gain] / {} [bias]".format(bw_new, gain_new, bias_new))
        print("reward {}".format(reward_new))

        break

"""
