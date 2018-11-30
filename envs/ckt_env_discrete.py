"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces

import framework
import numpy as np
import random

from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
from framework.wrapper.CSAmpClass import CSAmpClass

## helper functions for working with files
def rel_path(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def load_array(fname):
    with open(rel_path(fname), "rb") as f:
        arr = np.load(f)
    return arr

def rel_diff(curr, desired):
    diff = (curr-desired)/np.mean(desired)#statistics.mean([curr,desired])
    return diff

class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

def one_hot(num, base):
    a = np.zeros(base)
    a[num] = 1
    return a

class CSAmp(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    framework_path = os.path.abspath(framework.__file__).split("__")
    CIR_YAML = framework_path[0]+"/yaml_files/cs_amp.yaml"

    def __init__(self, sparse=False):

        self.sparse = sparse
        with open(CSAmp.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        specs = yaml_data['target_specs']
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        self.specs_ideal = []
        self.specs_id = list(specs.keys())
        self.num_os = len(list(self.specs.values())[0])

        # param array
        params = yaml_data['params']
        self.params = []
        self.params_id = params.keys()

        for value in params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)


        #initialize sim environment
        dsn_netlist = yaml_data['dsn_netlist']
        self.sim_env = CSAmpClass(design_netlist=dsn_netlist)
        # self.action_space = spaces.Box(low=np.array(np.zeros(len(self.params_id))), high=np.ones(len(self.params_id)), dtype=np.float32)
        self.action_meaning = [-10,-5,-1,0,1,5,10]
        self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(low=np.array([CSAmp.PERF_LOW]*2*len(self.specs_id)+[-5.0,-5.0,-5.0,-5.0]),
                                            high=np.array([CSAmp.PERF_HIGH]*2*len(self.specs_id)+[5.0,5.0,5.0,5.0]))

        #initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)
        # _ = self.reset()

    def reset(self, z=None, sigma=0.2):
        """
        z in random variable from N(0,I)
        the sigma is arbitraty chosen for now
        """

        spec_size = len(self.specs_id)
        # select first o*
        rand_oidx = 0 #random.randint(0,self.num_os-1)
        # self.task_id = one_hot(rand_oidx, self.num_os)
        self.specs_ideal = []
        for spec in list(self.specs.values()):
            self.specs_ideal.append(spec[rand_oidx])

        if z is None:
            z = np.zeros(spec_size)

        cov = sigma*np.identity(spec_size)
        self.specs_ideal_norm = np.ones(spec_size) + np.matmul(cov, np.ones(spec_size)*z)

        self.cur_params_idx = np.array([3, 10])
        cur_spec_norm = self._update()
        reward = self._reward()

        return np.concatenate([cur_spec_norm, self.cur_params_idx, [reward], self.specs_ideal_norm, [-1]], axis=0)

    def step(self, action):
        """

        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """
        a1 = self.action_meaning[int(action // 7)]
        a2 = self.action_meaning[int(action % 7)]
        self.cur_params_idx = self.cur_params_idx + np.array([a1,a2])
        self.cur_params_idx = np.clip(self.cur_params_idx, [0,0], [(len(param_vec)-1) for param_vec in self.params])
        cur_spec_norm = self._update()
        reward = self._reward()
        done = False

        #incentivize reaching goal state
        if (reward > -0.05):
            done = False
            if self.sparse:
                reward = 10
            else:
                reward = 10 + np.sum(self._lookup())
            print('-'*10)
            print('params = ', self.cur_params_idx)
            print('specs:', self.cur_specs)
            print('re:', reward)
            print('-'*10)

        return np.concatenate([cur_spec_norm, self.cur_params_idx, [reward], self.specs_ideal_norm, [action]], axis=0), reward, done, {"params": action}

    def _lookup(self):
        '''
        Calculates relative difference between ideal and current spec
        '''
        rel_specs = []
        for i, spec_tuple in enumerate(self.cur_specs.items()):
            spec_key, spec_val = spec_tuple
            rel_spec = rel_diff(spec_val,self.specs_ideal[i])
            if spec_key == 'ibias':
                rel_spec = rel_spec*(-1)
            rel_specs.append(rel_spec)
        return np.array(rel_specs)

    def _reward(self):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        rel_specs = self._lookup()
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            # if(list(self.cur_specs.keys())[i] == 'ibias'):
            #     rel_spec = rel_spec*-1.0
            if rel_spec < 0:
                reward += rel_spec

        if self.sparse:
            return -1 if reward < 0.05 else 0
        else:
            return reward

    def _update(self):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        params = [self.params[i][self.cur_params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id,params)))]

        #run param vals and simulate
        self.cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val, verbose=True)[0][1].items(), key=lambda k:k[0]))

        return self._lookup() #np.array(list(self.cur_specs.values()))
