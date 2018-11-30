"""
The old version of CKT-env, just copied it here so I can modify stuff faster and it's more convenient
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

    PERF_LOW = 0.0
    PERF_HIGH = 100.0e9

    framework_path = os.path.abspath(framework.__file__).split("__")
    CIR_YAML = framework_path[0]+"/yaml_files/cs_amp.yaml"

    def __init__(self):
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

        #initialize current param/spec observations
        self.cur_specs = []
        self.cur_params_idx = []

        #initialize sim environment
        dsn_netlist = yaml_data['dsn_netlist']
        self.sim_env = CSAmpClass(design_netlist=dsn_netlist)
        self.action_space = spaces.Box(low=np.array(np.zeros(len(self.params_id))), high=np.ones(len(self.params_id)), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.full(len(self.specs_id) + self.num_os, CSAmp.PERF_LOW),
                                            high=np.full(len(self.specs_id) + self.num_os, CSAmp.PERF_HIGH))

        # _ = self.reset()

    def reset(self):
        done = False

        #randomly select sizing values (indices) 
        self.cur_params_idx = []       
        reset_vals = []
        for j in range(len(self.params)):
            idx = random.randint(0,len(self.params[j])-1)
            self.cur_params_idx.append(idx/(len(self.params[j])-1))
            reset_vals.append(self.params[j][idx])

        #normalize between 0 and 1
         
        #set up current state/get specs/initialize counter
        param_val = [OrderedDict(list(zip(self.params_id,reset_vals)))]
        self.cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val, verbose=True)[0][1].items(), key=lambda k:k[0]))


        #randomly select o*
        rand_oidx = random.randint(0,self.num_os-1)
        self.task_id = one_hot(rand_oidx, self.num_os)
        self.specs_ideal = []
        for spec in list(self.specs.values()):
            self.specs_ideal.append(spec[rand_oidx])

        return np.concatenate([self._lookup(), self.task_id], axis=0),\
               np.zeros(self.observation_space.shape[0]) ,\
               np.array(self.cur_params_idx) # np.array(list(self.cur_specs.values())),  np.array(self.specs_ideal)

    def step(self, action):
      old = self._reward()
      obs = self._update(action)
      new = self._reward()

      if (new > -0.05): 
          done = False
      else:
          done = False

      #incentivize reaching goal state
      if (new > -0.05):
        print("GOAL MET")
        new = 10

      return np.concatenate([obs, self.task_id], axis=0), new, done, {"reward": new}

    def _lookup(self):
        '''
        Calculates relative difference between ideal and current spec
        '''
        rel_specs = []
        for i,spec in enumerate(self.cur_specs.values()):
            rel_specs.append(rel_diff(spec,self.specs_ideal[i]))
        return np.array(rel_specs)

    def _reward(self):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        rel_specs = self._lookup()
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if(list(self.cur_specs.keys())[i] == 'ibias'):
                rel_spec = rel_spec*-1.0
            if rel_spec < 0:
                reward += rel_spec
        return reward 

    def _update(self, action):
        params = []
        for i,a in enumerate(action):
            if a > 1:
                a = 1.0
            elif a < 0:
                a = 0.0
            self.cur_params_idx[i] = int((len(self.params[i])-1)*a)
            params.append(self.params[i][self.cur_params_idx[i]])
        param_val = [OrderedDict(list(zip(self.params_id,params)))]

        #run param vals and simulate
        self.cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val, verbose=True)[0][1].items(), key=lambda k:k[0]))

        return self._lookup() #np.array(list(self.cur_specs.values()))
     
    # additional/non-standard methods
    def configure(self, d):
      """A method to make this class reusable. Currently
         OpenAI Gym cannot pass parameters to an environment
         constructor. Configure should be called right
         after construction.
      
         @param d - a dictionary of parameter initializations
      """
      self.min_bw = d["bw"]
      self.min_gain = d["gain"]
      self.max_bias = d["bias"]
      self.min_phm = d["phm"]
