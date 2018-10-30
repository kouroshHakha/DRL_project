import os
import IPython
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import framework
import numpy as np
from scipy import interpolate
import random as rnd
import statistics
import random

from collections import OrderedDict
from gym_ckt.spaces.dtup import DiscreteTuple
import yaml
import yaml.constructor

import os
from framework.wrapper.CSAmpClass import CSAmpClass
from gym_ckt.envs.utils import *

class SweepCkt(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = 0.0
    PERF_HIGH = 100.0e9

    framework_path = os.path.abspath(framework.__file__).split("__")
    CIR_YAML = framework_path[0]+"/yaml_files/cs_amp.yaml"

    def __init__(self):
        with open(SweepCkt.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)
        
        # design specs
        specs = yaml_data['target_specs']
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        self.specs_ideal = []  
        self.specs_id = list(specs.keys()) 

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
        self.observation_space = spaces.Box(low=np.full(len(self.specs_id), SweepCkt.PERF_LOW), high=np.full(len(self.specs_id), SweepCkt.PERF_HIGH))

        _ = self.reset()

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
        rand_oidx = random.randint(0,len(list(self.specs.values())[0])-1)
        self.specs_ideal = []
        for spec in list(self.specs.values()):
            self.specs_ideal.append(spec[rand_oidx]) 

        return np.array(list(self.cur_specs.values())), np.array(self.specs_ideal), np.array(self.cur_params_idx) 

    def step(self, action):
      old = self._reward()
      obs = self._update(action)
      new = self._reward()

      if (new > -0.05): 
          done = True 
      else:
          done = False

      #incentivize reaching goal state
      if (new > -0.05):
        print("Actually reached done state")
        rew_del = 1000000#_del = 10000

      return obs, new, done, {"reward": new}

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

        return np.array(list(self.cur_specs.values()))#np.array(self._lookup()) 
     
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
