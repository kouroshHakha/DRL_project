"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces

import framework
import numpy as np
import random

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
from framework.wrapper.TwoStageClass import TwoStageClass
import IPython
import itertools
import pickle

#way of ordering the way a yaml file is read
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

class TwoStageAmp(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    framework_path = os.path.abspath(framework.__file__).split("__")
    CIR_YAML = framework_path[0]+"/yaml_files/two_stage_opamp.yaml"

    def __init__(self, env_config):
        #print("@@@@@@213123123@@@@@@@@@@@@@@@@@")
        multi_goal = env_config.get("multi_goal",False)
        generalize = env_config.get("generalize",False)
        num_valid = env_config.get("num_valid",50)
        rinokeras_specs = env_config.get("rinokeras_specs",False)
        specs_save = env_config.get("save_specs", False)
        valid = env_config.get("run_valid", False)

        self.env_steps = 0
        with open(TwoStageAmp.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        self.multi_goal = False #multi_goal
        self.generalize = generalize
        self.save_specs = specs_save
        self.valid = valid

        # design specs
        if generalize == False:
            specs = yaml_data['target_specs']
        else:
            specs_range = yaml_data['target_valid_specs']
            specs_range_vals = list(specs_range.values())
            if rinokeras_specs:
                specs = yaml_data['target_specs']
                specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
                arr = np.load(TwoStageAmp.framework_path[0]+"../ploting_sand_box/rinokeras_specs.npy")
                i = 0
                for key,value in specs.items():
                    specs[key] = arr[:,i]
                    i+=1
            else:
                specs_valid = []
                #random.seed(2992)
                for spec in specs_range_vals:
                    if isinstance(spec[0],int):
                        list_val = [random.randint(int(spec[0]),int(spec[1])) for x in range(0,num_valid)]
                    else:
                        list_val = [random.uniform(float(spec[0]),float(spec[1])) for x in range(0,num_valid)]
                    specs_valid.append(tuple(list_val))
                i=0
                for key,value in specs_range.items():
                    specs_range[key] = specs_valid[i]
                    i+=1
                specs_train = yaml_data['target_specs']
                specs_val = []
                for i,valid_arr in enumerate(list(specs_range.values())):
                    specs_val.append(valid_arr+list(specs_train.values())[i])
                specs = specs_train
                i = 0
                for key,value in specs.items():
                    specs[key] = specs_val[i]
                    i+=1
        
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        if specs_save:
            print(self.specs)
            with open("specs_"+str(num_valid)+str(random.randint(1,100000)), 'wb') as f:
                pickle.dump(self.specs, f)
        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1

        self.num_os = len(list(self.specs.values())[0])

        # param array
        params = yaml_data['params']
        self.params = []
        self.params_id = list(params.keys())

        for value in params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)

        #initialize sim environment
        #discrete action space for now, parameter values can move by either -1, 0, or 2
        #observation space only used to get how many there are for RL algorithm, actual range doesnt matter
        dsn_netlist = TwoStageAmp.framework_path[0] + yaml_data['dsn_netlist']
        self.sim_env = TwoStageClass(design_netlist=dsn_netlist)
        #self.action_meaning = [-4,-1,0,1,4]
        #print(len(self.params_id))
        self.action_meaning = [-1,0,2]
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))]*len(self.params_id))
        #self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array([TwoStageAmp.PERF_LOW]*2*len(self.specs_id)+len(self.params_id)*[1]),
            high=np.array([TwoStageAmp.PERF_HIGH]*2*len(self.specs_id)+len(self.params_id)*[1]))

        #initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        #Get the g* (overall design spec) you want to reach
        self.global_g = []
        for spec in list(self.specs.values()):
                self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)
        self.global_g = np.array(yaml_data['normalize'])

        #Initializing action space, works by creating all combos for each parameter
        self.action_arr = list(itertools.product(*([self.action_meaning for i in range(len(self.params_id))])))

        #objective number (used for validation)
        self.obj_idx = 0

    def reset(self):
        #if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os-1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0,self.num_os-1)
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star 
            else:
                idx = random.randint(0,self.num_os-1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)

        #applicable only when you have multiple goals, normalizes everything to some global_g
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        #initialize current parameters
        self.cur_params_idx = np.array([33, 33, 33, 33, 33, 33, 33])
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)

        #observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])

        #count num of env steps
        self.env_steps = self.env_steps + 1
        return self.ob

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """

        #Take action that RL agent returns to change current params
        action = list(np.reshape(np.array(action),(np.array(action).shape[0],)))
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])

#        self.cur_params_idx = self.cur_params_idx + np.array(self.action_arr[int(action)])
        self.cur_params_idx = np.clip(self.cur_params_idx, [0]*len(self.params_id), [(len(param_vec)-1) for param_vec in self.params])

        #Get current specs and normalize
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm  = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)
        done = False

        #incentivize reaching goal state
        if (reward >= 10):
            done = True
            print('-'*10)
            print('params = ', self.cur_params_idx)
            print('specs:', self.cur_specs)
            print('ideal specs:', self.specs_ideal)
            print('re:', reward)
            print('-'*10)

        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        self.env_steps = self.env_steps + 1
        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec-goal_spec)/(goal_spec+spec)
        return norm_spec

    def mit_lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = spec / goal_spec
        return norm_spec

    def mit_reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        rel_specs = self.mit_lookup(spec, goal_spec)
        hard_rewards = []
        opt_rewards = []
        alpha = 1 / 20
        e_0 = 0
        e_1 = len(rel_specs)
        norm = len(rel_specs) +1

        for i,rel_spec in enumerate(rel_specs):
          if(self.specs_id[i] == 'ibias_max'):
            rel_spec = 1 / rel_spec
            rel_spec = rel_spec * alpha
            opt_rewards.append(rel_spec)
          else:
            hard_rewards.append(rel_spec)

        reward = np.sum(hard_rewards)
        opt_reward = np.sum(opt_rewards)
        ret = 0
        if reward > len(hard_rewards):
          ret = opt_reward - norm + e_1
        else:
          ret = reward + alpha * opt_rewards - norm + e_0

        return ret

    def jenny_reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''

        rel_specs = self.lookup(spec, goal_spec)
        print(rel_specs)
        rewards = []
        for i,rel_spec in enumerate(rel_specs):
            if(self.specs_id[i] == 'ibias_max'):
                rel_spec = rel_spec*-1.0
            if rel_spec < 0:
                rewards.append(-rel_spec)
            else:
                rewards.append(0)

        reward = -np.linalg.norm(rewards)
        return reward if reward < -0.05 else 10+np.sum(rel_specs)

    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if(self.specs_id[i] == 'ibias_max'):
                rel_spec = rel_spec*-1.0
            if rel_spec < 0:
                reward += rel_spec
        return reward if reward < -0.02 else 10 #+np.sum(rel_specs)

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id,params)))]

        #run param vals and simulate
        cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val, verbose=True)[0][1].items(), key=lambda k:k[0]))
        cur_specs = np.array(list(cur_specs.values()))
        return cur_specs
