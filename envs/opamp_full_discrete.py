"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces

import framework
import numpy as np
import random
import scipy.interpolate as interp
import scipy.optimize as sciopt

from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
from framework.wrapper.TwoStageComplete import TwoStageOpenLoop,TwoStageCommonModeGain,TwoStagePowerSupplyGain,TwoStageTransient
import IPython
import itertools

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
    CIR_YAML = framework_path[0]+"/yaml_files/two_stage_full.yaml"

    #def __init__(self, multi_goal=False, generalize=False, num_valid=10):
    def __init__(self, env_config):
        multi_goal = env_config.get("multi_goal",False)
        generalize = env_config.get("generalize",False)
        num_valid = env_config.get("num_valid",10)

        with open(TwoStageAmp.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        self.multi_goal = multi_goal
        self.generalize = generalize

        # design specs
        if generalize == False:
            specs = yaml_data['target_specs']
        else:
            specs_range = yaml_data['target_valid_specs']
            specs_range_vals = list(specs_range.values())
                 
            specs_valid = []
            for spec in specs_range_vals:
                if isinstance(spec[0],int):
                    list_val = [random.randint(int(spec[0]),int(spec[1])) for x in range(0,num_valid)]
                else:
                    list_val = [random.uniform(spec[0],spec[1]) for x in range(0,num_valid)]
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
        ol_dsn_netlist = TwoStageAmp.framework_path[0] + yaml_data['ol_dsn_netlist']
        cm_dsn_netlist = TwoStageAmp.framework_path[0] + yaml_data['cm_dsn_netlist']
        ps_dsn_netlist = TwoStageAmp.framework_path[0] + yaml_data['ps_dsn_netlist']
        tran_dsn_netlist = TwoStageAmp.framework_path[0] + yaml_data['tran_dsn_netlist'] 

        self.ol_env = TwoStageOpenLoop(design_netlist=ol_dsn_netlist)
        self.cm_env = TwoStageCommonModeGain(design_netlist=cm_dsn_netlist)
        self.ps_env = TwoStagePowerSupplyGain(design_netlist=ps_dsn_netlist)
        self.tran_env = TwoStageTransient(design_netlist=tran_dsn_netlist)
        
        self.action_meaning = [-1,0,2]
        self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array([TwoStageAmp.PERF_LOW]*2*len(self.specs_id)+len(self.params_id)*[1]),
            high=np.array([TwoStageAmp.PERF_HIGH]*2*len(self.specs_id)+len(self.params_id)*[1]))

        #initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        self.global_g = []
        for spec in list(self.specs.values()):
                self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.global_g = np.array(self.global_g)

        #Initializing action space, works by creating all combos for each parameter
        self.action_arr = list(itertools.product(*([self.action_meaning for i in range(len(self.params_id))])))
        self.inval_params = [[36, 27, 48, 59, 42, 37, 24]]
        
        #objective number (used for validation)
        self.obj_idx = 0
        #print("self.global_g: {}".format(self.global_g))

    def reset(self):
        #if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.obj_idx > self.num_os-1:
                self.obj_idx = 0
            idx = self.obj_idx
            self.obj_idx += 1
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.global_g 
            else:
                idx = random.randint(0,self.num_os-1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)

        self.fdbck = self.specs_ideal[2]
        self.tot_err = self.specs_ideal[7]

        #applicable only when you have multiple goals, normalizes everything to some global_g
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        #initialize current parameters
        self.cur_params_idx = np.array([20, 20, 20, 20, 20, 20, 1])
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)

        #observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        #print("self.specs_ideal: {}".format(self.specs_ideal))
        #print("self.global_g: {}".format(self.global_g))
        return self.ob
 
    def step(self, action):
        """

        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """
        self.cur_params_idx = self.cur_params_idx + np.array(self.action_arr[int(action)])
        self.cur_params_idx = np.clip(self.cur_params_idx, [0,0,0,0,0,0,0], [(len(param_vec)-1) for param_vec in self.params])

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
        return self.ob, reward, done, {} 

    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec-goal_spec)/(goal_spec+spec)
        return norm_spec
    
    def jenny_rew_lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec-goal_spec)/(goal_spec)
        return norm_spec

    def unlookup(norm_spec, goal_spec):
        spec = -1*np.multiply((norm_spec+1), goal_spec)/(norm_spec-1) 
        return spec

    
    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''

        rel_specs = self.lookup(spec, goal_spec)
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if (self.specs_id[i] == 'offset_sys_max') or (self.specs_id[i] == 'tset_max') or (self.specs_id[i] == 'bias_max'):
                rel_spec = rel_spec*-1.0
            if self.specs_id[i] == 'bias_max':
                rel_spec = rel_spec/10
            if rel_spec < 0:
                reward += rel_spec
            #print('rel_spec:', rel_spec)
            #print('re:', reward)
        return reward if reward < -0.05 else 10+np.sum(rel_specs)

    def jenny_reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''

        rel_specs = self.lookup(spec, goal_spec)
        rewards = []
        for i,rel_spec in enumerate(rel_specs):
            #if(self.specs_id[i] == 'ibias_max'):
            if (self.specs_id[i] == 'offset_sys_max') or (self.specs_id[i] == 'tset_max') or (self.specs_id[i] == 'bias_max'):
                rel_spec = rel_spec*-1.0
            if self.specs_id[i] == 'bias_max':
                rel_spec = rel_spec/10
            if rel_spec < 0:
                rewards.append(-rel_spec)
            else:
                rewards.append(0)

        reward = -np.linalg.norm(rewards)
        return reward if reward < -0.05 else 10+np.sum(rel_specs)

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id,params)))]

        ol_results = self.ol_env.create_design_and_simulate(param_val, verbose=True)
        cm_results = self.cm_env.create_design_and_simulate(param_val, verbose=True)
        ps_results = self.ps_env.create_design_and_simulate(param_val, verbose=True)
        tran_results = self.tran_env.create_design_and_simulate(param_val, verbose=True)

        ugbw_cur = ol_results[0][1]['ugbw']
        gain_cur = ol_results[0][1]['gain']
        phm_cur = ol_results[0][1]['phm']
        ibias_cur = ol_results[0][1]['Ibias']
        
        # common mode gain and cmrr
        cm_gain_cur = cm_results[0][1]['cm_gain']
        cmrr_cur = 20*np.log10(gain_cur/cm_gain_cur) # in db
        
        # power supply gain and psrr
        ps_gain_cur = ps_results[0][1]['ps_gain']
        psrr_cur = 20*np.log10(gain_cur/ps_gain_cur) # in db

        # transient settling time and offset calculation
        t = tran_results[0][1]['time']
        vout = tran_results[0][1]['vout']
        vin = tran_results[0][1]['vin']

        tset_cur =  self.tran_env.get_tset(t, vout, vin, self.fdbck, tot_err=self.tot_err)
        offset_curr = abs(vout[0]-vin[0]/self.fdbck)
 
        #run param vals and simulate
        cur_specs = np.array([ibias_cur, cmrr_cur, self.fdbck, gain_cur, offset_curr, phm_cur, psrr_cur, self.tot_err, tset_cur, ugbw_cur])  
        return cur_specs
