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
from framework.wrapper.TwoStageComplete import TwoStageOpenLoop, TwoStageCommonModeGain, TwoStagePowerSupplyGain, TwoStageTransient
import IPython

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

class TwoStageAmp(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    framework_path = os.path.abspath(framework.__file__).split("__")
    CIR_YAML = framework_path[0]+"/yaml_files/two_stage_full.yaml"

    def __init__(self, sparse=False, multi_goal=False):

        self.sparse = sparse
        with open(TwoStageAmp.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        self.multi_goal = multi_goal
        # design specs
        specs = yaml_data['target_specs']
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        print(self.specs)
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
        ol_dsn_netlist = yaml_data['ol_dsn_netlist']
        cm_dsn_netlist = yaml_data['cm_dsn_netlist']
        ps_dsn_netlist = yaml_data['ps_dsn_netlist']
        tran_dsn_netlist = yaml_data['tran_dsn_netlist'] 

        self.ol_env = TwoStageOpenLoop(design_netlist=ol_dsn_netlist)
        self.cm_env = TwoStageCommonModeGain(design_netlist=cm_dsn_netlist)
        self.ps_env = TwoStagePowerSupplyGain(design_netlist=ps_dsn_netlist)
        self.tran_env = TwoStageTransient(design_netlist=tran_dsn_netlist)
        
        self.action_meaning = [-1,0,2]
        self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array([TwoStageAmp.PERF_LOW]*2*len(self.specs_id)+[-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0]),
            high=np.array([TwoStageAmp.PERF_HIGH]*2*len(self.specs_id)+[5.0,5.0,5.0,5.0,5.0,5.0,5.0]))

        #initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        self.global_g = []
        for spec in list(self.specs.values()):
                self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.global_g = np.array(self.global_g)

        self.action_arr = []
        for a in self.action_meaning:
            for b in self.action_meaning:
                for c in self.action_meaning:
                    for d in self.action_meaning:
                        for e in self.action_meaning:
                            for f in self.action_meaning:
                                for g in self.action_meaning:
                                    self.action_arr.append([a,b,c,d,e,f,g])
        self.inval_params = [[36, 27, 48, 59, 42, 37, 24]]

    def reset(self, z=None, sigma=0.2):

        if self.multi_goal == False:
            self.specs_ideal = self.global_g 
        else: 
            rand_oidx = random.randint(0,self.num_os-1)
            #self.specs_ideal = self.specs_list[rand_oidx]
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[rand_oidx])
            self.specs_ideal = np.array(self.specs_ideal)

        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        self.cur_params_idx = np.array([20, 20, 20, 20, 20, 20, 1])
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)

        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
        return self.ob #np.concatenate([cur_spec_norm, self.cur_params_idx, [reward], self.specs_ideal_norm, [-1]], axis=0)

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
        return self.ob, reward, done, None 

    def unlookup(self, norm_spec, goal_spec):
        spec = np.multiply(norm_spec, goal_spec) + goal_spec
        return spec

    def lookup(self, spec, goal_spec):
        norm_spec = (spec-goal_spec)/goal_spec
        return norm_spec
    
    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''

        rel_specs = self.lookup(spec, goal_spec)
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if (self.specs_id[i] == 'offset_sys_max') or (self.specs_id[i] == 'tset_max') or (self.specs_id[i] == 'ibias_max'):
                rel_spec = rel_spec*-1.0
            if self.specs_id[i] == 'ibias_max':
                rel_spec = rel_spec/10
            if rel_spec < 0:
                reward += rel_spec
        if self.sparse:
            return -1 if reward < -0.05 else 10
        else:
            return reward if reward < -0.05 else 10+np.sum(rel_specs)

    def sign(self,x):
        return 1-(x<=0)

    def get_tset(self, time_arr, vout, vin, fbck, tot_err=0.1, plt=False):
        # since the evaluation of the raw data needs some of the constraints we need to do tset calculation here
        vin_norm = (vin-vin[0])/(vin[-1]-vin[0])
        ref_value = 1/fbck * vin
        y = (vout-vout[0])/(ref_value[-1]-ref_value[0])

        if plt:
            import matplotlib.pyplot as plt
            plt.plot(time_arr, vin_norm/fbck)
            plt.plot(time_arr, y)
            plt.figure()
            plt.plot(time_arr, vout)
            plt.plot(time_arr,vin)


        last_idx = np.where(y < 1.0 - tot_err)[0][-1]
        last_max_vec = np.where(y > 1.0 + tot_err)[0]
        if last_max_vec.size > 0 and last_max_vec[-1] > last_idx:
            last_idx = last_max_vec[-1]
            last_val = 1.0 + tot_err
        else:
            last_val = 1.0 - tot_err

        if last_idx == time_arr.size - 1:
            return time_arr[-1]
        f = interp.InterpolatedUnivariateSpline(time_arr, y - last_val)
        t0 = time_arr[last_idx]
        t1 = time_arr[last_idx + 1]
        
        if self.sign(f(t0)) == self.sign(f(t1)):
            return 1000e-8
        else:
            return sciopt.brentq(f, t0, t1)        

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id,params)))]

        ol_results = self.ol_env.run(param_val, verbose=True)
        cm_results = self.cm_env.run(param_val, verbose=True)
        ps_results = self.ps_env.run(param_val, verbose=True)
        tran_results = self.tran_env.run(param_val, verbose=True)

        ugbw_cur = ol_results[0][0][1]['ugbw']
        gain_cur = ol_results[0][0][1]['gain']
        phm_cur = ol_results[0][0][1]['phm']
        ibias_cur = ol_results[0][0][1]['Ibias']
        
        # common mode gain and cmrr
        cm_gain_cur = cm_results[0][0][1]['cm_gain']
        cmrr_cur = 20*np.log10(gain_cur/cm_gain_cur) # in db
        
        # power supply gain and psrr
        ps_gain_cur = ps_results[0][0][1]['ps_gain']
        psrr_cur = 20*np.log10(gain_cur/ps_gain_cur) # in db

        # transient settling time and offset calculation
        time_arr = tran_results[0][0][1]['time']
        vout = tran_results[0][0][1]['vout']
        vin = tran_results[0][0][1]['vin']

        fbck = 1
        tset_cur =  self.get_tset(time_arr, vout, vin, fbck)
        offset_curr = abs(vout[0]-vin[0]/fbck)
 
        #run param vals and simulate
        cur_specs = np.array([cmrr_cur, gain_cur, ibias_cur, offset_curr, phm_cur, psrr_cur, tset_cur, ugbw_cur])  
        return cur_specs
