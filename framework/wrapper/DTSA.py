import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint

debug = False

import sys
sys.path.append('./')
from framework.wrapper.ngspice_wrapper import NgSpiceWrapper

class DTSAOverdriveRecovery(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        time, vout, vin, ibias = self.parse_output(output_path)

        raw_data = dict(
            time=time,
            vout=vout,
            vin=vin,
            ibias=ibias
        )

        return raw_data

    def parse_output(self, output_path):

        tran_fname = os.path.join(output_path, 'tran.csv')

        if not os.path.isfile(tran_fname):
            print("tran file doesn't exist: %s" % output_path)

        tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=1)
        t = tran_raw_outputs[:, 0]
        vout = tran_raw_outputs[:, 1]
        vin = tran_raw_outputs[:, 3]
        ivdd = tran_raw_outputs[:, 5]

        return t, vout, vin, ivdd

class EvaluationCore(object):

    def __init__(self, cir_yaml):
        import yaml
        with open(cir_yaml, 'r') as f:
            yaml_data = yaml.load(f)

        # specs
        specs = yaml_data['target_specs']
        self.vin_min    = specs['vmin_min']
        self.Tper       = specs['Tper']
        self.vout_min   = specs['vout_min']
        self.tsetup     = specs['tsetup']
        self.cff        = specs['cff']
        self.bias_max   = 1e-3

        num_process = yaml_data['num_process']
        dsn_netlist = yaml_data['dsn_netlist']

        self.env = DTSAOverdriveRecovery(num_process=num_process, design_netlist=dsn_netlist)

        params = yaml_data['params']
        self.param_vec_dict = {}
        for key, value in params.items():
            self.param_vec_dict[key] = np.arange(value[0], value[1], value[2])


    def cost_fun(self, verbose=False, **kwds):
        """

        :param res:
        :param mul:
        :param verbose: if True will print the specification performance of the best individual and file name of
        the netlist
        :return:
        """

        eval_start_time = time.time()
        if verbose:
            print("state_before_rounding:{}".format(kwds))

        param_dict = kwds.copy()
        # snap to the proper grid
        for key, value in param_dict.items():
            # get the last index that is less than the value of that param. it can be the upper bound.
            # if there is not value in the vec that is less than the value of the param:
            # the value should be clipped to the lower bound
            nearest_idx = np.where(self.param_vec_dict[key] <= int(value))[0][-1]
            if nearest_idx != None:
                param_dict[key] = self.param_vec_dict[key][nearest_idx]
            else:
                param_dict[key] = self.param_vec_dict[key][0]


        param_dict['Tper'] = self.Tper
        param_dict['cff'] = self.cff
        param_dict['vi_final'] = self.vin_min
        state = [param_dict]
        results = self.env.run(state, verbose=verbose)

        t = results[0][1]['time']
        vout = results[0][1]['vout']
        vin = results[0][1]['vin']
        ibias = results[0][1]['ibias']

        if verbose:
            import matplotlib.pyplot as plt
            plt.plot(t, vout)
            plt.plot(t, vin)
            plt.vlines(3.5*self.Tper, -1.2, 1.2, colors='r')
            plt.vlines(4.5*self.Tper, -1.2, 1.2, colors='b')
        # power
        iavg = np.mean(ibias)
        # calculate vout at the correct sampling time
        vout_func = interp.interp1d(t, vout, kind='quadratic')
        vsample_prev = vout_func(3.5*self.Tper-self.tsetup)
        vsample = vout_func(4.5*self.Tper-self.tsetup)

        if verbose:
            print('vsample=%f vs. vout_min=%f' %(vsample, self.vout_min))
            print('iavg=%f' %(abs(iavg)))

        cost = 0
        if not (vsample_prev > 0 and abs(vsample_prev) > self.vout_min):
            cost += abs(vsample_prev/self.vout_min - 1.0)
        if not (vsample < 0 and abs(vsample) > self.vout_min):
            cost += abs(-vsample/self.vout_min - 1.0)

        cost += abs(iavg/self.bias_max)/10

        eval_end_time = time.time()
        # print("eval_time    %s sec" %(eval_end_time-eval_start_time))
        return cost

if __name__ == '__main__':
    eval_core = EvaluationCore('./framework/yaml_files/dtsa.yaml')
    cost = eval_core.cost_fun(m1=10,
                              m2=30,
                              m3=40,
                              m4=4,
                              m5=2,
                              m6=1,
                              m7=6,
                              verbose=True)
    print(cost)