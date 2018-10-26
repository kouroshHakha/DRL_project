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

class TwoStageOpenLoop(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout,  ibias = self.parse_output(output_path)
        gain = self.find_dc_gain(vout)
        ugbw = self.find_ugbw(freq, vout)
        phm = self.find_phm(freq, vout)


        spec = dict(
            ugbw=ugbw,
            gain=gain,
            phm=phm,
            Ibias=ibias
        )

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ac.csv')
        dc_fname = os.path.join(output_path, 'dc.csv')

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        return self._get_best_crossing(freq, gain, val=1)

    def find_phm(self, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw = self._get_best_crossing(freq, gain, val=1)
        if phase[0] <= 0:
            if phase_fun(ugbw) > 0:
                return -180+phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
        else:
            print ('stuck in else statement')


    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop

class TwoStageCommonModeGain(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout = self.parse_output(output_path)
        gain = self.find_dc_gain(vout)


        spec = dict(
            cm_gain=gain,
        )

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'cm.csv')

        if not os.path.isfile(ac_fname):
            print("cm file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag

        return freq, vout

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

class TwoStagePowerSupplyGain(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout = self.parse_output(output_path)
        gain = self.find_dc_gain(vout)


        spec = dict(
            ps_gain=gain,
        )

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ps.csv')

        if not os.path.isfile(ac_fname):
            print("ps file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag

        return freq, vout

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

class TwoStageTransient(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        time, vout, vin = self.parse_output(output_path)
        # vout_norm = vout/vout[-1]
        # settling_time = self.get_tset(time, vout_norm, tot_err=0.01)


        spec = dict(
            time=time,
            vout=vout,
            vin=vin
        )

        return spec

    def parse_output(self, output_path):

        tran_fname = os.path.join(output_path, 'tran.csv')

        if not os.path.isfile(tran_fname):
            print("tran file doesn't exist: %s" % output_path)

        tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=1)
        time =  tran_raw_outputs[:, 0]
        vout =  tran_raw_outputs[:, 1]
        vin =   tran_raw_outputs[:, 3]

        return time, vout, vin

class EvaluationCore(object):

    def __init__(self, cir_yaml):
        import yaml
        with open(cir_yaml, 'r') as f:
            yaml_data = yaml.load(f)

        # specs
        self.specs = yaml_data['target_specs']
        self.ugbw_min   = self.specs['ugbw_min']
        self.gain_min   = self.specs['gain_min']
        self.phm_min    = self.specs['phm_min']
        self.tset_max   = self.specs['tset_max']
        self.fdbck      = self.specs['feedback_factor']
        self.tot_err    = self.specs['tot_err']
        self.psrr_min   = self.specs['psrr_min']
        self.cmrr_min   = self.specs['cmrr_min']
        self.offset_max = self.specs['offset_sys_max']
        self.bias_max   = self.specs['bias_max']

        num_process = yaml_data['num_process']
        ol_dsn_netlist = yaml_data['ol_dsn_netlist']
        cm_dsn_netlist = yaml_data['cm_dsn_netlist']
        ps_dsn_netlist = yaml_data['ps_dsn_netlist']
        tran_dsn_netlist = yaml_data['tran_dsn_netlist']

        self.ol_env = TwoStageOpenLoop(num_process=num_process, design_netlist=ol_dsn_netlist)
        self.cm_env = TwoStageCommonModeGain(num_process=num_process, design_netlist=cm_dsn_netlist)
        self.ps_env = TwoStagePowerSupplyGain(num_process=num_process, design_netlist=ps_dsn_netlist)
        self.tran_env = TwoStageTransient(num_process=num_process, design_netlist=tran_dsn_netlist)

        self.params = yaml_data['params']
        self.params_vec = []
        for value in self.params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params_vec.append(param_vec)
        self.mp1_vec = np.arange(self.params['mp1'][0], self.params['mp1'][1], self.params['mp1'][2])
        self.mn1_vec = np.arange(self.params['mn1'][0], self.params['mn1'][1], self.params['mn1'][2])
        self.mn3_vec = np.arange(self.params['mn3'][0], self.params['mn3'][1], self.params['mn3'][2])
        self.mn4_vec = np.arange(self.params['mn4'][0], self.params['mn4'][1], self.params['mn4'][2])
        self.mp3_vec = np.arange(self.params['mp3'][0], self.params['mp3'][1], self.params['mp3'][2])
        self.mn5_vec = np.arange(self.params['mn5'][0], self.params['mn5'][1], self.params['mn5'][2])
        self.cc_vec = np.arange(self.params['cc'][0], self.params['cc'][1], self.params['cc'][2])


    def cost_fun(self, design, verbose=False):
        """

        :param design: a list containing relative indices according to yaml file
        :param verbose:
        :return:
        """

        eval_start_time = time.time()
        if verbose:
            print("state_before_rounding:{}".format(design))

        state_dict = dict()
        for i, key in enumerate(self.params.keys()):
            state_dict[key] = self.params_vec[i][design[i]]
        state = [state_dict]

        ol_results = self.ol_env.run(state, verbose=verbose)
        cm_results = self.cm_env.run(state, verbose=verbose)
        ps_results = self.ps_env.run(state, verbose=verbose)
        tran_results = self.tran_env.run(state, verbose=verbose)

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

        tset_cur = EvaluationCore.get_tset(t, vout, vin, self.fdbck, tot_err=self.tot_err, plt=verbose)
        offset_curr = abs(vout[0]-vin[0]/self.fdbck)

        if verbose:
            print('gain = %f vs. gain_min = %f' %(gain_cur, self.gain_min))
            print('ugbw = %f vs. ugbw_min = %f' %(ugbw_cur, self.ugbw_min))
            print('phm = %f vs. phm_min = %f' %(phm_cur, self.phm_min))
            print('tset = %.2f ns vs. tset_max = %.2f ns' %(tset_cur*1e9, self.tset_max*1e9))
            print('cmrr = %.2f db vs. cmrr_min = %.2f db' %(cmrr_cur, self.cmrr_min))
            print('psrr = %.2f vs. psrr_min = %.2f' %(psrr_cur, self.psrr_min))
            print('offset = %.2f mv vs. offset_max = %.2f mv' %(offset_curr*1e3, self.offset_max*1e3))
            print('Ibias = %f' %(ibias_cur))

        cost = 0
        if ugbw_cur < self.ugbw_min:
            cost += abs(ugbw_cur/self.ugbw_min - 1.0)
        if gain_cur < self.gain_min:
            cost += abs(gain_cur/self.gain_min - 1.0)
        if phm_cur < self.phm_min:
            cost += abs(phm_cur/self.phm_min - 1.0)
        if tset_cur > self.tset_max:
            cost += abs(tset_cur/self.tset_max - 1.0)
        if cmrr_cur < self.cmrr_min:
            cost += abs(cmrr_cur/self.cmrr_min - 1.0)
        if psrr_cur < self.psrr_min:
            cost += abs(psrr_cur/self.psrr_min - 1.0)
        if offset_curr > self.offset_max:
            cost += abs(offset_curr/self.offset_max - 1.0)

        cost += abs(ibias_cur/self.bias_max)/10

        eval_end_time = time.time()
        # print("eval_time    %s sec" %(eval_end_time-eval_start_time))
        # updated the output because we want to have access to what each individual spec is
        # return cost
        return cost, ugbw_cur, gain_cur, phm_cur, tset_cur, psrr_cur, cmrr_cur, offset_curr, ibias_cur

    @classmethod
    def get_tset(cls, t, vout, vin, fbck, tot_err=0.1, plt=False):

        # since the evaluation of the raw data needs some of the constraints we need to do tset calculation here
        vin_norm = (vin-vin[0])/(vin[-1]-vin[0])
        ref_value = 1/fbck * vin
        y = (vout-vout[0])/(ref_value[-1]-ref_value[0])

        if plt:
            import matplotlib.pyplot as plt
            plt.plot(t, vin_norm/fbck)
            plt.plot(t, y)
            plt.figure()
            plt.plot(t, vout)
            plt.plot(t,vin)

        last_idx = np.where(y < 1.0 - tot_err)[0][-1]
        last_max_vec = np.where(y > 1.0 + tot_err)[0]
        if last_max_vec.size > 0 and last_max_vec[-1] > last_idx:
            last_idx = last_max_vec[-1]
            last_val = 1.0 + tot_err
        else:
            last_val = 1.0 - tot_err

        if last_idx == t.size - 1:
            return t[-1]
        f = interp.InterpolatedUnivariateSpline(t, y - last_val)
        t0 = t[last_idx]
        t1 = t[last_idx + 1]
        return sciopt.brentq(f, t0, t1)


if __name__ == '__main__':

    # test each design manager class
    num_process = 1
    ol_dsn_netlist = './framework/netlist/two_stage_full/two_stage_ol.cir'
    cm_dsn_netlist = './framework/netlist/two_stage_full/two_stage_cm.cir'
    ps_dsn_netlist = './framework/netlist/two_stage_full/two_stage_ps.cir'
    tran_dsn_netlist = './framework/netlist/two_stage_full/two_stage_tran.cir'

    ol_env = TwoStageOpenLoop(num_process=num_process, design_netlist=ol_dsn_netlist)
    cm_env = TwoStageCommonModeGain(num_process=num_process, design_netlist=cm_dsn_netlist)
    ps_env = TwoStagePowerSupplyGain(num_process=num_process, design_netlist=ps_dsn_netlist)
    tran_env = TwoStageTransient(num_process=num_process, design_netlist=tran_dsn_netlist)

    # example of running it for one example point and getting back the data
    state_list = [{'mp1': 18,
                   'mn1': 38,
                   'mn3': 35,
                   'mp3': 158,
                   'mn5': 98,
                   'mn4': 51,
                   'cc': 3.1e-12
                   }]

    ol_results = ol_env.run(state_list, verbose=True)
    cm_results = cm_env.run(state_list, verbose=True)
    ps_results = ps_env.run(state_list, verbose=True)
    tran_results = tran_env.run(state_list, verbose=True)

    t = tran_results[0][1]['time']
    vout = tran_results[0][1]['vout']
    vin = tran_results[0][1]['vin']
    fbck = 1

    tset = EvaluationCore.get_tset(t, vout, vin, fbck, tot_err=0.1)


    if debug:
        print(ol_results[0][1])
        print(cm_results[0][1])
        print(ps_results[0][1])
        print("tset: %.2f ns" %(tset*1e9))

    # test evaluation core of the opamp
    eval_core = EvaluationCore('./framework/yaml_files/two_stage_full.yaml')
    # cost = eval_core.cost_fun(mp1=4, #mp1=18,
    #                           mn1=19, #mn1=38,
    #                           mn3=7, #mn3=35,
    #                           mp3=62, #mp3=24,
    #                           mn5=49, #mn5=24,
    #                           mn4=68, #mn4=51,
    #                           cc=6,  #cc=3.1e-12,
    #                           verbose=True)
    # print(len(eval_core.params_vec[1]))
    cost = eval_core.cost_fun([4,99,62,7,68,49,6], verbose=True)
    print(cost)