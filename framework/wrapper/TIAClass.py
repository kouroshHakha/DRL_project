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
import math
import IPython

import numpy as np
from scipy import interpolate
import random as rnd
debug = True
import subprocess
from framework.wrapper.ngspice_wrapper import NgSpiceWrapper

class TIAOpenLoop(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """
        # use parse output here
        freq, vout,  ibias = self.parse_output(output_path)
        gain = self.find_dc_gain(vout)
        f3db = self.find_f3db(freq, vout)

        spec = dict(
            f3db=f3db,
            gain=gain,
            ibias=ibias
        )

        return spec

    def parse_output(self, output_path):

        output = str(subprocess.check_output("ngspice -v", shell=True))

        if ("26" in output) or ("revision 27" in output):
            ac_fname = os.path.join(output_path, 'ac.csv')
            dc_fname = os.path.join(output_path, 'dc.csv')
            ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
            dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)    
        else: 
            ac_fname = os.path.join(output_path, 'ac.csv.data')
            dc_fname = os.path.join(output_path, 'dc.csv.data')
            ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=0)
            dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=0)

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        freq = ac_raw_outputs[:, 0]
        voutp_real = ac_raw_outputs[:, 1]
        voutp_imag = ac_raw_outputs[:, 2]
        voutn_real = ac_raw_outputs[:, 4]
        voutn_imag = ac_raw_outputs[:, 5]
        voutp = voutp_real + 1j*voutp_imag
        voutn = voutn_real + 1j*voutn_imag
        vout = voutp - voutn
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_f3db(self, freq, vout):
        vout_log = 20*np.log10(np.abs(vout))
        gain_log_3db = 20*np.log10(np.abs(vout[0])) - 3

        diff_arr = vout_log - gain_log_3db
        idx_arr = np.argmax(diff_arr < 0, axis=0)
        freq_log = np.log10(freq)
        freq_log_max = freq_log[idx_arr]

        fun = interp.interp1d(freq_log, diff_arr, kind='cubic', copy=False, assume_sorted=True)
        f3db = 10.0 ** (sciopt.brentq(fun, freq_log[0], freq_log_max))
        rtia = np.abs(vout[0])
        return f3db 

class TIATransient(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        time, vout, iin = self.parse_output(output_path)

        spec = dict(
            time=time,
            vout=vout,
            iin=iin
        )

        return spec

    def parse_output(self, output_path):

        output = str(subprocess.check_output("ngspice -v", shell=True))

        if "26" in output or ("revision 27" in output):
            tran_fname = os.path.join(output_path, 'tran.csv')
            tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=1)
        else: 
            tran_fname = os.path.join(output_path, 'tran.csv.data')
            tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=0)

        if not os.path.isfile(tran_fname):
            print("tran file doesn't exist: %s" % output_path)

        time =  tran_raw_outputs[:, 0]
        voutp =  tran_raw_outputs[:, 1]
        voutn = tran_raw_outputs[:, 3]
        vout = voutp - voutn
        iin = tran_raw_outputs[:, 5]

        return time, vout, iin

    @classmethod
    def get_tset(self, tran_results, ol_results, plt, tot_err):
        # since the evaluation of the raw data needs some of the constraints we need to do tset calculation here

        gain = ol_results[0][1]['gain']
        t = tran_results[0][1]['time']
        vout = tran_results[0][1]['vout']
        vin = tran_results[0][1]['iin']

        y = np.abs((vout - vout[0])/gain/(vin[-1] - vin[0]))

        if plt:
            import matplotlib.pyplot as plt
            plt.plot(t, [1+tot_err]*len(t), 'r')
            plt.plot(t, [1-tot_err]*len(t), 'r')
            plt.plot(t, y)
            plt.savefig('tset_debug.png', dpi=200)

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

    def create_design_and_simulate(self, state, verbose=False):
        #if debug:
        #    print('state', state)
        #    print('verbose', verbose)
        design_folder, fpath = self.create_design(state)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)
        return [(state, specs, info)]

class TIANoise(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        inoise_total = self.parse_output(output_path)

        spec = dict(
            inoise_total=inoise_total
        )

        return spec

    def parse_output(self, output_path):

        output = str(subprocess.check_output("ngspice -v", shell=True))

        if "26" in output or ("revision 27" in output):
            noise_fname = os.path.join(output_path, 'noise.csv')
            noise_raw_outputs = np.genfromtxt(noise_fname, skip_header=1)
        else: 
            noise_fname = os.path.join(output_path, 'noise.csv.data')
            noise_raw_outputs = np.genfromtxt(noise_fname, skip_header=0)

        if not os.path.isfile(noise_fname):
            print("noise file doesn't exist: %s" % output_path)

        inoise_total = noise_raw_outputs[1]

        return math.sqrt(inoise_total)


if __name__ == '__main__':
    # test each design manager class
    ol_dsn_netlist = '/home/ksettaluri6/DRL_project/framework/netlist/TIA_full/TIA_ol.cir'
    tran_dsn_netlist = '/home/ksettaluri6/DRL_project/framework/netlist/TIA_full/TIA_tran.cir'
    noise_dsn_netlist = '/home/ksettaluri6/DRL_project/framework/netlist/TIA_full/TIA_noise.cir'

    ol_env = TIAOpenLoop(design_netlist=ol_dsn_netlist)
    tran_env = TIATransient(design_netlist=tran_dsn_netlist)
    noise_env = TIANoise(design_netlist=noise_dsn_netlist)

    #comprehensive list of range of specs
    nsers = [2,4,6,8]
    npars = 1
    wp1s = np.array([2.0,4.0,6.0,8.0])*1.0e-6
    wn1s = np.array([2.0,4.0,6.0,8.0])*1.0e-6
    mp1s = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
    mn1s = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]

    r_unit = 5642
    min_gain = 1000000 
    max_gain = 0
    min_f3db = 1000000000
    max_f3db = 0
    min_cur = 10000
    max_cur = 0
    for nser in nsers:
        for wp1 in wp1s:
            for wn1 in wn1s:
                for mp1 in mp1s:
                    for mn1 in mn1s:
                        # example of running it for one example point and getting back the data
                        rfb = nser*r_unit
                        state_list = [{'mp1': mp1,
                                       'mn1': mn1,
                                       'rfb': rfb,
                                       'wn1': wn1,
                                       'wp1': wp1,
                                       }]

                        ol_results = ol_env.create_design_and_simulate(state_list, verbose=True)
                        tran_results = tran_env.create_design_and_simulate(state_list, verbose=True)

                        tset = tran_env.get_tset(tran_results, ol_results, plt=False, tot_err=0.1)
                        gain = ol_results[0][1]['gain']/2
                        f3db = ol_results[0][1]['f3db']
                        if gain > max_gain:
                            max_gain = gain
                        elif gain < min_gain:
                            min_gain = gain
                        if f3db > max_f3db:
                            max_f3db = f3db
                        elif f3db < min_f3db:
                            min_f3db = f3db
                        a = np.array((min_gain,max_gain,min_f3db,max_f3db,min_cur,max_cur))
                        print('gain max: '+str(gain))
                        print('bw max: '+str(f3db))
                        print('tset max: '+str(tset)) 
                        np.save('sweep_metrics.npy',a)
