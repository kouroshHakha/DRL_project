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

import IPython

import numpy as np
from scipy import interpolate
import random as rnd
debug = True

from framework.wrapper.ngspice_wrapper import NgSpiceWrapper

class CSAmpClass(NgSpiceWrapper):

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

        spec = dict(
            ugbw=ugbw,
            gain=gain,
            ibias=ibias
        )

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ac.csv.data')
        dc_fname = os.path.join(output_path, 'dc.csv.data')

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=0)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=0)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        if ac_raw_outputs.shape[1] > 2:
            vout_imag = ac_raw_outputs[:, 2]
            vout = vout_real + 1j*vout_imag
        else:
            vout = vout_real
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        if gain[0] > 1:
            return self._get_best_crossing(freq, gain, val=1)
        else: return 0

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
        return 0

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


