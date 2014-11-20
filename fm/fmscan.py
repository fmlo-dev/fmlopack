# coding: utf-8

'''
module: fmlopack.fm.fmscan
author: Akio Taniguchi
affill: Institute of Astronomy, the University of Tokyo
mailto: taniguchi_at_ioa.s.u-tokyo.ac.jp
'''

version = '1.0'

# ==============================================================================
# ==============================================================================
import os
import re
import shelve
import shutil
import sys
import tempfile
from datetime import datetime
from decimal import Decimal
from itertools import product
from subprocess import Popen, PIPE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

import fmlopack.pl.statistics as stat


# ==============================================================================
# ==============================================================================
class FmScan(np.ndarray):
    def __new__(cls, scan, tsys, fmrecord, fmstatus):
        scan     = np.asarray(scan).view(cls)
        tsys     = np.asarray(tsys)
        fmrecord = np.asarray(fmrecord).view(np.recarray)
        fmstatus = fmstatus

        # check valid
        fmstatus_all = ['modulated', 'demodulated-obs', 'demodulated-img']
        fmrecord_set = {'CHANFM', 'FREQRANGE', 'INTERVAL'}

        if not fmstatus in fmstatus_all:
            raise FmScanError('invalid fmstatus')
        if not fmrecord_set.issubset(set(fmrecord.dtype.names)):
            raise FmScanError('invalid fmrecord')

        # go to __array_finalize__ .............................................
        scan.tsys     = tsys
        scan.fmrecord = fmrecord
        scan.fmstatus = fmstatus
        return scan

    # OK------------------------------------------------------------------------
    def __array_finalize__(self, scan):
        if scan is None: return
        self.tsys     = getattr(scan, 'tsys', None)
        self.fmrecord = getattr(scan, 'fmrecord', None)
        self.fmstatus = getattr(scan, 'fmstatus', None)

    # OK------------------------------------------------------------------------
    def demodulate(self, target='observed'):
        if target == 'observed':
            fmstatus   = 'demodulated-obs'
            demod_scan = self.__demodulate__(self, self.fmrecord.CHANFM)
        elif target == 'image':
            fmstatus   = 'demodulated-img'
            demod_scan = self.__demodulate__(self, self.fmrecord.CHANFM*(-1))
        else:
            raise FmScanError('invalid target selected')

        return FmScan(demod_scan, self.tsys, self.fmrecord, fmstatus)

    # OK------------------------------------------------------------------------
    def modulate(self):
        scan_wid = len(self.tsys)

        if self.fmstatus == 'demodulated-obs':
            mod_scan = self.__modulate__(self, self.fmrecord.CHANFM, scan_wid)
        elif self.fmstatus == 'demodulated-img':
            mod_scan = self.__modulate__(self, self.fmrecord.CHANFM*(-1), scan_wid)
        else:
            raise FmScanError('invalid target selected')

        return FmScan(mod_scan, self.tsys, self.fmrecord, 'modulated')

    # OK------------------------------------------------------------------------
    def integtime(self, mode='spectrum'):
        ch_fm    = self.fmrecord.CHANFM
        fq_range = self.fmrecord.FREQRANGE
        interval  = self.fmrecord.INTERVAL
        scan_wid = len(self.tsys)

        # check valid ..........................................................
        if self.fmstatus == 'modulated':
            raise FmScanError('this attribute is unavailable in modulated scan')

        # demodulate interval_map ...............................................
        interval_map = np.tile(interval, (scan_wid,1)).T

        if self.fmstatus == 'demodulated-obs':
            integ_map = self.__demodulate__(interval_map, ch_fm)

        elif self.fmstatus == 'demodulated-img':
            integ_map = self.__demodulate__(interval_map, ch_fm*(-1))

        else:
            raise FmScanError('invalid target selected')

        # return map (2d-array) or spectrum (1d-array) .........................
        if mode == 'map': 
            return integ_map

        elif mode == 'spectrum':
            return np.sum(integ_map, axis=0)

        else:
            raise FmScanError('invalid mode selected')

    # OK------------------------------------------------------------------------
    def spectrum(self):
        spec   = np.asarray(self).mean(axis=0)
        tinteg = self.integtime(mode='spectrum')

        return spec / (tinteg/np.max(tinteg))

    # OK------------------------------------------------------------------------
    def noisespectrum(self):
        ch_fm    = self.fmrecord.CHANFM
        fq_range = self.fmrecord.FREQRANGE
        interval = self.fmrecord.INTERVAL
        scan_wid = len(self.tsys)

        # check valid ..........................................................
        if self.fmstatus == 'modulated':
            raise FmScanError('this attribute is unavailable in modulated scan')
        
        if np.unique(fq_range[:,1]-fq_range[:,0]).size > 1:
            raise FmScanError('width of modulated scan must be the same in all time-slices')
        else:
            ch_wid = np.unique(fq_range[:,1]-fq_range[:,0])[0]/(scan_wid-1)

        # make noise level map .................................................
        interval_map = np.tile(interval, (scan_wid,1)).T
        noise_map   = (self.tsys / np.sqrt(interval_map*ch_wid))**(-2)

        if self.fmstatus == 'demodulated-obs':
            noise_map = self.__demodulate__(noise_map, ch_fm)

        elif self.fmstatus == 'demodulated-img':
            noise_map = self.__demodulate__(noise_map, ch_fm*(-1))

        else:
            raise FmScanError('invalid target selected')

        return np.sum(noise_map, axis=0)**(-0.5)

    # OK------------------------------------------------------------------------
    def frequency(self):
        fq_range = self.fmrecord.FREQRANGE

        # check valid ..........................................................
        if self.fmstatus == 'modulated':
            raise FmScanError('this attribute is unavailable in modulated scan')

        nuobs_min = fq_range[:,0].min()
        nuobs_max = fq_range[:,1].max()
        nuobs_num = self.shape[1]

        return np.linspace(nuobs_min, nuobs_max, nuobs_num)


    # OK------------------------------------------------------------------------
    def pca(self, target='clean', fraction=0.990, time_chunk=None, npc_max=None):
        in_scan  = np.asarray(self)
        out_scan = np.zeros_like(in_scan)
        tchunk   = time_chunk or len(in_scan)

        # PCA cleaning .........................................................
        for i in range(len(in_scan)/time_chunk):
            in_scan_i = in_scan[i*tchunk:(i+1)*tchunk]
            p     = stat.PCA(in_scan_i, fraction)
            npc   = p.npc if npc_max is None else min(p.npc, npc_max)
            com_i = np.dot(p.U[:,:npc], np.dot(np.diag(p.d[:npc]), p.Vt[:npc]))

            if target == 'clean':
                out_scan[i*tchunk:(i+1)*tchunk] = in_scan_i - com_i
            elif target == 'common':
                out_scan[i*tchunk:(i+1)*tchunk] = com_i
            else:
                raise FmScanError('invalid target selected')

        return FmScan(out_scan, self.tsys, self.fmrecord, self.fmstatus)

    # OK------------------------------------------------------------------------
    def __demodulate__(self, mod, ch_fm):
        ch_fm = np.asarray(ch_fm)
        mod   = np.asarray(mod)
        empty = np.zeros((mod.shape[0], ch_fm.ptp()))
        demod = np.hstack((mod, empty))

        for i in range(len(demod)):
            demod[i] = np.roll(demod[i], ch_fm[i]-ch_fm.min())

        return demod

    # OK------------------------------------------------------------------------
    def __modulate__(self, demod, ch_fm, wid):
        ch_fm = np.asarray(ch_fm)
        demod = np.asarray(demod)

        for i in range(len(demod)):
            demod[i] = np.roll(demod[i], ch_fm[i]*(-1)+ch_fm.min())

        return demod[:,:wid]

    # OK------------------------------------------------------------------------
    def __idx_fmrecord__(self, *idx):
        idx = idx[0]

        if type(idx) == tuple:
            if type(idx[0]) == int:
                idx_fmrecord = slice(idx[0], idx[0]+1, 1)
            else:
                idx_fmrecord = idx[0]
        elif type(idx) == int:
            idx_fmrecord = slice(idx, idx+1, 1)
        else:
            idx_fmrecord = idx

        return idx_fmrecord

    # OK------------------------------------------------------------------------
    def __getitem__(self, *idx):
        sl_scan     = np.asarray(self).__getitem__(*idx)
        sl_fmrecord = self.fmrecord.__getitem__(self.__idx_fmrecord__(*idx))

        return FmScan(sl_scan, self.tsys, sl_fmrecord, self.fmstatus)

    # OK------------------------------------------------------------------------
    def __getslice__(self, *idx):
        sl_scan     = np.asarray(self).__getslice__(*idx)
        sl_fmrecord = self.fmrecord.__getslice__(*idx)

        return FmScan(sl_scan, self.tsys, sl_fmrecord, self.fmstatus)

    # OK------------------------------------------------------------------------
    def __repr__(self):
        repr_fmt = 'FmScan({}x{}: {})'
        if self.ndim == 0:
            return repr_fmt.format(1, 1, self.fmstatus)
        elif self.ndim <= 1:
            return repr_fmt.format(1, self.shape[0], self.fmstatus)
        else:
            return repr_fmt.format(self.shape[0], self.shape[1], self.fmstatus)

    # -OK------------------------------------------------------------------------
    def __str__(self):
        str_fmt = 'FmScan({}x{}: {})'
        if self.ndim == 0:
            return str_fmt.format(1, 1, self.fmstatus)
        elif self.ndim <= 1:
            return str_fmt.format(1, self.shape[0], self.fmstatus)
        else:
            return str_fmt.format(self.shape[0], self.shape[1], self.fmstatus)

    # OK------------------------------------------------------------------------
    def __reduce__(self):
        obj_state = np.ndarray.__reduce__(self)
        sub_state = (self.tsys, self.fmstatus, self.fmrecord)
        return (obj_state[0], obj_state[1], (obj_state[2], sub_state))

    # OK------------------------------------------------------------------------
    def __setstate__(self, state):
        obj_state, sub_state = state
        np.ndarray.__setstate__(self, obj_state)
        (self.tsys, self.fmstatus, self.fmrecord) = sub_state


# ==============================================================================
# ==============================================================================
def zeros_like(fmscan):
    scan = np.zeros_like(fmscan)
    tsys = fmscan.tsys
    fmrecord = fmscan.fmrecord
    fmstatus = fmscan.fmstatus
    return FmScan(scan, tsys, fmrecord, fmstatus)


# ==============================================================================
# ==============================================================================
class FmScanError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


# ==============================================================================
# ==============================================================================
class PCA:
    def __init__(self, scan, fraction):
        assert 0 <= fraction <= 1
        self.U, self.d, self.Vt = np.linalg.svd(scan, full_matrices=False)

        assert np.all(self.d[:-1] >= self.d[1:])
        self.eigen = self.d**2
        self.sumvariance = np.cumsum(self.eigen)
        self.sumvariance /= self.sumvariance[-1]
        self.npc = np.searchsorted(self.sumvariance, fraction) + 1

        for d in self.d:
            if d > self.d[0] * 1e-6:
                self.dinv = np.array([1/d])
            else:
                self.dinv = np.array([0])