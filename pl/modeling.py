# coding: utf-8

'''
module: fmlopack.pl.modeling
author: Akio Taniguchi
affill: Institute of Astronomy, the University of Tokyo
mailto: taniguchi_at_ioa.s.u-tokyo.ac.jp
'''

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import fmlopack.fm.fmscan as fms


# ==============================================================================
# ==============================================================================
class CutoffModel(object):
    # --------------------------------------------------------------------------
    def __init__(self, fmscan, sn_threshold=10):
        self.nuobs = fmscan.nuobs(useGHz=True)
        self.spec  = fmscan.spectrum('signal')
        self.noise = fmscan.spectrum('noise')
        self.thres = sn_threshold
        self.model, self.resid = self.modeling()

    # --------------------------------------------------------------------------
    def modeling(self):
        model = np.copy(self.spec)
        sn = self.spec / self.noise
        model[np.where(sn < self.thres)] = 0

        return model, self.spec - model


# ==============================================================================
# ==============================================================================
class GaussianModel(object):
    # --------------------------------------------------------------------------
    def __init__(self, fmscan, sn_threshold=5, sub_fraction=0.5,
                iter_max=10000, init_width=2.0/4096, dev=False):
        self.nuobs = fmscan.nuobs(useGHz=True)
        self.spec  = fmscan.spectrum('signal')
        self.noise = fmscan.spectrum('noise')
        self.thres = sn_threshold
        self.niter = iter_max
        self.width = init_width
        self.frac  = sub_fraction
        self.dev   = dev
        self.model, self.resid = self.modeling()

    # --------------------------------------------------------------------------
    def modeling(self):
        model = np.zeros_like(self.spec)
        resid = np.copy(self.spec)

        for n in range(self.niter):
            self.log('dev: {}th iteration...'.format(n+1))

            sn = resid / self.noise
            if np.max(sn) < self.thres:
                self.log('dev: converged at {}th iteration!'.format(n+1))
                return model, resid

            a_peak = resid[np.argmax(sn)]
            f_peak = self.nuobs[np.argmax(sn)]

            def gaussian(x, amp, fwhm):
                func = a_peak * np.exp(-4.0*np.log(2)*((x-f_peak)/fwhm)**2)
                return func

            init_params = [a_peak, self.width]

            try:
                popt, pcov = opt.curve_fit(gaussian, self.nuobs, resid, init_params)
            except:
                break

            model += self.frac * gaussian(self.nuobs, *popt)
            resid -= self.frac * gaussian(self.nuobs, *popt)

        self.log('dev: reached max iteration!')
        return model, resid

    # --------------------------------------------------------------------------
    def plot(self):
        nuobs = self.nuobs
        plt.figure(figsize=(20,10))
        plt.plot(nuobs, self.spec, label='spectrum')
        plt.plot(nuobs, self.model, label='model')
        plt.xlim([nuobs.min(), nuobs.max()])
        plt.xlabel('Observed frequency (GHz)')
        plt.ylabel('Ta* (K)')
        plt.legend()
        plt.grid()
        plt.show()

    # --------------------------------------------------------------------------
    def log(self, message):
        if self.dev: print(message)


# ==============================================================================
# ==============================================================================
class DeconvolutionModel(object):
    # --------------------------------------------------------------------------
    def __init__(self, fmscan, sn_threshold=10, sub_fraction=0.5,
                iter_max=10000, init_width=2.0/4096, init_cutoff=20, dev=False):
        self.nuobs  = fmscan.nuobs(useGHz=True)
        self.spec   = fmscan.spectrum('signal')
        self.noise  = fmscan.spectrum('noise')
        self.thres  = sn_threshold
        self.niter  = iter_max
        self.width  = init_width
        self.cutoff = init_cutoff
        self.frac   = sub_fraction
        self.dev    = dev
        self.model, self.convl, self.resid = self.modeling()

    # --------------------------------------------------------------------------
    def modeling(self):
        model = np.zeros_like(self.spec)
        convl = np.zeros_like(self.spec)
        resid = np.copy(self.spec)

        for n in range(self.niter):
            self.log('dev: {}th iteration...'.format(n+1))

            sn = resid / self.noise
            if np.max(sn) < self.thres:
                self.log('dev: converged at {}th iteration!'.format(n+1))
                return model, convl, resid

            a_peak = resid[np.argmax(sn)]
            f_peak = self.nuobs[np.argmax(sn)]

            def gaussian(x, amp, fwhm):
                func = amp * np.exp(-4.0*np.log(2)*((x-f_peak)/fwhm)**2)
                return func

            def hpf(spec, cutoff):
                ft = np.fft.fft(spec)
                ft[0:cutoff] *= 0.001
                ft[len(spec)-cutoff:len(spec)] *= 0.001
                return np.fft.ifft(ft).real

            def convolution(x, cutoff, amp, fwhm):
                return hpf(gaussian(x, amp, fwhm), cutoff)

            init_params = [self.cutoff, a_peak, self.width]
            popt = self.chi2fit(gaussian, self.nuobs, resid, init_params)

            model += self.frac * gaussian(self.nuobs, *popt[1:])
            convl += self.frac * convolution(self.nuobs, *popt)
            resid -= self.frac * convolution(self.nuobs, *popt)

        return model, convl, resid

    # --------------------------------------------------------------------------
    def chi2fit(self, f, xdata, ydata, p0):
        cutoffs = np.arange(p0[0]-10, p0[0]+10)
        chi2s   = np.zeros(len(cutoffs))
        popts   = np.zeros((len(cutoffs), len(p0)))

        for (i, cutoff) in enumerate(cutoffs):
            def hpf(spec):
                ft = np.fft.fft(spec)
                ft[0:cutoff] *= 0.001
                ft[len(spec)-cutoff:len(spec)] *= 0.001
                return np.fft.ifft(ft).real

            def convolution(x, amp, fwhm):
                return hpf(f(x, amp, fwhm))

            init_params = p0[1:]
            popt, pcov = opt.curve_fit(convolution, xdata, ydata, init_params)

            chi2s[i] = np.sum(((ydata-convolution(xdata, *popt))/self.noise)**2)
            popts[i] = (cutoff, popt[0], popt[1])

        self.log('dev: optimum popt={}'.format(popts[np.argmin(chi2s)]))
        return popts[np.argmin(chi2s)]

    # --------------------------------------------------------------------------
    def plot(self):
        nuobs = self.nuobs
        plt.figure(figsize=(20,10))
        plt.plot(nuobs, self.spec, label='spectrum')
        plt.plot(nuobs, self.convl, label='convolved model')
        plt.plot(nuobs, self.model, label='deconvolved model')
        plt.xlim([nuobs.min(), nuobs.max()])
        plt.xlabel('Observed frequency (GHz)')
        plt.ylabel('Ta* (K)')
        plt.legend()
        plt.grid()
        plt.show()

    # --------------------------------------------------------------------------
    def log(self, message):
        if self.dev: print(message)