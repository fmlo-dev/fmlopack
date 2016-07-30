# coding: utf-8

'''
module: fmlopack.pl.otf
author: Akio Taniguchi
affill: Institute of Astronomy, the University of Tokyo
mailto: taniguchi_at_ioa.s.u-tokyo.ac.jp
'''

import numpy as np
import pyfits as pf
import scipy.special as sp
import scipy.interpolate as ip
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import SpanSelector
from scipy.ndimage.interpolation import map_coordinates
import fmlopack.fm.fmscan as fms


# ==============================================================================
# ==============================================================================
class MakeCube(object):
    # --------------------------------------------------------------------------
    def __init__(self, fmscan, x_regrid, y_regrid, gcf='bessel_gauss'):
        # input fmscan (cleaned and demodulated)
        self.x_grid, self.x_regrid = fmscan.fmrecord.RADEC.T[0], x_regrid
        self.y_grid, self.y_regrid = fmscan.fmrecord.RADEC.T[1], y_regrid
        self.z_grid, self.z_regrid = fmscan.nuobs(useGHz=False), fmscan.nuobs(useGHz=False)
        self.s_grid   = self.gridsize(self.x_grid, self.y_grid)
        self.s_regrid = self.gridsize(self.x_regrid, self.y_regrid)
        self.factor   = (self.s_regrid/self.s_grid)**2
        self.gcf = gcf

        # make cube and modeled fmscan
        self.fms_in  = fmscan
        self.noise   = self.noisemap()
        self.cube    = self.regrid()
        self.fms_out = self.modeling()

    # --------------------------------------------------------------------------
    def regrid_freezed(self):
        print('regrid...')
        shape = (len(self.z_regrid), len(self.y_regrid), len(self.x_regrid))
        scan = np.asarray(self.fms_in)
        cube = np.zeros(shape)

        for (y,x) in product(*map(range, shape[1:])):
            idxs, rads = self.is_search_radius(y,x)
            for (idx,rad) in zip(idxs, rads):
                cube[:,y,x] += scan[idx] * getattr(self, self.gcf)(rad)

        return cube / self.factor

    # --------------------------------------------------------------------------
    def regrid(self):
        print('regrid...')
        shape = (len(self.z_regrid), len(self.y_regrid), len(self.x_regrid))
        scan = np.asarray(self.fms_in)
        cube = np.zeros(shape)

        for (y,x) in product(*map(range, shape[1:])):
            idxs, rads = self.is_search_radius(y,x)
            weightsum  = 0

            for (idx,rad) in zip(idxs, rads):
                cube[:,y,x] += getattr(self, self.gcf)(rad) * scan[idx]
                weightsum   += getattr(self, self.gcf)(rad)

            cube[:,y,x] /= weightsum

        return cube

    # --------------------------------------------------------------------------
    def regrid_freezed2(self):
        print('regrid...')
        shape = (len(self.z_regrid), len(self.y_regrid), len(self.x_regrid))
        scan = np.asarray(self.fms_in)
        cube = np.zeros(shape)

        for (y,x) in product(*map(range, shape[1:])):
            idxs  = self.is_search_box(y,x)
            if len(idxs) == 0:
                cube[:,y,x] = 0
            else:
                cube[:,y,x] = np.mean(scan[idxs], axis=0)

        return cube

    # --------------------------------------------------------------------------
    def modeling(self):
        print('make model...')
        model  = fms.zeros_like(self.fms_in)
        x_ip   = ip.interp1d(self.x_regrid, range(len(self.x_regrid)))
        y_ip   = ip.interp1d(self.y_regrid, range(len(self.y_regrid)))
        x_idxs = x_ip(self.x_grid)
        y_idxs = y_ip(self.y_grid)

        for i in range(model.shape[1]):
            model[:,i] = map_coordinates(self.cube[i], (y_idxs, x_idxs))

        return model

    # --------------------------------------------------------------------------
    def is_search_radius(self, y, x, search_radius=3.0):
        '''
        memo:
        - idxs: list of indices
        - rads: list of radius (in units of pixel)
        '''
        y_delt = self.y_grid-self.y_regrid[y]
        x_delt = self.x_grid-self.x_regrid[x]
        rads = np.sqrt(y_delt**2 +x_delt**2)/self.s_regrid
        idxs = np.where(rads<search_radius)[0]
        return idxs, rads[idxs]

    # --------------------------------------------------------------------------
    def is_search_box(self, y, x):
        y_delt = (self.y_grid-self.y_regrid[y])/self.s_regrid
        x_delt = (self.x_grid-self.x_regrid[x])/self.s_regrid
        idxs   = np.where((np.abs(x_delt)<0.5) & (np.abs(y_delt)<0.5))[0]

        return idxs

    # --------------------------------------------------------------------------
    def gridsize(self, x_axis, y_axis, cutoff=1.0/3600):
        x_diff = np.abs(np.diff(x_axis))
        y_diff = np.abs(np.diff(y_axis))
        x_size = np.mean(x_diff[x_diff>cutoff])
        y_size = np.mean(y_diff[y_diff>cutoff])
        return np.sqrt(x_size*y_size)

    # --------------------------------------------------------------------------
    def noisemap(self):
        tsys = self.fms_in.tsys
        interval  = np.tile(self.fms_in.fmrecord.INTERVAL, (len(tsys),1))
        chan_wid  = np.diff(self.fms_in.fmrecord.FREQRANGE)[:,0]/(len(tsys)-1)

        return tsys / np.sqrt((interval*chan_wid).T)

    # --------------------------------------------------------------------------
    def bessel_gauss(self, r, a=1.55, b=2.52):
        if r == 0: return 0.5
        else:      return sp.j1(np.pi*r/a)/(np.pi*r/a) * np.exp(-(r/b)**2)

    def sinc_gauss(self, r, a=1.55, b=2.52):
        return np.sinc(r/a) * np.exp(-(r/b)**2)

    def gauss(self, r, a=1.00):
        return np.exp(-(r/a)**2)

    # --------------------------------------------------------------------------
    def write_to_fits(self, objname, fitsname):
        hdu = pf.PrimaryHDU(self.cube[:,:,::-1])
        # basic info.
        hdu.header['OBJECT']  = objname
        hdu.header['BTYPE']   = 'Intensity'
        hdu.header['BUNIT']   = 'K'
        hdu.header['RADESYS'] = 'FK5'
        # axis 1
        hdu.header['CTYPE1']  = 'RA---SIN'
        hdu.header['CRVAL1']  = np.max(self.x_regrid)
        hdu.header['CRPIX1']  = 1.0
        hdu.header['CDELT1']  = -self.s_regrid
        # axis 2
        hdu.header['CTYPE2']  = 'DEC--SIN'
        hdu.header['CRVAL2']  = np.min(self.y_regrid)
        hdu.header['CRPIX2']  = 1.0
        hdu.header['CDELT2']  = +self.s_regrid
        # axis 3
        hdu.header['CTYPE3']  = 'FREQ'
        hdu.header['CRVAL3']  = np.min(self.z_regrid)
        hdu.header['CRPIX3']  = 1.0
        hdu.header['CDELT3']  = np.unique(np.diff(self.z_regrid))[0]
        # other info
        hdu.header['RESTFRQ'] = 110.20135e9
        hdu.header['SPECSYS'] = 'LSRK'
        hdu.header['VELREF']  = 257
        # write to fits
        hdu.writeto(fitsname, clobber=True)


# ==============================================================================
# ==============================================================================
class RectSelect(object):
    '''
    region = RectSelect()
    plt.show()
    '''
    def __init__(self, ax=None):
        self.ax = ax or plt.gca()
        self.rect = Rectangle((0,0), 0, 0, color='orange', alpha=0.5)
        self.ax.add_patch(self.rect)
        self.blc = np.zeros(2)
        self.brc = np.zeros(2)
        self.tlc = np.zeros(2)
        self.trc = np.zeros(2)

        def selector(event):
            if event.key in ['Q', 'q'] and selector.RS.active:
                print ('RectangleSelector deactivated.')
                selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not selector.RS.active:
                print ('RectangleSelector activated.')
                selector.RS.set_active(True)

        selector.RS = RectangleSelector(self.ax, self.callback)
        self.ax.figure.canvas.mpl_connect('key_press_event', selector)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.release)

    def callback(self, eclick, erelease):
        x0, x1 = eclick.xdata, erelease.xdata
        y0, y1 = eclick.ydata, erelease.ydata
        self.blc = min(x0, x1), min(y0, y1)
        self.brc = max(x0, x1), min(y0, y1)
        self.tlc = min(x0, x1), max(y0, y1)
        self.trc = max(x0, x1), max(y0, y1)
        blc_print = '({:0.4},{:0.4})'.format(*self.blc)
        trc_print = '({:0.4},{:0.4})'.format(*self.trc)
        print('blc={}, trc={}'.format(blc_print, trc_print))

    def release(self, event):
        self.rect.set_width(self.trc[0] - self.blc[0])
        self.rect.set_height(self.trc[1] - self.blc[1])
        self.rect.set_xy(self.blc)
        self.ax.figure.canvas.draw()


# ==============================================================================
# ==============================================================================
class SpansSelect(object):
    '''
    region = SpansSelect()
    plt.show()
    '''
    def __init__(self, ax=None):
        self.ax = ax or plt.gca()
        self.spans = list()
        self.selector = SpanSelector(self.ax, self.onselect, 'horizontal')

    def onselect(self, x_min, x_max):
        self.spans.append((x_min, x_max))
        self.ax.axvspan(x_min, x_max, facecolor='orange', alpha=0.5)
        self.ax.figure.canvas.draw()
