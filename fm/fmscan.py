# coding: utf-8

'''
module: fmlopack.fm.fmscan
author: Akio Taniguchi
affill: Institute of Astronomy, the University of Tokyo
mailto: taniguchi_at_ioa.s.u-tokyo.ac.jp
'''

version = '1.2'

# ==============================================================================
# ==============================================================================
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

        # go to __array_finalize__
        scan.tsys     = tsys
        scan.fmrecord = fmrecord
        scan.fmstatus = fmstatus
        return scan

    # --------------------------------------------------------------------------
    def demodulate(self, target='observed'):
        # check valid
        if not self.fmstatus == 'modulated':
            raise FmScanError('this attribute is unavailable in demodulated scan')

        # demodulate scan
        if target == 'observed':
            fmstatus   = 'demodulated-obs'
            demod_scan = self._demod(self, self.fmrecord.CHANFM)

        elif target == 'image':
            fmstatus   = 'demodulated-img'
            demod_scan = self._demod(self, self.fmrecord.CHANFM*(-1))

        else: raise FmScanError('invalid target selected')

        return FmScan(demod_scan, self.tsys, self.fmrecord, fmstatus)

    # --------------------------------------------------------------------------
    def modulate(self):
        # check valid
        if self.fmstatus == 'modulated':
            raise FmScanError('this attribute is unavailable in modulated scan')

        # modulate scan
        if self.fmstatus == 'demodulated-obs':
            mod_scan = self._mod(self, self.fmrecord.CHANFM, len(self.tsys))

        elif self.fmstatus == 'demodulated-img':
            mod_scan = self._mod(self, self.fmrecord.CHANFM*(-1), len(self.tsys))

        return FmScan(mod_scan, self.tsys, self.fmrecord, 'modulated')

    # --------------------------------------------------------------------------
    def nuobs(self, useGHz=True):
        # check valid
        if self.fmstatus == 'modulated':
            raise FmScanError('this attribute is unavailable in modulated scan')

        factor = 1e-9 if useGHz else 1.0
        nuobs_min = self.fmrecord.FREQRANGE[:,0].min()
        nuobs_max = self.fmrecord.FREQRANGE[:,1].max()
        nuobs_num = self.shape[1]

        return factor * np.linspace(nuobs_min, nuobs_max, nuobs_num)

    # --------------------------------------------------------------------------
    def spectrum(self, mode='signal'):
        # check valid 
        if self.fmstatus == 'modulated':
            raise FmScanError('this attribute is unavailable in modulated scan')

        if   mode == 'signal': return self._spec()
        elif mode == 'noise-tsys': return self._noise_tsys_spec()
        elif mode == 'noise-jk':   return self._noise_jk_spec()

    # --------------------------------------------------------------------------
    def pca(self, target='clean', fraction=0.990, time_chunk=None, npc_max=None):
        in_scan  = np.asarray(self)
        out_scan = np.zeros_like(in_scan)
        tchunk   = time_chunk or len(in_scan)

        # PCA cleaning
        for i in range(len(in_scan)/tchunk):
            in_i  = in_scan[i*tchunk:(i+1)*tchunk]
            p     = stat.PCA(in_i, fraction)
            npc   = p.npc if npc_max is None else min(p.npc, npc_max)
            print('Npc = {}'.format(npc))
            com_i = np.dot(p.U[:,:npc], np.dot(np.diag(p.d[:npc]), p.Vt[:npc]))

            if target == 'clean':
                out_scan[i*tchunk:(i+1)*tchunk] = in_i - com_i
            elif target == 'common':
                out_scan[i*tchunk:(i+1)*tchunk] = com_i
            else:
                raise FmScanError('invalid target selected')

        return FmScan(out_scan, self.tsys, self.fmrecord, self.fmstatus)

    # --------------------------------------------------------------------------
    def ppca(self, target='clean', mode='laplace', time_chunk=None, npc_max=None, dev=False):
        in_scan  = np.asarray(self)
        out_scan = np.zeros_like(in_scan)
        tchunk   = time_chunk or len(in_scan)
        probs    = []

        # PCA cleaning
        for i in range(len(in_scan)/tchunk):
            in_i  = in_scan[i*tchunk:(i+1)*tchunk]
            p     = stat.PPCA(in_i, mode, npc_max)
            npc   = np.argmax(p.probs)+1
            probs.append(p.probs)
            print('Npc = {}'.format(npc))
            com_i = np.dot(p.U[:,:npc], np.dot(np.diag(p.d[:npc]), p.Vt[:npc]))

            if target == 'clean':
                out_scan[i*tchunk:(i+1)*tchunk] = in_i - com_i
            elif target == 'common':
                out_scan[i*tchunk:(i+1)*tchunk] = com_i
            else:
                raise FmScanError('invalid target selected')

        probs = np.asarray(probs).mean(0)

        if dev: 
            return FmScan(out_scan, self.tsys, self.fmrecord, self.fmstatus), probs
        else:
            return FmScan(out_scan, self.tsys, self.fmrecord, self.fmstatus)

    # --------------------------------------------------------------------------
    def offset(self, fmrecord_offset=0):
        offset = int(fmrecord_offset)
        if offset >= 0:
            scan     = np.copy(self[offset:])
            fmrecord = np.copy(self.fmrecord[:-offset])
        else:
            scan     = np.copy(self[:offset])
            fmrecord = np.copy(self.fmrecord[-offset:])

        return FmScan(scan, self.tsys, fmrecord, self.fmstatus)

    # --------------------------------------------------------------------------
    def _demod(self, mod, ch_fm):
        ch_fm = np.copy(np.asarray(ch_fm))
        mod   = np.copy(np.asarray(mod))
        empty = np.copy(np.zeros((mod.shape[0], ch_fm.ptp())))
        demod = np.copy(np.hstack((mod, empty)))

        for i in range(len(demod)):
            demod[i] = np.roll(demod[i], ch_fm[i]-ch_fm.min())

        return demod

    # --------------------------------------------------------------------------
    def _mod(self, demod, ch_fm, wid):
        ch_fm = np.copy(np.asarray(ch_fm))
        demod = np.copy(np.asarray(demod))

        for i in range(len(demod)):
            demod[i] = np.roll(demod[i], ch_fm[i]*(-1)+ch_fm.min())

        return demod[:,:wid]

    # --------------------------------------------------------------------------
    def _spec(self):
        spec   = np.sum(np.asarray(self), axis=0)
        tinteg = np.sum((self._integmap() != 0.0), axis=0)

        return spec / tinteg

    # --------------------------------------------------------------------------
    def _noise_tsys_spec(self):
        interval  = np.tile(self.fmrecord.INTERVAL, (len(self.tsys),1))
        chan_wid  = np.diff(self.fmrecord.FREQRANGE)[:,0]/(len(self.tsys)-1)
        noise_map = (interval*chan_wid).T / (self.tsys)**2

        if self.fmstatus == 'demodulated-obs':
            noise_map = self._demod(noise_map, self.fmrecord.CHANFM)

        elif self.fmstatus == 'demodulated-img':
            noise_map = self._demod(noise_map, self.fmrecord.CHANFM*(-1))

        else: raise FmScanError('invalid target selected')

        return np.sum(noise_map, axis=0)**(-0.5)

    # --------------------------------------------------------------------------
    def _noise_tsys_spec2(self):
        '''
        for test
        '''
        interval  = np.tile(self.fmrecord.INTERVAL, (len(self.tsys),1))
        interval[:] = 0.09
        chan_wid  = np.diff(self.fmrecord.FREQRANGE)[:,0]/(len(self.tsys)-1)
        noise_map = (self.tsys)**2 / (interval*chan_wid).T

        if self.fmstatus == 'demodulated-obs':
            noise_map = self._demod(noise_map, self.fmrecord.CHANFM)

        elif self.fmstatus == 'demodulated-img':
            noise_map = self._demod(noise_map, self.fmrecord.CHANFM*(-1))

        else: raise FmScanError('invalid target selected')

        spec   = np.sum(noise_map, axis=0)
        tinteg = np.sum((self._integmap() != 0.0), axis=0)

        return spec**0.5 / tinteg

    # --------------------------------------------------------------------------
    def _noise_jk_spec(self, sampling=100):
        in_scan  = np.asarray(self)
        out_scan = np.zeros((sampling, in_scan.shape[1]))

        for i in range(sampling):
            r = np.random.randint(0,2,in_scan.shape[0]) * 2 - 1
            spec   = np.sum((in_scan.T * r).T, axis=0)
            tinteg = np.sum((self._integmap() != 0.0), axis=0)
            out_scan[i] = spec / tinteg

        return np.std(out_scan, axis=0)

    # --------------------------------------------------------------------------
    def _noise_jk_spec2(self, sampling=100):
        in_scan  = np.asarray(self)/0.9
        out_scan = np.zeros((sampling, in_scan.shape[1]))

        for i in range(sampling):
            r = np.random.randint(0,2,in_scan.shape[0]) * 2 - 1
            spec   = np.sum((in_scan.T * r).T, axis=0)
            tinteg = np.sum((self._integmap() != 0.0), axis=0)
            out_scan[i] = spec / tinteg

        return np.std(out_scan, axis=0)

    # --------------------------------------------------------------------------
    def _integmap(self):
        interval = np.tile(self.fmrecord.INTERVAL, (len(self.tsys),1)).T

        if self.fmstatus == 'demodulated-obs':
            integ_map = self._demod(interval, self.fmrecord.CHANFM)

        elif self.fmstatus == 'demodulated-img':
            integ_map = self._demod(interval, self.fmrecord.CHANFM*(-1))

        else:
            raise FmScanError('invalid target selected')

        return integ_map

    # --------------------------------------------------------------------------
    def __idx__(self, *idx):
        '''
        return index (slice object) of fmrecord
        this method is for system and automatically called when slicing FmScan
        '''
        idx = idx[0]

        if type(idx) == tuple:
            if type(idx[0]) == int: idx_fmrecord = slice(idx[0], idx[0]+1, 1)
            else: idx_fmrecord = idx[0]

        elif type(idx) == int:
            idx_fmrecord = slice(idx, idx+1, 1)

        else: idx_fmrecord = idx

        return idx_fmrecord

    # --------------------------------------------------------------------------
    def __array_finalize__(self, scan):
        if scan is None: return
        self.tsys     = getattr(scan, 'tsys', None)
        self.fmrecord = getattr(scan, 'fmrecord', None)
        self.fmstatus = getattr(scan, 'fmstatus', None)

    # --------------------------------------------------------------------------
    def __array_wrap__(self, out_scan, context=None):
        return np.ndarray.__array_wrap__(self, out_scan, context)

    # --------------------------------------------------------------------------
    def __getitem__(self, *idx):
        sl_scan     = np.asarray(self).__getitem__(*idx)
        sl_fmrecord = self.fmrecord.__getitem__(self.__idx__(*idx))

        return FmScan(sl_scan, self.tsys, sl_fmrecord, self.fmstatus)

    # --------------------------------------------------------------------------
    def __getslice__(self, *idx):
        sl_scan     = np.asarray(self).__getslice__(*idx)
        sl_fmrecord = self.fmrecord.__getslice__(*idx)

        return FmScan(sl_scan, self.tsys, sl_fmrecord, self.fmstatus)

    # --------------------------------------------------------------------------
    def __repr__(self):
        repr_fmt = 'FmScan({}x{}: {},\n      {})'
        str_scan = (',\n'+' '*6).join(str(np.asarray(self)).split('\n'))
        if   self.ndim == 0: n, m = 1, 1
        elif self.ndim <= 1: n, m = 1, self.shape[0]
        else: n, m = self.shape

        return repr_fmt.format(n, m, self.fmstatus, str_scan)

    # --------------------------------------------------------------------------
    def __str__(self):
        str_fmt  = 'FmScan({}x{}: {},\n      {})'
        str_scan = (',\n'+' '*6).join(str(np.asarray(self)).split('\n'))
        if   self.ndim == 0: n, m = 1, 1
        elif self.ndim <= 1: n, m = 1, self.shape[0]
        else: n, m = self.shape

        return str_fmt.format(n, m, self.fmstatus, str_scan)

    # --------------------------------------------------------------------------
    def __reduce__(self):
        obj_state = np.ndarray.__reduce__(self)
        sub_state = (self.tsys, self.fmstatus, self.fmrecord)
        return (obj_state[0], obj_state[1], (obj_state[2], sub_state))

    # --------------------------------------------------------------------------
    def __setstate__(self, state):
        obj_state, sub_state = state
        np.ndarray.__setstate__(self, obj_state)
        (self.tsys, self.fmstatus, self.fmrecord) = sub_state


# ==============================================================================
# ==============================================================================
def zeros_like(fmscan):
    scan     = np.zeros_like(fmscan)
    tsys     = fmscan.tsys
    fmrecord = fmscan.fmrecord
    fmstatus = fmscan.fmstatus

    return FmScan(scan, tsys, fmrecord, fmstatus)

def copy(fmscan):
    scan     = np.copy(fmscan)
    tsys     = fmscan.tsys
    fmrecord = fmscan.fmrecord
    fmstatus = fmscan.fmstatus

    return FmScan(scan, tsys, fmrecord, fmstatus)

def savez(filename, fmscan):
    scan     = np.copy(fmscan)
    tsys     = fmscan.tsys
    fmrecord = fmscan.fmrecord
    fmstatus = fmscan.fmstatus
    np.savez(filename, scan=scan, tsys=tsys, fmrecord=fmrecord, fmstatus=fmstatus)

def load(filename):
    f = np.load(filename)
    return FmScan(f['scan'], f['tsys'], f['fmrecord'], f['fmstatus'])



# ==============================================================================
# ==============================================================================
class FmScanError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message