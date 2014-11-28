# coding: utf-8

'''
module: fmlopack.io.nro45m
author: Akio Taniguchi
affill: Institute of Astronomy, the University of Tokyo
mailto: taniguchi_at_ioa.s.u-tokyo.ac.jp
'''

version  = '1.0'

# ==============================================================================
# ==============================================================================
try:
    import builtins
    import tkinter
except ImportError:
    import __builtin__ as builtins
    import Tkinter as tkinter

import os
import re
import sys
import shutil
import tempfile
import tkFileDialog
from datetime import datetime
from decimal  import Decimal
from subprocess import Popen, PIPE

import numpy  as np
import pyfits

import fmlopack.fm.fmscan as fms


# ==============================================================================
# ==============================================================================
def load(obstable, sam45log, fmlolog, antlog, **kwargs):
    hdulist = Nro45mData()
    hdulist._obstable(obstable)
    hdulist._sam45dict()
    hdulist._fmlolog(fmlolog)
    hdulist._antlog(antlog)
    hdulist._sam45log(sam45log)
    hdulist.info()

    return hdulist

def open(fitsname=None, mode='readonly', memmap=None, save_backup=False, **kwargs):
    if fitsname == '' or fitsname is None:
        root = tkinter.Tk()
        root.withdraw()
        fitsname = tkFileDialog.askopenfilename()
        root.destroy()
        if fitsname == '': return

    hdulist = Nro45mData.fromfile(fitsname, mode, memmap, save_backup, **kwargs)
    hdulist._sam45dict()
    hdulist.info()

    return hdulist


# ==============================================================================
# ==============================================================================
class Nro45mData(pyfits.HDUList):
    # --------------------------------------------------------------------------
    def __init__(self, hdus=[], file=None):
        pyfits.HDUList.__init__(self, hdus, file)

    def version(self):
        return self['PRIMARY'].header['VERSION']

    # --------------------------------------------------------------------------
    def fmscan(self, array_id, offset_fmlolog=3, offset_anglog=995):
        '''
        Return a FmScan of the selected array ID (e.g. A5)
        '''
        data     = self[array_id].data
        header   = self[array_id].header
        scan_wid = header['NAXIS1']
        scan_len = header['NAXIS2']
        chan_wid = header['BANDWID'] / scan_wid

        fmlolog = self['FMLOLOG'].data[offset_fmlolog:offset_fmlolog+scan_len]
        chan_fm = fmlolog.FREQFM    / chan_wid

        antlog  = self['ANTLOG'].data[offset_fmlolog:offset_fmlolog+scan_len]
        radec   = antlog.RADEC

        freq_min   = header['RESTFREQ'] - 0.5*(scan_wid-1)*chan_wid
        freq_max   = header['RESTFREQ'] + 0.5*(scan_wid-1)*chan_wid
        freq_min  += chan_fm * chan_wid
        freq_max  += chan_fm * chan_wid
        freq_range = np.vstack((freq_min, freq_max)).T

        # other components
        date_time = fmlolog.TIME
        interval  = np.tile(header['CDELT2'], scan_len)
        tsys      = np.asarray(header['TSYS'].split(','), 'f8')
        fmstatus  = 'modulated'

        # make fminfo
        alist  = [chan_fm, freq_range, interval, date_time, radec]
        names  = ['CHANFM', 'FREQRANGE', 'INTERVAL', 'DATETIME', 'RADEC']
        dtypes = ['i8', 'f8', 'f8', 'a26', 'f8']
        shapes = [1, 2, 1, 1, 2]
        fmrecord = np.rec.fromarrays(alist, zip(names, dtypes, shapes))

        return fms.FmScan(data, tsys, fmrecord, fmstatus)

    # --------------------------------------------------------------------------
    def _obstable(self, obstable):
        '''
        Load a obstable and append a PrimaryHDU to HDUList
        '''
        hdu = pyfits.PrimaryHDU()
        hdu.header['ORGFILE'] = obstable.split('/')[-1], 'Original file'
        hdu.header['VERSION'] = version, 'Version of fmlopack'

        idx = 0
        for line in builtins.open(obstable, 'r'):
            if re.search('Initialize', line): break

            if re.search('^SET SAM45', line): key_type = 'SAM'
            elif re.search('^SET ANT', line): key_type = 'ANT'
            elif re.search('^SET MRG', line): key_type = 'MRG'
            elif re.search('^SET RXT', line): key_type = 'RXT'
            elif re.search('^SET IFATT', line): key_type = 'ATT'
            elif re.search('^SET GRPTRK', line): key_type = 'GRP'
            elif re.search('^SET SYNTHE_H', line): key_type = 'SYH'
            elif re.search('^SET SYNTHE_E', line): key_type = 'SYE'
            else: continue

            key  = '{}-{:0>3}'.format(key_type, idx)
            item = '{}={}'.format(line.split()[2], line.split()[3].strip("(')")) 
            hdu.header[key] = item
            idx += 1

        self.append(hdu)

    # --------------------------------------------------------------------------
    def _sam45dict(self):
        '''
        Append a dict containing various parameters of SAM45
        '''
        hdr = self['PRIMARY'].header
        dic = dict()

        for key_hdr in hdr:
            if not(re.search('^SAM', key_hdr)): continue

            key      = hdr[key_hdr].split('=')[0]
            item     = hdr[key_hdr].split('=')[1].split(',')
            dtype    = self._sam45dict_config(key, 'dtype')
            shape    = self._sam45dict_config(key, 'shape')
            item     = np.array(item, dtype).reshape(shape)
            dic[key] = item.tolist()[0] if len(item)==1 else item

        self.sam45dict = dic

    # --------------------------------------------------------------------------
    def _sam45log(self, sam45log, array_ids='all', tamb=293.0, sldump='sldump'):
        '''
        Load a SAM45 logging and append ImageHDU(s) to HDUList.
        '''
        # obstable info from obstdict
        sam = self.sam45dict
        array_use = sam['ARRAY'] == 1
        array_max = sam['ARRAY'].sum()
        scan_wid  = np.max(sam['CH_RANGE'])
        scan_len  = int(sam['INTEG_TIME'] / sam['IPTIM'])
        idx_to_id = lambda idx: 'A{}'.format(idx+1)
        id_to_idx = lambda aid: int(aid.strip('A'))-1

        if array_ids == 'all':
            array_idx = range(array_max)
        else:
            array_ids = sorted(array_ids)
            array_idx = map(id_to_idx, array_ids)

        # dump sam45log using sldump (external code)
        print('dumping:  {}'.format(sam45log))
        dump_dir  = tempfile.mkdtemp(dir=os.environ['HOME'])
        dump_file = dump_dir+'/dump.txt'
        proc = Popen([sldump, sam45log, dump_file, '1', '4096'], stderr=PIPE)
        proc.communicate()

        # load dumped sam45log
        print('loading:  {} (temporary file)'.format(dump_file))

        zero = np.empty(scan_wid*array_max, 'f8')
        r    = np.empty(scan_wid*array_max, 'f8')
        sky  = np.empty(scan_wid*array_max, 'f8')
        on   = np.empty((scan_len, scan_wid*array_max), 'f8')

        try:
            f = builtins.open(dump_file, 'r')
            for (idx, line) in enumerate(f):
                i, j = idx // array_max, idx % array_max
                time_slice = np.asarray(line.split()[3:], 'f8')

                # zero, r, sky, on
                if   i == 0: zero[scan_wid*j: scan_wid*(j+1)] = time_slice
                elif i == 1: r[scan_wid*j: scan_wid*(j+1)]    = time_slice
                elif i == 2: sky[scan_wid*j: scan_wid*(j+1)]  = time_slice
                elif 3 <= i < scan_len+3:
                    on[i-3, scan_wid*j: scan_wid*(j+1)] = time_slice
        
        finally:
            print('removing: {}'.format(dump_file))
            f.close()
            shutil.rmtree(dump_dir)

        # calibration and tsys
        att  = np.repeat(sam['IFATT'][array_use], scan_wid)
        tsys = tamb / (10**(0.1*att) * ((r-zero)/(sky-zero)) - 1)
        scan = tamb * (on-np.median(on,0)) / (10**(0.1*att)*(r-zero)-(sky-zero))

        # append ImageHDUs
        for j in range(array_max):
            if not(j in array_idx): continue
            if sam['SIDBD_TYP'][j] == 'USB':
                hdu_data = scan[:, scan_wid*j: scan_wid*(j+1)]
                hdu_tsys = tsys[scan_wid*j: scan_wid*(j+1)]
                hdu_tsys_str = str(list(hdu_tsys)).strip('[]')

            elif sam['SIDBD_TYP'][j] == 'LSB':
                hdu_data = scan[:, scan_wid*j: scan_wid*(j+1)][:,::-1]
                hdu_tsys = tsys[scan_wid*j: scan_wid*(j+1)][::-1]
                hdu_tsys_str = str(list(hdu_tsys)).strip('[]')

            hdu = pyfits.ImageHDU()
            hdu.data = hdu_data
            hdu.header['EXTNAME']  = idx_to_id(j), 'Name of HDU'
            hdu.header['ORGFILE']  = sam45log.split('/')[-1], 'Original file'
            hdu.header['OBJECT']   = sam['SRC_NAME']
            hdu.header['RA']       = sam['SRC_POS'][0], 'Right Ascention (deg)'
            hdu.header['DEC']      = sam['SRC_POS'][1], 'Declination (deg)'
            hdu.header['BANDWID']  = sam['OBS_BAND'][j], 'Band width (Hz)'
            hdu.header['RESTFREQ'] = sam['REST_FREQ'][j], 'Rest frequency (Hz)'
            hdu.header['SIDEBAND'] = sam['SIDBD_TYP'][j], 'USB or LSB'
            hdu.header['CTYPE1']   = 'Spectral'
            hdu.header['CUNIT1']   = 'ch'
            hdu.header['CDELT1']   = hdu.header['BANDWID'] / hdu.header['NAXIS1']
            hdu.header['CTYPE2']   = 'Time'
            hdu.header['CDELT2']   = sam['IPTIM']
            hdu.header['CUNIT2']   = '{} sec'.format(sam['IPTIM'])
            hdu.header['BSCALE']   = 1.0, 'PHYSICAL = PIXEL*BSCALE + BZERO'
            hdu.header['BZERO']    = 0.0
            hdu.header['BUNIT']    = 'K'
            hdu.header['BTYPE']    = 'Intensity'
            hdu.header['TAMB']     = tamb, 'Ambient temperature (K)'
            hdu.header['TSYS']     = hdu_tsys_str, 'System noise temperature (K)'

            self.append(hdu)

    # --------------------------------------------------------------------------
    def _fmlolog(self, fmlolog, skiprows=1):
        '''
        Load a fmlolog and append a BinTableHDU to HDUList. 
        '''
        time    = []
        status  = []
        freq_fm = []
        freq_lo = []
        v_rad   = []

        # load fmlolog
        f = builtins.open(fmlolog, 'r')
        for (i, line) in enumerate(f):
            if i < skiprows: continue

            items = line.split()
            time_fmt_r = '%Y%m%d%H%M%S.%f'
            time_fmt_w = '%Y-%m-%dT%H:%M:%S.%f'
            time_str   = '{:.6f}'.format(Decimal(items[0]))
            time_dt    = datetime.strptime(time_str, time_fmt_r)

            time.append(time_dt.strftime(time_fmt_w))
            status.append(items[1])
            freq_fm.append(items[2])
            freq_lo.append(items[3])
            v_rad.append(items[4])

        f.close()

        # append BinTableHDU
        alist  = [time, freq_fm, freq_lo, v_rad]
        names  = ['TIME', 'FREQFM', 'FREQLO', 'VRAD']
        dtypes = ['a26', 'f8', 'f8', 'f8']

        hdu = pyfits.BinTableHDU()
        hdu.data = np.rec.fromarrays(alist, zip(names, dtypes))

        hdu.header['EXTNAME'] = 'FMLOLOG', 'Name of HDU'
        hdu.header['ORGFILE'] = fmlolog.split('/')[-1], 'Original file'
        hdu.header['TUNIT1']  = 'YYYY-MM-DDThh:mm:ss.ssssss'
        hdu.header['TUNIT2']  = 'Hz'
        hdu.header['TUNIT3']  = 'Hz'
        hdu.header['TUNIT4']  = 'km/s'
 
        self.append(hdu)

    # --------------------------------------------------------------------------
    def _antlog(self, antlog, skiprows=1):
        '''
        Load a antlog and append a BinTableHDU to HDUList
        '''
        time   = []
        radec  = []
        azel_1 = []
        azel_2 = []
        offset = []

        # load antlog
        f = builtins.open(antlog, 'r')
        for (i, line) in enumerate(f):
            if i < skiprows: continue

            items = line.split()
            time_fmt_r = '%y%m%d%H%M%S.%f'
            time_fmt_w = '%Y-%m-%dT%H:%M:%S.%f'
            time_str   = '{:.6f}'.format(Decimal(items[0]))
            time_dt    = datetime.strptime(time_str, time_fmt_r)

            time.append(time_dt.strftime(time_fmt_w))
            radec.append([items[1], items[2]])
            azel_1.append([items[3], items[4]])
            azel_2.append([items[5], items[6]])
            offset.append([items[7], items[8]])

        f.close()

        # append BinTableHDU
        alist  = [time, radec, azel_1, azel_2, offset]
        names  = ['TIME', 'RADEC', 'AZEL1', 'AZEL2', 'OFFSET']
        dtypes = ['a26', 'f8', 'f8', 'f8', 'f8']
        shapes = [1, 2, 2, 2, 2]

        hdu = pyfits.BinTableHDU()
        hdu.data = np.rec.fromarrays(alist, zip(names, dtypes, shapes))

        hdu.header['EXTNAME'] = 'ANTLOG', 'Name of HDU'
        hdu.header['ORGFILE'] = antlog.split('/')[-1], 'Original file'
        hdu.header['TUNIT1']  = 'YYYY-MM-DDThh:mm:ss.ssssss'
        hdu.header['TUNIT2']  = 'deg'
        hdu.header['TUNIT3']  = 'deg'
        hdu.header['TUNIT4']  = 'deg'
        hdu.header['TUNIT5']  = 'deg'

        self.append(hdu)

    # --------------------------------------------------------------------------
    def _sam45dict_config(self, key, prop):
        cfg = dict()
        cfg['INTEG_TIME']         = {'dtype': np.int,   'shape': None}
        cfg['CALB_INT']           = {'dtype': np.int,   'shape': None}
        cfg['IPTIM']              = {'dtype': np.float, 'shape': None}
        cfg['FREQ_INTVAL']        = {'dtype': np.int,   'shape': None}
        cfg['VELO']               = {'dtype': np.float, 'shape': None}
        cfg['MAP_POS']            = {'dtype': np.int,   'shape': None}
        cfg['FREQ_SW']            = {'dtype': np.int,   'shape': None}
        cfg['MULT_OFF']           = {'dtype': np.float, 'shape': None}
        cfg['MULT_NUM']           = {'dtype': np.int,   'shape': None}
        cfg['REF_NUM']            = {'dtype': np.int,   'shape': None}
        cfg['REST_FREQ']          = {'dtype': np.float, 'shape': None}
        cfg['OBS_FREQ']           = {'dtype': np.float, 'shape': None}
        cfg['FREQ_IF1']           = {'dtype': np.float, 'shape': None}
        cfg['OBS_BAND']           = {'dtype': np.float, 'shape': None}
        cfg['ARRAY']              = {'dtype': np.int,   'shape': None}
        cfg['IFATT']              = {'dtype': np.int,   'shape': None}
        cfg['FQDAT_F0']           = {'dtype': np.float, 'shape': None}
        cfg['FQDAT_FQ']           = {'dtype': np.float, 'shape': None}
        cfg['FQDAT_CH']           = {'dtype': np.int,   'shape': (32,2)}
        cfg['SRC_POS']            = {'dtype': np.float, 'shape': None}
        cfg['CH_BAND']            = {'dtype': np.int,   'shape': None}
        cfg['CH_RANGE']           = {'dtype': np.int,   'shape': (32,2)}
        cfg['QL_RMSLIMIT']        = {'dtype': np.float, 'shape': None}
        cfg['QL_POINTNUM']        = {'dtype': np.int,   'shape': None}
        cfg['BIN_NUM']            = {'dtype': np.int,   'shape': None}
        cfg['N_SPEC_WINDOW_SUB1'] = {'dtype': np.int,   'shape': None}
        cfg['START_CHAN_SUB1']    = {'dtype': np.int,   'shape': None}
        cfg['END_CHAN_SUB1']      = {'dtype': np.int,   'shape': None}
        cfg['CHAN_AVG_SUB1']      = {'dtype': np.int,   'shape': None}
        cfg['N_SPEC_WINDOW_SUB2'] = {'dtype': np.int,   'shape': None}
        cfg['START_CHAN_SUB2']    = {'dtype': np.int,   'shape': None}
        cfg['END_CHAN_SUB2']      = {'dtype': np.int,   'shape': None}
        cfg['CHAN_AVG_SUB2']      = {'dtype': np.int,   'shape': None}

        try: return cfg[key][prop]
        except: return None


# ==============================================================================
# ==============================================================================
class Nro45mDataError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message