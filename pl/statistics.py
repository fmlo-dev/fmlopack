# coding: utf-8

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from itertools import product

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


# ==============================================================================
# ==============================================================================
class PPCA(PCA):
    # --------------------------------------------------------------------------
    def __init__(self, scan):
        PCA.__init__(self, scan, fraction=1.00)
        self.dim_data = scan.shape[1]
        self.dim_max  = scan.shape[0]
        self.l_org = np.zeros(self.dim_data)
        self.l_org[:len(self.eigen)] = self.eigen / len(self.eigen)

    # --------------------------------------------------------------------------
    def laplace(self, k, sharpness=1.0):
        '''
        return probability (log10, not normalized) of k-dim model
        '''
        d = self.dim_data
        N = self.dim_max
        m = d*k - k*(k+1)/2.0
        a = sharpness
        n = N + 1.0 - a
        assert 0 < k < N

        v_opt = N*np.sum(self.l_org[k:]) / (n*(d-k)-2.0)
        l_opt = np.zeros_like(self.l_org)
        l_opt[:k] = (N*self.l_org[:k]+a) / (N-1.0+a)
        l_opt[k:] = v_opt

        Az = self._Az(k, self.l_org, l_opt)
        pU = self._pU(k)

        p_dim  = 0.0
        p_dim += pU
        p_dim += (-N/2.0) * np.log10(np.prod(self.l_org[:k]))
        p_dim += (-N*(d-k)/2.0) * np.log10(v_opt)
        p_dim += ((m+k)/2.0) * np.log10(2.0*np.pi)
        p_dim += -0.5 * Az
        p_dim += -(k/2.0) * np.log10(N)

        return p_dim

    # --------------------------------------------------------------------------
    def bic(self, k, sharpness=1.0):
        '''
        return probability (log10, not normalized) of k-dim model
        using BIC approximation
        '''
        d = self.dim_data
        N = self.dim_max
        m = d*k - k*(k+1)/2.0
        a = sharpness
        n = N + 1.0 - a
        assert 0 < k < N

        v_opt = N*np.sum(self.l_org[k:]) / (n*(d-k)-2.0)

        p_dim  =  0.0
        p_dim += (-N/2.0) * np.log10(np.prod(self.l_org[:k]))
        p_dim += (-N*(d-k)/2.0) * np.log10(v_opt)
        p_dim += (-(m+k)/2.0) * np.log10(N)

        return p_dim

    # --------------------------------------------------------------------------
    def _pU(self, k):
        d  = self.dim_data
        i  = np.arange(k)+1.0
        q  = (d-i+1.0)/2.0

        pU = 0.0
        pU += -k * np.log10(2)
        pU += np.sum(-q * np.log10(np.pi))
        pU += np.sum((sp.gammaln(q)/np.log(10)))

        return pU

    # --------------------------------------------------------------------------
    def _Az(self, k, l_org, l_opt):
        N  = self.dim_max
        d  = self.dim_data

        l_org_v = l_org[:k].reshape((k,1))
        l_org_h = l_org
        l_opt_v = l_opt[:k].reshape((k,1))
        l_opt_h = l_opt

        Az_map = np.log10((1/l_opt_h-1/l_opt_v) * (l_org_v-l_org_h) * N)
        Az_sum = np.sum(np.ma.masked_invalid(np.triu(Az_map)))

        return Az_sum