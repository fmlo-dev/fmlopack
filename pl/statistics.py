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
        scan = np.asarray(scan)
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
    def __init__(self, scan, mode='laplace', npc_max=None):
        PCA.__init__(self, scan, fraction=1.00)
        self.dim_data = scan.shape[1]
        self.dim_max  = scan.shape[0]
        self.npc_max  = self.dim_max if npc_max is None else npc_max
        self.k_range  = np.arange(1, self.npc_max)
        self.mode     = mode

        self.l_org = np.zeros(self.dim_data)
        self.l_org[:len(self.eigen)] = self.eigen / len(self.eigen)
        
        self.probs = np.zeros_like(self.k_range)
        for k in self.k_range:
            self.probs[k-1] = getattr(self, self.mode)(k)

    # --------------------------------------------------------------------------
    def laplace(self, k):
        '''
        return probability (log10, not normalized) of k-dim model
        '''
        D = self.dim_data
        N = self.dim_max
        m = D*k - k*(k+1)/2.0
        assert 0 < k < N
        v_opt = np.sum(self.l_org[k:]) / (D-k)
        l_opt = np.zeros_like(self.l_org)
        l_opt[:k] = self.l_org[:k]
        l_opt[k:] = v_opt

        Az = self._Az(k, self.l_org, l_opt)
        pU = self._pU(k)

        p_dim  = 0.0
        p_dim += pU
        p_dim += (-N/2.0) * np.sum(np.log10(self.l_org[:k]))
        p_dim += (-N*(D-k)/2.0) * np.log10(v_opt)
        p_dim += ((m+k)/2.0) * np.log10(2.0*np.pi)
        p_dim += -0.5 * Az
        p_dim += -(k/2.0) * np.log10(N)

        return p_dim

    # --------------------------------------------------------------------------
    def bic(self, k):
        '''
        return probability (log10, not normalized) of k-dim model
        using BIC approximation
        '''
        D = self.dim_data
        N = self.dim_max
        m = D*k - k*(k+1)/2.0
        assert 0 < k < N

        v_opt = np.sum(self.l_org[k:]) / (D-k)

        p_dim  =  0.0
        p_dim += (-N/2.0) * np.sum(np.log10(self.l_org[:k]))
        p_dim += (-N*(D-k)/2.0) * np.log10(v_opt)
        p_dim += (-(m+k)/2.0) * np.log10(N)

        return p_dim

    # --------------------------------------------------------------------------
    def _pU(self, k):
        D  = self.dim_data
        i  = np.arange(k)+1.0
        q  = (D-i+1.0)/2.0

        pU = 0.0
        pU += -k * np.log10(2)
        pU += np.sum(-q * np.log10(np.pi))
        pU += np.sum((sp.gammaln(q)/np.log(10)))

        return pU

    # --------------------------------------------------------------------------
    def _Az(self, k, l_org, l_opt):
        N  = self.dim_max

        l_org_v = l_org[:k].reshape((k,1))
        l_org_h = l_org
        l_opt_v = l_opt[:k].reshape((k,1))
        l_opt_h = l_opt

        Az_map = np.log10((1/l_opt_h-1/l_opt_v) * (l_org_v-l_org_h) * N)
        Az_sum = np.sum(np.ma.masked_invalid(np.triu(Az_map)))

        return Az_sum