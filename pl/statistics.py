# coding: utf-8

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

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


class PPCA(PCA):
    # --------------------------------------------------------------------------
    def __init__(self, scan):
        PCA.__init__(self, scan, fraction=1.00)
        self.dim_data = scan.shape[1]
        self.dim_max  = scan.shape[0]
        self.l_org = np.zeros(self.dim_data)
        self.l_org[:len(self.eigen)] = self.eigen / len(self.eigen)
        self.bic = np.asarray([self.prob_dim_bic(k) for k in range(1, self.dim_max)])

    # --------------------------------------------------------------------------
    def prob_dim_laplace(self, k, sharpness=1.0):
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

        p_dim = pU
        p_dim *= np.prod(self.l_org[k:])**(-N/2.0)
        p_dim *= v_opt**(-N*(d-k)/2.0)
        p_dim *= (2*np.pi)**((m+k)/2.0)
        p_dim *= Az**(-0.5)
        p_dim *= N**(-k/2.0)

        return p_dim

    # --------------------------------------------------------------------------
    def prob_dim_bic(self, k, sharpness=1.0):
        d = self.dim_data
        N = self.dim_max
        m = d*k - k*(k+1)/2.0
        a = sharpness
        n = N + 1.0 - a
        assert 0 < k < N

        v_opt = N*np.sum(self.l_org[k:]) / (n*(d-k)-2.0)

        p_dim =  0.0
        p_dim += (-N/2.0) * np.log10(np.prod(self.l_org[:k]))
        p_dim += (-N*(d-k)/2.0) * np.log10(v_opt)
        p_dim += (-(m+k)/2.0) * np.log10(N)

        return p_dim

    # --------------------------------------------------------------------------
    def _pU(self, k):
        d  = self.dim_data
        i  = np.arange(k)+1.0
        q  = (d-i+1.0)/2.0
        
        pU = 2**(-k) * np.prod(sp.gamma(q)*np.pi**q)

        return pU

    # --------------------------------------------------------------------------
    def _Az(self, k, l_org, l_opt):
        N  = self.dim_max
        d  = self.dim_data
        Az = 1.0

        for i in range(k):
            for j in range(i+1, d):
                Az *= (1/l_opt[j]-1/l_opt[i]) * (l_org[i]-l_org[j]) * N

        return Az