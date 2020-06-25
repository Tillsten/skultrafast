import numpy as np
from skultrafast.utils import pfid_r4, pfid_r6, pfid
import lmfit
from dataclasses import dataclass, field
import typing as T

from skultrafast.dataset import TimeResSpec, PolTRSpec
from skultrafast.unit_conversions import dichro2angle, angle2dichro

import numba

@numba.vectorize
def mexp(x):
    return np.exp(x)

@dataclass
class PFID_Fitter:
    ds: PolTRSpec
    params: lmfit.Parameters = field(default_factory=lmfit.Parameters)
    num_peaks: int = 0
    alpha: float = 0


    def start_fit(self):
        if 't0' not in self.params:
            self.params.add("t0", 0, vary=False)
        mini = lmfit.Minimizer(self.eval, self.params.copy())
        
        fr = mini.least_squares(diff_step=0.001)
        print(fr)
        # self.params = fr.params
        fr.minimizer = mini
        return fr

    def add_pfid(
        self, A: float, x0: float, T2: float, angle: float, B: float, shift: float
    ):
        i = self.num_peaks
        items = zip("A x0 T2 angle B shift".split(" "), (A, x0, T2, angle, B, shift))
        for name, val in items:
            maxval, minval = np.inf, -np.inf
            if name == "T2":
                minval = 0.3
                maxval = 2
            if name == "angle":
                minval = 0
                maxval = 90
            if name == 'B':
                minval = 0
                maxval = 1
            if name == 'A':
                maxval = 0
            self.params.add(f"{name}_{i}", val, min=minval, max=maxval)
        
        self.num_peaks += 1


    def eval(self, params=None, residual=True, t=None, wn=None):
        # print(params.values())
        if params is None:
            params=self.params

        if t is None:
            t = self.ds.t
        if 't0' in params:
            t = t-params['t0'].value
        if wn is None:
            wn = self.ds.wavenumbers
        vals = np.array(list(params.valuesdict().values()), dtype="f")
        vals = vals[:-1].reshape(self.num_peaks, 6)

        dshape = self.ds.iso.data.shape
        out_pa = np.zeros((*dshape, self.num_peaks))
        out_pe = np.zeros_like(self.ds.iso.data)
        alpha = 0

        #for i in vals:
        #    A, x0, T2, angle, B, shift = i
        
        A, x0, T2, angle, B, shift = vals.T
        B = -B*A
        dichro = angle2dichro(angle)
        pe = A * pfid_r4(-t, wn, x0, T2)
        pe += B * pfid_r6(-t, wn, x0, shift, T2)
        pa = dichro*pe
        
        out_pe = pe.sum(-1)
        out_pa = pa.sum(-1)
        alpha = np.sum(A * A) + np.sum(B * B)
        if residual:
            out_pa -= self.ds.para.data
            out_pe -= self.ds.perp.data
            out = np.hstack((out_pa, out_pe))
            if self.alpha > 0:
                out = np.hstack((out.ravel(),  alpha / self.num_peaks * self.alpha))
            return out
        else:
            return pa, pe