__author__ = 'Tillsten'


import lmfit

class Min(lmfit.Minimizer):
    def prepare_fit(self, params=None):
        lmfit.Minimizer(self, params)
        if hasattr(self, 'dof_modifier'):
            self.nfree = self.nfree - self.dof_modifier

    def __residual(self, fvars):
        """
        residual function used for least-squares fit.
        With the new, candidate values of fvars (the fitting variables),
        this evaluates all parameters, including setting bounds and
        evaluating constraints, and then passes those to the
        user-supplied function to calculate the residual.
        """

        # set parameter values
        for varname, val in zip(self.var_map, fvars):
            par = self.params[varname]
            par.value = par.from_internal(val)
            if hasattr(self, para_transform):
                par.value = para_transform[varname](par.value)
        self.nfev = self.nfev + 1
        self.update_constraints()
        out = self.userfcn(self.params, *self.userargs, **self.userkws)
        if hasattr(self.iter_cb, '__call__'):
            self.iter_cb(self.params, self.nfev, out,
                         *self.userargs, **self.userkws)
        return out


