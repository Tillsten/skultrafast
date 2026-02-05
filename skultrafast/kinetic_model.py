# -*- coding: utf-8 -*-
"""
This module helps to build a transfer matrix and applying it to an
DAS.
"""

import numpy as np
import scipy.linalg as la
from typing import List
import numbers


class Transition(object):
    """
    Represents a transtion between comparments.
    """

    def __init__(self, from_comp, to_comp, rate=None, qy=None):
        if rate is None:
            self.rate = from_comp + "_" + to_comp
        else:
            self.rate = rate
        self.from_comp = from_comp
        self.to_comp = to_comp
        if qy is None:
            qy = 1
        self.qu_yield = qy


class Model(object):
    """
    Helper class to make a model
    """

    def __init__(self):
        self.transitions: List[Transition] = []

    def add_transition(self, from_comp, to_comp, rate=None, qy=None):
        """
        Adds an transition to the model.

        Parameters
        ----------
        from_comp : str
            Start of the transition
        to_comp : str
            Target of the transition
        rate : str, optional
            Name of the associated rate, by default None, which generates a
            default name.
        qy : str of float, optional
            The yield of the transition, by default 1
        """

        trans = Transition(from_comp, to_comp, rate, qy)
        self.transitions.append(trans)

    def build_matrix(self):
        """
        Builds the n x n k-matrix as a symbolic representation (list of lists of tuples).
        Each entry contains a list of (coefficient, rate_name, yield_value) tuples.
        """
        comp = get_comparments(self.transitions)
        idx_dict = dict(enumerate(comp))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))
        # Create a matrix where each entry is a list of terms
        mat = [[[] for _ in range(len(comp))] for _ in range(len(comp))]

        for t in self.transitions:
            i = inv_idx[t.from_comp]
            # Add -rate * qu_yield to diagonal
            mat[i][i].append((-1, t.rate, t.qu_yield))
            if t.to_comp != "zero":
                # Add +rate * qu_yield to off-diagonal
                mat[inv_idx[t.to_comp]][i].append((1, t.rate, t.qu_yield))

        self.mat = mat
        return mat

    def build_mat_func(self):
        """
        Creates a function that builds the K matrix from parameter values.
        Uses fully vectorized numpy operations for maximum speed.
        """
        # Pre-compute everything once
        rates = list({t.rate: None for t in self.transitions}.keys())
        yields = [
            t.qu_yield
            for t in self.transitions
            if not isinstance(t.qu_yield, numbers.Number)
        ]
        params = rates + yields
        param_to_idx = {p: i for i, p in enumerate(params)}

        # Pre-compute matrix template and compartments
        mat_template = self.build_matrix()
        comp = get_comparments(self.transitions)
        n_comp = len(comp)

        # Pre-compile matrix structure into numpy arrays
        indices_i = []
        indices_j = []
        coeffs = []
        rate_indices = []
        yield_indices = []  # -1 for numeric, >= 0 for param index
        yield_numeric = []  # numeric yield values

        for i in range(n_comp):
            for j in range(n_comp):
                for coeff, rate_name, yield_val in mat_template[i][j]:
                    indices_i.append(i)
                    indices_j.append(j)
                    coeffs.append(coeff)
                    rate_indices.append(param_to_idx.get(rate_name, -1))

                    # Separate numeric from parametric yields
                    if isinstance(yield_val, str):
                        yield_indices.append(param_to_idx.get(yield_val, -1))
                        yield_numeric.append(1.0)
                    else:
                        yield_indices.append(-1)
                        yield_numeric.append(float(yield_val))

        # Convert to numpy arrays
        indices_i = np.array(indices_i, dtype=np.int32)
        indices_j = np.array(indices_j, dtype=np.int32)
        coeffs = np.array(coeffs, dtype=np.float64)
        rate_indices = np.array(rate_indices, dtype=np.int32)
        yield_indices = np.array(yield_indices, dtype=np.int32)
        yield_numeric = np.array(yield_numeric, dtype=np.float64)

        n_terms = len(coeffs)

        def K_func(*args, **kwargs):
            # Convert args to numpy array
            param_values = np.array(args, dtype=np.float64)

            # Update with any keyword arguments
            if kwargs:
                param_dict = {p: v for p, v in zip(params, args)}
                param_dict.update(kwargs)
                param_values = np.array(
                    [
                        param_dict.get(p, args[i] if i < len(args) else 0)
                        for i, p in enumerate(params)
                    ],
                    dtype=np.float64,
                )

            # Vectorized computation: get all rate values
            rate_vals = param_values[rate_indices]

            # Vectorized computation: get all yield values
            yield_vals = yield_numeric.copy()
            param_yield_mask = yield_indices >= 0
            yield_vals[param_yield_mask] = param_values[yield_indices[param_yield_mask]]

            # Compute all contributions
            contributions = coeffs * rate_vals * yield_vals

            # Accumulate into K matrix using fancy indexing
            K = np.zeros((n_comp, n_comp), dtype=np.float64)
            np.add.at(K, (indices_i, indices_j), contributions)

            return K

        return K_func

    def get_compartments(self):
        return get_comparments(self.transitions)

    def make_diff_equation(self):
        """
        Creates a representation of the differential equations.
        Note: Full symbolic differential equation support removed with sympy dependency.
        """
        A = self.build_matrix()
        comp = self.get_compartments()
        eqs = []
        for i, comp_name in enumerate(comp):
            # Represent the equation as: d{comp_name}/dt = sum of rates for that compartment
            row_sum = sum(A[i].values())
            eqs.append(f"d{comp_name}/dt = {row_sum}")
        return eqs

    def get_trans(self, y0, taus, t):
        """
        Return the solution using vectorized matrix exponential computation.
        """
        # Build the numeric matrix by substituting tau values
        comp = get_comparments(self.transitions)
        n_comp = len(comp)
        K = np.zeros((n_comp, n_comp))
        idx_dict = {c: i for i, c in enumerate(comp)}

        # Pre-convert taus to array
        taus_array = np.asarray(taus, dtype=np.float64)

        for trans in self.transitions:
            i = idx_dict[trans.from_comp]
            rate_idx = self.transitions.index(trans)
            rate_val = (
                taus_array[rate_idx] if hasattr(taus, "__getitem__") else taus_array
            )
            yield_val = (
                trans.qu_yield if isinstance(trans.qu_yield, numbers.Number) else 1.0
            )

            K[i, i] -= rate_val * yield_val
            if trans.to_comp != "zero":
                j = idx_dict[trans.to_comp]
                K[j, i] += rate_val * yield_val

        # Pre-convert t to array and compute exponentials once
        t_array = np.asarray(t, dtype=np.float64)
        n_t = len(t_array)
        o = np.zeros((n_t, n_comp))

        # Vectorized computation of matrix exponentials
        for i in range(n_t):
            o[i, :] = la.expm(K * t_array[i]).dot(y0)[:, 0]

        return o


def get_comparments(list_trans):
    """
    Getting a list of transtions, return the compartments
    """
    l = []
    for trans in list_trans:
        if trans.from_comp not in l:
            l.append(trans.from_comp)
        if trans.to_comp not in l and trans.to_comp != "zero":
            l.append(trans.to_comp)
    return l


def get_symbols(list_trans):
    """
    Return the used symbols
    """
    return [t.rate for t in list_trans]
