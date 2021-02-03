"""
Comparment Modelling
====================

Here in this tutorial, we will use skultrafast to do global analysis. We are
using the approach as presented by van Stokkum
(doi:10.1016/j.bbabio.2004.04.01).

The general outline is as follows:

    1. Generate a decay associated spectrum
    2. Build the transfer matrix K, choosing starting concentrations j
    3. Generate transformation matricies from K and j
    4. Apply them to the basis vectors of the DAS

Here we assume step 1 is already done. skultrafast has a module to help with step 2.
For that we need the `Model` class. Please note that the module is quite barebones
and may still have bugs.
"""
# %%
import numpy as np
from matplotlib import pyplot as plt 
from skultrafast.kinetic_model import Model

model = Model()

# %%
# To build the transfer matrix, we have to add transitions. The if one the
# compartments is not part of the model it will be added automatically. The
# following line means that the model has an transition from S2 to S1 with a
# rate k2. Since no yield is given, it defaults to 1

model.add_transition('S2', 'S1', 'k2')

# %%
# Lets also assume, that S1 decays to zero. The comparment zero is special,
# since it is not modelled explicty and the name `zero` is reserved.

model.add_transition('S1', 'zero', 'k1')

# %%
# To get the corresponding matrix from the model we have to call the
# `build_matrix`-function

mat = model.build_matrix()
mat

# %%
# To fill the matrix with numeric values we can use the `build_matfunc` method.
# This gives us a function which takes all free parameters, first the rates and
# then the yields.

f = model.build_mat_func()
f

# %%
# Now, what can we do with that matrix? We can use it to project the DAS to the
# described model. For that, we first evaluate the matrix nummerically by
# supplying values for the parameters

num_mat = f(k2=2, k1=1)
num_mat

# %%
# Next, we need the eigenvectors of the matrix. Note, that the presented approach
# assumes that the eigenvalues are simple eigenvalues. If this is not the case,
# one has to use the jordan normal form. As we see, the eigenvalues of the 
# matrix are the negative rates. That basically means, that the eigenbasis of the
# problem is given by a diagonal transfer matrix, which is the parallel model
# described by an DAS. Hence, the eigenvectors allow us to transform the DAS to
# a SAS and vice versa by using the inverse.

vals, vecs = np.linalg.eig(num_mat)
vals

# %%
# To continue, we also need to choose the starting concentrations.

j = np.zeros(len(vals))
j[0] = 1

j
# %%
# The transformation matrix is then given by

A = vecs @ np.diag(np.linalg.inv(vecs) @ j)
A_inv = np.linalg.inv(A)

# %%
# Multiplying and DAS by `DAS @ A` with A should give the SAS, while `A_inv @
# basis_vecs` should return the time-depedence of the concentrations.


