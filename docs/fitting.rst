********
Appendix
********

Mathematical details of the fitting procedure
=============================================

This section will describe the algorithm used to do the exponential fit. It
mostly follows prior work, with only some small modifications to increase the
speed and the stability of the fitting.

The fitting function
--------------------
As its default, *skultrafast* assumes the traces have an gaussian IRF.
Therefore, the data is described by the convolution of a sum of one-sided
exponentials

:math:`y(t, \omega)= \sum_i A(i, \omega) exp(-t/\tau_i) \Theta(t)`,

with :math:`\Theta` being the Heaviside-function, and gaussian instrument
response function (IRF):

:math:`IRF(t) = \frac{1}{\sqrt{2 \pi \sigma}} \exp\left(-\frac{t^2}{2\sigma^2}\right)`.

The result of the convolution

:math:`y_{\textrm{conv}} = IRF \circledast y`

can be expressed in terms of the complementary error-function `erfc`. Using
sympy, the calculation is done in 'convolution.ipynb' notebook. Therefore,
by default *skultrafast* fits the function

.. math::

        y(t, \omega)= A \exp(\frac{-t}{\tau_i}+\frac{\sigma^2}{2\tau_i^2})\frac{1}{2} erfc(\frac{\sigma}{\sqrt 2 \tau_i}-\frac{t}{\sqrt 2\sigma})

.. toctree::
   :maxdepth: 1

   auto_examples/convolution


Variable projection
-------------------
For given :math:`\tau` and :math:`\omega`, the least squares problem is linear
since the function is a sum of term where only the coefficients are unknown.
Therefore we use our nonlinear functions as a basis matrix :math:`A_{ij} =
y(t_i, \tau_j)`. The linear least-squares problem can be written as :math:`min_x
|Ax-b|_2` and can be directly solved. The separation of the linear and
non-linear parameters is also know as variable projection.

Since the exponential function basis is numerically unstable, skultrafast uses
L2-regularization by default. This is also called Tikhonov regularization or
Rigde regression. It modifies the problem to :math:`min_x |Ax-b|_2+\alpha
|x|_2`, alpha being small.

Depending on how the dispersion is handled, we can accelerate these steps. First
we will assume that each frequency was interpolated or binned on the same common
time-points. Then we just have to calculate the matrix A and its pseudoinverse
once. The coefficients c for a single channel :math:`b` are than just the dot
product :math:`c = A_{PINV}b`.

If the different frequencies don't share a common time-axis, the matrix A has
and its pesudoinverse has to be calculated for every channel, which gets
time-consuming for larger datasets.

The advantage of the latter approach is that it allows for easier inclusion of
the dispersion parameters to the fitting model.


