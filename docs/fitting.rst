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
sympy, the calculation is done in `convolution.ipynb`_ notebook. Therefore,
by default *skultrafast* fits the function

:math:`y(t, \omega)= A \exp(\frac{-t}{\tau_i}+\frac{\sigma^2}{2\tau_i^2})
       \frac{1}{2} erfc(\frac{\sigma}{\sqrt 2 \tau_i}-\frac{t}{\sqrt 2\sigma})`.

For given :math:`\tau` and :math:`\omega`, the least squares problem is
linear since the function is a sum where only the coefficents are missing.




