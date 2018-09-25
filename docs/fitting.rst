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

:math:`y(t, \omega)= \sum_i A(i, \omega) exp(-t/\tau_i) \textrm{for} t>0`

with a gaussian IRF

:math:`IRF(t) = \frac{1}{\sqrt{2 \pi \sigma}} \exp\left
(-\frac{t^2}{2\sigma^2}\right)`.

The result of the convolution :math:`y_{\textrm{conv}} = IRF \circledast y`
can be expressed in terms of the complementary errorfunction `erfc`.


