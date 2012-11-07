==========================
Cross-correlation analysis
==========================
Roughly, cross-correlation is useful for determining the lag at which the maximum
covariance (correlation) between two processes lies.

Formally, it is

.. math::
    \rho\left(\ell\right) = \frac{E\left[U_{0}V_{\ell}\right] -
    E\left[U_{0}\right]E\left[V_{0}\right]}{\sqrt{\mathrm{var}\left[U_{0}\right]\mathrm{var}\left[V_{0}\right]}}

where :math:`U_{k}` and :math:`V_{k}` are binary time series and
:math:`k,\ell\in\mathbb{Z}`. This module provides functions to compute
:math:`\rho\left(\ell\right)` in an efficient manner.


--------------
``span.xcorr``
--------------

.. automodule:: span.xcorr.xcorr
   :members:
