=================
Cross-correlation
=================

Roughly, cross-correlation is useful for determining the lag at which
the maximum correlation between two processes lies.

Formally, it is

    .. math::
        \rho\left(\ell\right) = \frac{E\left[U_{0}V_{\ell}\right] -
        E\left[U_{0}\right]E\left[V_{0}\right]}{\operatorname{std}\left(U_{0}\right)
        \operatorname{std}\left(V_{0}\right)}

where :math:`U_{k}` and :math:`V_{k}` are binary time series and
:math:`k,\ell\in\mathbb{Z}` [1]_. This module provides functions to compute
:math:`\rho\left(\ell\right)` in an efficient manner by using the
identity

    .. math::
        \operatorname{xcorr}\left(\mathbf{x}, \mathbf{y}\right) =
        \operatorname{ifft}\left(\operatorname{fft}\left(\mathbf{x}\right)
        \cdot\operatorname{fft}\left(\mathbf{y}\right)^{*}\right)

where :math:`\operatorname{ifft}\left(\mathbf{x}\right)` is the
inverse Fourier transform of a vector,
:math:`\operatorname{fft}\left(\mathbf{x}\right)` is the Fourier
transform of a vector, :math:`\mathbf{x}` and :math:`\mathbf{y}` are
vectors, and :math:`*` is the complex conjugate.

.. [1] Amarasingham et. al (2012), in press

See any signal processing text for a presentation of cross-correlation
in a more general setting.


--------------
``span.xcorr``
--------------

.. automodule:: span.xcorr.xcorr
   :members:
