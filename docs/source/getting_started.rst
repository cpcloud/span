Getting Started
===============

**Caveat Emptor**
-----------------
I developed this code with the bleeding edge versions of everything
listed below, so don't be surprised if you try this out on an older
version of NumPy (for example) and it doesn't work. When in doubt set
up a virtualenv and install the git versions of all the dependencies.

Dependencies
------------
1. `Python <http://www.python.org>`_ >= 2.6
2. `NumPy <http://numpy.scipy.org>`_
3. `SciPy <http://numpy.scipy.org>`_
4. `Cython <http://www.cython.org>`_
5. `pandas <http://pandas.pydata.org>`_

Optional Dependencies
---------------------
1. `nose <http://nose.readthedocs.org>`_ if you want to run the tests

Installation
------------
You should set up a virtualenv when using this code, so that you don't
break anything. Then you can do the usual

.. code-block:: python

    python setup.py install


or if you don't want to install it and you want to
use ``span`` from its directory then you can do

.. code-block:: python

    python setup.py build_ext --inplace
    python setup.py build
