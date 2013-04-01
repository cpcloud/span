Getting Started
===============

Caveat Emptor
-------------
I developed this code with the bleeding edge versions of everything
listed below, so don't be surprised if you try this out on an older
version of NumPy (for example) and it doesn't work. When in doubt set
up a `virtualenv <http://www.virtualenv.org>`_ and install the git
versions of all the dependencies. This code has only been used and
tested on `Arch Linux <http://archlinux.org>`_ and `Red Hat Enterprise
Linux 5 <http://www.redhat.com/products/enterprise-linux/>`_. If you set up a
virtualenv and follow the install instructions there should be no problems.

Dependencies
------------
1. `Python <http://www.python.org>`_ >= 2.6
2. `NumPy <http://numpy.scipy.org>`_
3. `SciPy <http://numpy.scipy.org>`_
4. `Cython <http://www.cython.org>`_
5. `pandas <http://pandas.pydata.org>`_
6. `six <http://pythonhosted.org/six>`_

Optional Dependencies
---------------------
1. `nose <http://nose.readthedocs.org>`_ if you want to run the tests
2. `sphinx <http://www.sphinx-doc.org>`_ if you want to build the documentation
3. `numpydoc <https://pypi.python.org/pypi/numpydoc>`_ if you want to build the
   documentation with NumPy formatting support (recommended)

Installation
------------
You should set up a `virtualenv <http://www.virtualenv.org>`_ when
using this code, so that you don't break anything. Personally, I prefer the
excellent `virtualenvwrapper <http://virtualenvwrapper.readthedocs.org/en/latest/index.html>`_
tool to set up Python environments.

First things first:

.. code-block:: bash

    # change to span's directory
    cd wherever/you/downloaded/span

    # install dependencies
    pip install -r requirements.txt


.. code-block:: bash

    # install span
    python setup.py install

or if you don't want to install it and you want to use ``span`` from its 
directory then you can do

.. code-block:: bash

    python setup.py build_ext --inplace
    python setup.py build

For the last one you must set the ``PYTHONPATH`` environment variable or mess
around with ``sys.path`` (not recommended).
