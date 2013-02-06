============
Full Example
============
Here is a worked example to get to a dataset that is ready for
analysis.

All of the following steps assume that you've executed ``import span``
in a Python interpreter.

---------------
Read a TDT file
---------------

.. code-block:: python

    import span
    tankname = 'some/path/to/a/tdt/tank/file'
    tank = span.tdt.PandasTank(tankname)
    sp = tank.spikes # spikes is a computed property that is cached

------------------
Threshold the data
------------------

.. code-block:: python

    # create an array of bools indicating which spikes have voltage values
    # greater than 4 standard deviations
    thr = sp.threshold(4 * sp.std())

---------------------------
Clear the refractory period
---------------------------

.. code-block:: python

    # clear the refractory period of any spikes
    clr = sp.clear_refrac(thr)

------------
Bin the data
------------

.. code-block:: python

    # binned the data in 1000 millisecond bins
    binned = sp.bin(clr, binsize=1000)

---------------
Full Code Block
---------------

.. code-block:: python

    import span
    tankname = 'some/path/to/a/tdt/tank/file'
    tank = span.tdt.PandasTank(tankname)
    sp = tank.spikes

    # create an array of bools indicating which spikes have voltage values
    # greater than 4 standard deviations
    thr = sp.threshold(4 * sp.std())

    # clear the refractory period of any spikes
    clr = sp.clear_refrac(thr)

    # binned the data in 1000 millisecond bins
    binned = sp.bin(clr, binsize=1000)
