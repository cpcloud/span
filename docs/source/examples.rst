============
Full Example
============
Here is a worked example to get to a dataset that is ready for
analysis.

All of the following steps assume that you've executed ``import span``
in a Python interpreter.

------------------
1. Read a TDT file
------------------

.. code-block:: python

    import span
    tankname = 'some/path/to/a/tdt/tank/file' [#f1]_
    tank = span.tdt.PandasTank(tankname)
    sp = tank.spik # spikes is a computed property based on the names of events

---------------------
2. Threshold the data
---------------------

.. code-block:: python

    # create an array of bools indicating which spikes have voltage values
    # greater than 4 standard deviations from the mean
    thr = sp.threshold(4 * sp.std())

------------------------------
3. Clear the refractory period
------------------------------

.. code-block:: python

    # clear the refractory period of any spikes; in place to save memory
    thr.clear_refrac(inplace=True)

---------------
4. Bin the data
---------------

.. code-block:: python

    # bin the data in 1 second bins
    binned = clr.resample('S', how='sum')

--------------------------------
5. Compute the cross correlation
--------------------------------

.. code-block:: python

    # compute the cross-correlation of all channels
    # note that there are a lot more options to this method
    # you should explore the docs
    xcorr = sp.xcorr(binned)

---------------
Full Code Block
---------------

.. code-block:: python

    import span
    tankname = 'some/path/to/a/tdt/tank/file'
    tank = span.tdt.TdtTank(tankname)
    sp = tank.spik

    # create an array of bools indicating which spikes have voltage values
    # greater than 4 standard deviations
    thr = sp.threshold(4 * sp.std())

    # clear the refractory period of any spikes
    thr.clear_refrac(inplace=True)

    # binned the data in 1 second bins
    binned = clr.resample('S', how='sum')

    # compute the cross-correlation of all channels
    xcorr = sp.xcorr(binned)
