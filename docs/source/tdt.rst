Reading in TDT Files
====================

.. _NumPy: http://numpy.scipy.org
.. _pandas: http://pandas.pydata.org
.. _dtype:
.. http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
.. _dtypes:
.. http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
.. _DataFrame:
.. http://pandas.pydata.org/pandas-docs/dev/dsintro.html#dataframe
.. _Cython: http://www.cython.org
.. _fused types: http://docs.cython.org/src/userguide/fusedtypes.html
.. _MultiIndex: http://pandas.pydata.org/pandas-docs/dev/dsintro.html#multindex

TDT File Structure
------------------
There are two types of TDT files necessary to create an instance of
:class:`span.tdt.spikedataframe.SpikeDataFrame`: one file ending in "tev" and
one ending in "tsq". Note that this differs slightly from TDT's definition of a
tank.

TSQ Event Headers
-----------------
The TSQ file is a C ``struct`` making it trivial to work with in `NumPy`_ using
a compound `dtype`_.

According to `Jaewon Hwang
<http://jaewon.mine.nu/jaewon/2010/10/04/how-to-import-tdt-tank-into-matlab/>`_,
the C struct is

.. code-block:: c

    struct TsqEventHeader {
        long size;
        long type;
        long name;
        unsigned short chan;
        unsigned short sortcode;
        double timestamp;
        union {
            __int64 fp_loc;
            double strobe;
        };

        long format;
        float frequency;
    };

*but* this code **will not work** on most modern systems because ``long`` is
implementation defined--the compiler writer defines it. I have not run across a
compiler on a 64 bit system that defines ``sizeof(long)`` to be 32. Thus the
most accurate version (and the one used in **span**) is

.. code-block:: c

    #include <stdint.h>

    struct TsqEventHeader {
        int32_t size;
        int32_t type;
        int32_t name;

        uint16_t chan;
        uint16_t sortcode;

        double timestamp;

        union {
            int64_t fp_loc;
            double strobe;
        };

        int32_t format;
        float frequency;
    };


.. warning::

   If you're using this code on data that were created on a Windows 7
   machine then may have to change ``int32_t`` to ``int64_t``. I have not
   tested this code on data created on a Windows 7 machine so **use at your own
   risk**.


Reading the TSQ file into `NumPy`_ is, fortunately, very easy now that we have
this ``struct``.


.. code-block:: python

    import numpy as np
    from pandas import DataFrame
    from numpy import int32, uint32, uint16, float64, int64, int32, float32

    names = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'fp_loc', 'strobe', 'format', 'fs')
    formats = (int32, int32, uint32, uint16, uint16, float64, int64,
               float64, int32, float32)
    offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
    tsq_dtype = np.dtype({'names': names, 'formats': formats,
                          'offsets': offsets}, align=True)
    tsq_name = 'name/of/file.tsq'
    tsq = np.fromfile(tsq_name, dtype=tsq_dtype)
    df = DataFrame(tsq)


The variable ``tsq`` in the above code snippet is a `NumPy record array
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_. I personally find 
these very annoying. Luckily, `Wes McKinney <http://www.wesmckinney.com>`_
created the wonderful `pandas`_ library which automatically converts `NumPy`_
record arrays into a `pandas`_ `DataFrame`_ where each field from the record
array is now a column in the `DataFrame`_ ``df``.

TL;DR (too long; don't read)
============================
Reading in the Raw Data
-----------------------
Now that we've got the header data we can get what we're really
interested in: raw voltage traces. There are some indexing acrobatics
here that require a little bit of detail about the tsq file and little
bit of knowledge of "group by" style operations.

First off, there is a Cython function that does all of the heavy
lifting in terms of reading raw bytes into a NumPy array. What is
passed in to that function is important.

The first argument is of course the filename, no surprise there. The
second argument is important. This is the numpy array of file
locations grouped by channel number. This is an array that contains
the file pointer location of each consective chunk of data in the TEV
file. That means that if, for example, I want to read all of the data
from channel 1 then I would loop over the first column of this array.
Since each element is a file pointer location I would seek to that
location and read ``blocksize`` bytes. The Cython function does this
automatically for every channel. The third argument is ``blocksize``
and the fourth argument is the output array that contains the raw
voltage data.

Here is the inner loop that does the work of reading in the raw data
from the tev file.

.. literalinclude:: ../../span/tdt/read_tev.pyx
   :language: cython
   :lines: 21-


You can see here that this part of the
:py:func:`span.tdt._read_tev._read_tev_raw` function skips to the point in
the file where the next chunk lies and placing it in the array
``spikes``. This codes works on any kind of floating point spike data (by used
`fused types`_ and it also runs in parallel for a slight speedup in I/O.

As usual, the best way to understand what's going on is to read the
source code.

Organizing the Data
-------------------
Whew! Reading in these data are tricky.

Now we have a dataset. However it's not properly arranged, meaning
the dimensions are not those that make sense from the point of
analysis.

I'm not exactly sure how this works, but TDT stores their data in
chunks and that chunk size is usually a power of 2.

The number of chunks depends on the length of the recording and is the
number of rows in the TSQ array. So, ``tsq.shape[0]`` equals the
number of chunks in the recording.

Now, each chunk has a few properties, which you can explore on your
own if you're interested. For now, we'll only concern ourselves with
the ``channel`` (``chan`` in the C ``struct``) column.

The ``channel`` column gives each chunk a ... you guessed it ...
channel, and thus provides a way to map sample chunks to channels.

Electrode Configuration
-----------------------
I'm currently working on a flexible implementation to allow for
arbitrary, but within physical reason, electrode array configuration.
Stay tuned! What's currently available is in the
:mod:`span.tdt.recording` module.


``span.tdt.tank``
-----------------

.. automodule:: span.tdt.tank
   :show-inheritance:
   :members:


``span.tdt.spikedataframe``
---------------------------

.. automodule:: span.tdt.spikedataframe
   :show-inheritance:
   :members:
