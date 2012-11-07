Reading in TDT Files
====================

TDT File Structure
------------------
There are two types of TDT files necessary to create an instance of
``SpikeDataFrame``: one file ending in "tev" and one ending in "tsq".

TSQ Event Headers
-----------------
The TSQ file is fundamentally a C ``struct`` making it almost trivial
to work with in ``numpy`` using compound ``dtype`` descriptors.

According to `Jaewon Hwang <http://jaewon.mine.nu/jaewon/2010/10/04/how-to-import-tdt-tank-into-matlab/>`_, the C struct is

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

What's great about numpy and the fact that the entire file is
basically just an array of C ``struct`` is that you can read in the entire tsq file in just a few lines of code:

.. code-block:: python

    from future_builtins import zip
    import numpy as np
    from numpy import int32, uint32, uint16, float64, int64, int32, float32

    fields = 'size', 'type', 'name', 'channel', 'sort_code', 'timestamp', 'fp_loc', 'format', 'fs'
    np_types = int32, int32, uint32, uint16, uint16, float64, int64, int32, float32
    tsq_dtype = np.dtype(list(zip(fields, np_types)))
    tsq_name = 'name/of/file.tsq'
    tsq = np.fromfile(tsq_name, dtype=tsq_dtype)


The caveat here is that most operating systems from about 4 years ago
onward define ``long`` as a 64-bit integer, but as you see in the above
code block ``size`` is mapped to a signed 32-bit integer. A better way
to define the C ``struct`` above is to use the ``typedef`` s in
``stdint.h`` for maximum portability, but I digress.

The reason why ``long`` is 32-bit is because these data were recorded on a machine
running Windows XP which has ``INT_MAX`` as :math:`2^{31} - 1`.

After this the data are thrown into a ``DataFrame`` for easier
processing. See `pandas <http://pandas.pydata.org>`_ for more details
on ``DataFrame``.


Reading in the Raw Data
-----------------------
Now that we've got the header data we can get what we're really
interested in: raw voltage traces.

To do this, we really only need two of the attributes of ``tsq``:
``fp_loc`` and ``chan``.

The ``fp_loc`` attribute provides an array of integers with the
location of a chunk of data in the TEV file. So if we were to loop
through ``fp_loc`` we can read each chunk of data into a NumPy array.
This exactly what is done in the code.

Here is the inner loop that does the work of reading in the raw data
from the TEV file.

.. literalinclude:: ../../span/tdt/read_tev.pyx
   :language: cython
   :linenos:
   :lines: 53,55,58,61-62


You can see here that this part of the ``read_tev`` function is
skipping to the point in the file where the next chunk lies and
placing it in the array ``spikes``.

As usual the best way to understand what's going on is to `read the
source`!

-------------------
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

Let :math:`n` be the number of samples, :math:`m` be the number of
chunks, :math:`k` be the number of channels,
:math:`c\in\left\{1,\ldots,n/k\right\}` be the number of times a
particular channel has been seen.


-----------------
``span.tdt.tank``
-----------------

.. automodule:: span.tdt.tank
   :members:


---------------------------
``span.tdt.spikedataframe``
---------------------------

.. automodule:: span.tdt.spikedataframe
   :members:
