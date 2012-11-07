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

According to Jaewon at :link:`jaewon.mine.nu` the C struct is

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

After this the data are thrown into a ``DataFrame`` for easier processing.

Reading in the Raw Data
-----------------------
Now that we've got the header data we can get what we're really
interested in: raw voltage traces.

To do this, we really only two of

Exported Classes
----------------

.. autoclass:: span.tdt.TdtTankAbstractBase
   :members:

.. autoclass:: span.tdt.TdtTankBase
   :members:

.. autoclass:: span.tdt.PandasTank
   :members:
