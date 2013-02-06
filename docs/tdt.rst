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
:class:`span.tdt.spikedataframe.SpikeDataFrame`: one file ending in "tev" and one ending in "tsq".

TSQ Event Headers
-----------------
The TSQ file is fundamentally a C ``struct`` making it almost trivial
to work with in `NumPy`_ using a compound `dtype`_.

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


One way to read this in C (including ugly error handling) would be:

.. code-block:: c

    #include <stdlib.h>
    #include <stdint.h>
    #include <string.h>
    #include <stdio.h>
    #include <errno.h>
    #include <sys/stat.h>


    typedef struct {
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
    } TsqEventHeader;


    ssize_t fsize(const char* filename)
    {
        struct stat st;

        if (stat(filename, &st) == 0)
            return st.st_size;

        (void) fprintf(stderr, "Cannot determine size of %s: %s\n", filename,
                       strerror(errno));

        return -1;
    }


    void* file_error(const char* filename, const char* msg)
    {
        (void) fprintf(stderr, msg, strerror(errno));
        return NULL;
    }


    TsqEventHeader* read_tsq(const char* filename)
    {
        FILE* f = fopen(filename, "rb");

        if (f == NULL)
            return file_error(filename, "Cannot open file %s, ERR: %s\n");


        ssize_t header_size = sizeof(TsqEventHeader);
        ssize_t filesize = fsize(filename);
        ssize_t nstructs = filesize / header_size;

        TsqEventHeader* header = (TsqEventHeader*) malloc(filesize);

        if (header == NULL) {
            fprintf(stderr, "Out of memory: ERR: %s\n", strerror(errno));

            if (fclose(f) != 0)
                return file_error(filename, "Cannot close file %s: %s\n");

            f = NULL;
            return NULL;
        }

        size_t bytes_read = fread(header, header_size, nstructs, f);

        if (!bytes_read) {
            free(header);
            header = NULL;

            return file_error(filename, "Read 0 bytes from file %s: %s\n");
        }

        if (fclose(f) != 0) {
            free(header);
            header = NULL;

            return file_error(filename, "Cannot close file %s: %s\n");
        }

        f = NULL;

        return header;
    }


Reading the TSQ file into `NumPy`_ is, fortunately, **much** easier than this.


.. code-block:: python

    import numpy as np
    import pandas as pd
    from numpy import int32, uint32, uint16, float64, int64, int32, float32

    fields = 'size', 'type', 'name', 'channel', 'sort_code', 'timestamp', 'fp_loc', 'format', 'fs'
    np_types = int32, int32, uint32, uint16, uint16, float64, int64, int32, float32
    tsq_dtype = np.dtype(zip(fields, np_types))
    tsq_name = 'name/of/file.tsq'
    tsq = np.fromfile(tsq_name, dtype=tsq_dtype)
    df_tsq = pd.DataFrame(tsq)


``tsq`` is a `NumPy record array
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_. I personally
find them very annoying. Luckily, `Wes McKinney
<http://www.wesmckinney.com>`_ created the wonderful `pandas`_ library
which automagically converts `NumPy`_ record arrays into a pandas `DataFrame`_ where
each field from the record array is now a column in the `DataFrame`_.
Great.


The caveat here is that most operating systems from about 4 years ago
onward define ``long`` as a 64-bit integer, but as you see in the above
code block ``size`` is mapped to a signed 32-bit integer. A better way
to define the C ``struct`` above is to use the ``typedef`` s in
``stdint.h`` for maximum portability, but I digress.

The reason why ``long`` is 32-bit is because these data were recorded on a machine
running Windows XP which has ``INT_MAX`` as :math:`2^{31} - 1` because
of `2's complement <http://en.wikipedia.org/wiki/Two's_complement>`_.

After this the data are thrown into a `DataFrame`_ for easier
processing.


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

.. literalinclude:: ../span/tdt/read_tev.pyx
   :language: cython
   :lines: 21-


You can see here that this part of the :py:func:`span.tdt._read_tev._read_tev`
function skips to the point in the file where the next chunk lies and
placing it in the array ``spikes``. A possible improvement on the
`Cython`_ code here is to use `fused types`_ to allow for floating
arrays that use either 32 or 64-bit representations.

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

-----------------------
Electrode Configuration
-----------------------
I'm currently working on a flexible implementation to allow for
arbitrary, but within physical reason, electrode array configuration.
Stay tuned! What's currently available is in the :mod:`span.tdt.recording` module.


-----------------
``span.tdt.tank``
-----------------

.. automodule:: span.tdt.tank
   :show-inheritance:
   :members:


---------------------------
``span.tdt.spikedataframe``
---------------------------

.. automodule:: span.tdt.spikedataframe
   :show-inheritance:
   :members:
