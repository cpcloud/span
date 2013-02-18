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
    df = DataFrame.from_records(tsq)


``tsq`` is a `NumPy record array
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_. I personally
find these very annoying. Luckily, `Wes McKinney
<http://www.wesmckinney.com>`_ created the wonderful `pandas`_ library
which automatically converts `NumPy`_ record arrays into a pandas
`DataFrame`_ where each field from the record array is now a column in
the `DataFrame`_ ``df``.

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
:py:func:`span.tdt._read_tev._read_tev` function skips to the point in
the file where the next chunk lies and placing it in the array
``spikes``.

As usual, the best way to understand what's going on is to read the
source.

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
Stay tuned! What's currently available is in the
:mod:`span.tdt.recording` module.


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
