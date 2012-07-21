#!/usr/bin/env python

import threading
import Queue
import multiprocessing
import warnings

from numpy import vstack
import tables


class ProcessorCountWarning(UserWarning):
    pass


def load_data(f, where=None, raw_name='raw',
              num_threads=multiprocessing.cpu_count()):
    """Load an hdf5 file using the threading API.

    Naive testing indicates an order of magnitude speed up in loading
    HDF5 files in this manner.

    Parameters
    ----------
    f : tables.file.File
        An open-for-reading HDF5 file
    where : tables.file.Node, optional
        The parent node of the raw data arrays.
    raw_name : str, optional
        Name of the node containing the instance of tables.array.Array
    num_threads : int, optional
        Number of threads used to open the file. Defaults to the number of
        cores on the machine.

    Returns
    -------

    """
    assert isinstance(num_threads, (int, long)), \
        '"num_threads" must be an integer'
    assert f.isopen, 'f must be open'
    assert f.mode != 'w', 'f must be readable'
    assert hasattr(f, 'root'), 'f must have a "root" node'
    if where is None:
        where = f.root.data
    assert all(hasattr(node, raw_name) for node in f.iterNodes(where)), \
        'all nodes (shanks) must have a "%s" attribute' % raw_name
    assert all(isinstance(getattr(node, raw_name), tables.array.Array)
               for node in f.iterNodes(where)), \
               ("all '%s' attributes must be instances of tables.array.Array" %
                raw_name)

    num_cores = multiprocessing.cpu_count()
    if num_threads != num_cores:
        warnings.warn('You have %i cores, but you requested %i threads' %
                      (num_cores, num_threads), ProcessorCountWarning)

    def set_local():
        while not input_queue.empty():
            raw, i = input_queue.get()
            output_queue.put((raw.read(), i))
            input_queue.task_done()

    input_queue = Queue.Queue()
    output_queue = Queue.Queue()
    for i, shank in enumerate(f.iterNodes(where)):
        input_queue.put((getattr(shank, raw_name), i))

    threads = (threading.Thread(target=set_local) for _ in xrange(num_threads))
    for thread in threads:
        thread.daemon = True
        thread.start()

    input_queue.join()
    out = []
    while not output_queue.empty():
        out.append(output_queue.get())
    out.sort(key=lambda x: x[-1])
    return vstack(item[0] for item in out)
