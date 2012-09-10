import re
import glob
import os

import numpy as np

from pandas import DataFrame, HDFStore


folder = os.path.expanduser('~/analysis/data/112311_p14_rat_9_2_spikes_yes_24414.1/')
text_files = glob.glob('%s*.txt' % folder)
pattern = re.compile(r'.*[cC][hH](\d{1,2}).*')
text_files.sort(key=lambda x: int(pattern.match(x).group(1)))

data = []
for i, text_file in enumerate(text_files, start=1):
    npy = np.loadtxt(text_file)
    data.append(npy)
    np.save(str(i), npy)
    os.remove(text_file)

data = np.asarray(data)

m = np.asarray([[1, 2, 3, 6],
                [4, 5, 7, 8],
                [9, 10, 12, 13],
                [14, 11, 16, 15]]) - 1

store = HDFStore('%s/npy/store.h5' % folder, complevel=9)
for i, mhat in enumerate(m, start=1):
    store['sh%i' % i] = DataFrame(data[mhat, :].T,
                                  columns=(mhat + 1).astype(np.int32).tolist())
store.flush()
store.close()
