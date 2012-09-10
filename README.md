`tdt`
===
is a Python module to read [Tucker-Davis Technologies](http://www.tdt.com)
[TTank](http://jaewon.mine.nu/jaewon/2010/10/04/how-to-import-tdt-tank-into-matlab)
files.

Dependencies
---
### Must haves:
* [Python](http://python.org) >= 3
* [NumPy](http://numpy.scipy.org) >= 1.6 
* [SciPy](http://scipy.org) >= 0.10 
* [matplotlib](http://matplotlib.sourceforge.net) >= 1.1.1
* [Pandas](http://pandas.pydata.org) >= 0.8.1
* [Clint](https://github.com/kennethreitz/clint)

### If you want:
* [Nose](http://nose.readthedocs.org/en/latest) >= 1.1.3 if you want
  to run tests
* [IPython](http://ipython.org) >= 0.11 How anyone uses Python without this boggles my mind

####Notes
---
* I need to read large, proprietary [TDT](http://www.tdt.com) files quickly into
  [`ndarrays`](http://docs.scipy.org/doc/numpy/reference/arrays.html).
* Currently, it's very "hacky" and thus the API is very unstable. I
  wholeheartedly welcome any contributions!
