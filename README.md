Python TDT
===

Python module to read Tucker-Davis Technologies TTank files

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
* [Nose](http://nose.readthedocs.org/en/latest) >= 1.1.3 if you want to run tests

Purpose
---
I need to read large, proprietary files quickly into `ndarrays` and do all sorts
of analysis, thus I created this package to help me do that.

Currently, it's very "hacky" and thus the API is very unstable. I
wholeheartedly welcome any contributions!

I have another project `span` which is more general than this module,
and I would eventually like to integrate this package into `span`.
