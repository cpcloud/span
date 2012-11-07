============
Introduction
============

----------
Motivation
----------
``span`` arose out of the need to do the following operations in a
reasonably efficient manner:

* Read in TDT files into `NumPy <http://numpy.scipy.org>`_ arrays.
* Group data arbitrarily for firing rate analyses, e.g., average
  firing rate over shanks, collapsing across channels.
* Perform cross-correlation analysis on the binned and thresholded
  spikes
