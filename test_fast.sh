#!/bin/bash
procs=`grep -c 'model name' /proc/cpuinfo`
nosetests -w span -A "not slow" --nocapture --processes=$procs $*
