#!/bin/bash

local num_procs=`grep -c 'model name' /proc/cpuinfo`
nosetests -w span -A "not slow" --processes=$num_procs $*
