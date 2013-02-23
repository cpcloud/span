#!/bin/bash
procs=`nproc`
nosetests -w span -A "not slow" \
    --nologcapture --nocapture --processes=$procs $*
