#!/bin/bash

nosetests -w span --with-coverage \
    --cover-tests \
    --cover-erase \
    --cover-package=span \
    --cover-inclusive \
    --cover-branches \
    --ignore=make_feature_file.py \
    --ignore='.*flymake.*' \
    --detailed-errors
