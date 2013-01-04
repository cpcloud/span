#!/bin/bash

nosetests -w span --ignore='make_feature_file.py' --ignore='.*flymake.*' \
    --detailed-errors \
    --nocapture \
    --with-coverage \
    --cover-package=span \
    --cover-tests \
    --cover-erase \
    --cover-inclusive \
    --cover-branches
