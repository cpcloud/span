#!/bin/bash

function ncores
{
    grep -c 'model name' /proc/cpuinfo
}

function nocoverfast
{
    nosetests -w span -A "not slow" --ignore='make_feature_file.py' \
        --ignore='.*flymake.*' --detailed-errors --processes=$(ncores)
}

function nocover
{
    nosetests -w span --ignore='make_feature_file.py' --ignore='.*flymake.*' \
        --detailed-errors --processes=$(ncores)
}

function cover
{
    nosetests -w span --ignore='make_feature_file.py' --ignore='.*flymake.*' \
        --detailed-errors \
        --with-coverage \
        --cover-package=span \
        --cover-tests \
        --cover-erase \
        --cover-inclusive \
        --cover-branches
}


case $1 in
    cover|c) cover ;;
    nocover|nc) nocover ;;
    *) nocoverfast ;;
esac
