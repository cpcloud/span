#!/bin/bash


function nocoverfast
{
    nosetests -w span -A 'not slow' --ignore='make_feature_file.py' \
        --ignore='.*flymake.*' --detailed-errors --processes=`nproc`
}

function nocover
{
    nosetests -w span --ignore='make_feature_file.py' --ignore='.*flymake.*' \
        --detailed-errors --processes=`nproc`
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

function cover_no_tests
{
    nosetests -w span --ignore='make_feature_file.py' --ignore='.*flymake.*' \
        --detailed-errors \
        --with-coverage \
        --cover-package=span \
        --cover-erase \
        --cover-inclusive \
        --cover-branches
}


case $1 in
    cover|c) cover ;;
    nocover|nc) nocover ;;
    covernotest|cnt) cover_no_tests ;;
    *) nocoverfast ;;
esac
