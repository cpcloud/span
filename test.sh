#!/bin/bash

numprocs=`nproc`
function nocoverfast
{
    nosetests -w span -A 'not slow' \
        --ignore='.*flymake.*' \
        --nocapture \
        --nologcapture \
        --processes=$numprocs
}

function nocover
{
    nosetests --ignore='.*flymake.*' --detailed-errors
}

function cover
{
    nosetests -w span --ignore='.*flymake.*' \
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
    nosetests -w span --ignore='.*flymake.*' \
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
    *) nocover ;;
esac
