#!/bin/bash

# There are 2 distinct pieces that get zipped and cached
# - The venv site-packages dir including the installed dependencies
# - The pandas build artifacts, using the build cache support via
#   scripts/use_build_cache.py
#
# if the user opted in to use the cache and we're on a whitelisted fork
# - if the server doesn't hold a cached version of venv/pandas build,
#   do things the slow way, and put the results on the cache server
#   for the next time.
# -  if the cache files are available, instal some necessaries via apt
#    (no compiling needed), then directly goto script and collect 200$.
#

echo "inside $0"

# Install Dependencies
# as of pip 1.4rc2, wheel files are still being broken regularly, this is a known good
# commit. should revert to pypi when a final release is out
pip install -I git+https://github.com/pypa/pip@42102e9deaea99db08b681d06906c2945f6f95e2#egg=pip
pip install -I -U setuptools
pip install wheel

# comment this line to disable the fetching of wheel files
PIP_ARGS+=" -I --use-wheel --find-links=http://cache27diy-cpycloud.rhcloud.com/span/${TRAVIS_PYTHON_VERSION}/"

# Force virtualenv to accpet system_site_packages
rm -f $VIRTUAL_ENV/lib/python$TRAVIS_PYTHON_VERSION/no-global-site-packages.txt

time pip install $PIP_ARGS -r ci/requirements-${TRAVIS_PYTHON_VERSION}.txt

time python setup.py build_ext --inplace
time python setup.py install

true
