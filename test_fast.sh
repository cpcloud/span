#!/bin/bash

nosetests -w span -A "not slow" $*
