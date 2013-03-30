#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    import os
    from multiprocessing import cpu_count
    from nose import run

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fast', action='store_true', default=False)
    parser.add_argument('-t', '--cover-tests', action='store_true',
                        default=False)
    parser.add_argument('-c', '--cover', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parsed_args = parser.parse_args()

    package_name = os.path.basename(os.path.abspath(os.curdir))

    argv = ['nosetests', '-w', package_name]

    if parsed_args.fast:
        argv.extend(['-A "not slow"', '--processes=%d' % cpu_count()])

    if parsed_args.cover:
        argv.extend(['--with-coverage', '--cover-package=%s' % package_name,
                     '--cover-erase', '--cover-inclusive', '--cover-branches'])

    if parsed_args.cover_tests:
        argv.append('--cover-tests')

    run(argv=argv)
