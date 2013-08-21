#!/usr/bin/env python

from pandas import read_csv


def parse_log(filename, sep=r'\||='):
    df = read_csv(filename, sep=sep, names=['date', 'kind', 'param_name',
                                            'param_value'])
    kind = df.kind.str.lower()
    mask = (df.param_name == 'filename') & (kind == 'error')
    bad_files = df[mask].drop_duplicates()
    return bad_files



if __name__ == '__main__':
    pass
