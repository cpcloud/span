#!/usr/bin/env python


def parse_log(filename, sep='|', value_splitter='='):
    df = read_csv(filename, sep=sep)
    split_value_names = df.value.str.split(value_splitter)
    value_names = split_value_names.str[0]
    values = split_value_names.str[1]
    filenames = values[value_names == 'filename']



if __name__ == '__main__':
    pass
