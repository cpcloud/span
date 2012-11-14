import functools
import nose

import numpy as np

def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg


def skip(test):
    @functools.wraps(test)
    def wrapper():
        if mock:
            return test()
        raise nose.SkipTest


def randrange(a=0, b=1, size=None):
    return (b - a) * np.random.random_sample(size=size) + a
