from functools import wraps
import nose

from numpy.testing import assert_allclose, assert_array_equal
from numpy.testing.decorators import slow

from nose.tools import nottest
from nose import SkipTest


from numpy.random import uniform as randrange


def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg


def skip(test):
    @wraps(test)
    def wrapper():
        if mock:
            return test()
        raise nose.SkipTest
