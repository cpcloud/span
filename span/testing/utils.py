import functools
import nose

def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg

def skip(test):
    @functools.wraps(test)
    def wrapper():
        if mock:
            return test()
        raise nose.SkipTest
