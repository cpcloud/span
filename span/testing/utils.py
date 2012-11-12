import functools
import nose


def skip(test):
    @functools.wraps(test)
    def wrapper():
        if mock:
            return test()
        raise nose.SkipTest
