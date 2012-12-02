import time
import unittest

from numpy.random import rand
from numpy.testing import assert_allclose

from span.utils.decorate import cached_property, thunkify


def test_cached_property():
    class CachedPropertyClass(object):
        def __init__(self, a, b):
            self.a, self.b = a, b

        @cached_property
        def c(self):
            return self.a + self.b

        @cached_property
        def d(self):
            return self.c

    a, b = rand(), rand()
    cpc = CachedPropertyClass(a, b)
    c = cpc.c
    assert hasattr(cpc, '__property_cache')
    assert c == a + b
    assert cpc.d == c


class TestThunkify(unittest.TestCase):
    def test_thunkify(self):
        @thunkify
        def thunky(i):
            time.sleep(0.1)
            return i * 2

        def call_thunky(x):
            double_thunk = thunky(x)
            time.sleep(0.3)

            if x == 100:
                res = double_thunk()
            else:
                res = None
            return res

        t0 = time.time()
        call_thunky(10)
        t1 = time.time() - t0

        t0 = time.time()
        call_thunky(100)
        t2 = time.time() - t0

        assert_allclose(t1, t2, rtol=1e-2)

        @thunkify
        def thrower():
            return None ** 2

        self.assertRaises(TypeError, thrower())
