import time
from unittest import TestCase

from numpy.random import rand
from numpy.testing import assert_allclose

from span.utils.decorate import cached_property, thunkify
from span.testing import slow


class ThunkifyException(Exception):
    pass


class TestCachedProperty(TestCase):
    def test_cached_property(self):
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
        self.assert_(hasattr(cpc, '__property_cache'))
        self.assertEqual(c, a + b)
        self.assertEqual(cpc.d, c)


class TestThunkify(TestCase):
    @slow
    def test_thunkify(self):
        @thunkify
        def thunky(i):
            time.sleep(0.1)
            return i * 2

        def call_thunky(x):
            double_thunk = thunky(x)
            time.sleep(0.4)

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

        assert_allclose(t1, t2, atol=1e-3)

    def test_thunkify_raise(self):
        @thunkify
        def type_error_thrower():
            return None ** 2

        @thunkify
        def exception_thrower():
            raise ThunkifyException('thunkify exception')

        self.assertRaises(TypeError, type_error_thrower())
        self.assertRaises(Exception, exception_thrower())
