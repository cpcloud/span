import sys
import functools
import threading


def thunkify(f):
    """Perform `f` using a threaded thunk.

    Parameters
    ----------
    f : callable
    """
    @functools.wraps(f)
    def thunked(*args, **kwargs):
        """The thunked version of `f`

        Parameters
        ----------
        args : tuple, optional
        kwargs : dict, optional

        Returns
        -------
        thunk : callable
        """
        wait_event = threading.Event()
        result = [None]
        exc = [False, None]

        def worker():
            """The worker thread with which to run `f`."""
            try:
                result[0] = f(*args, **kwargs)
            except Exception:
                exc[0], exc[1] = True, sys.exc_info()
            finally:
                wait_event.set()

        def thunk():
            """The actual thunk.

            Returns
            -------
            res : type(f(*args, **kwargs))
            """
            wait_event.wait()
            if exc[0]:
                raise exc[1][0], exc[1][1], exc[1][2]
            return result[0]
        threading.Thread(target=worker).start()
        return thunk
    return thunked


def cached_property(f):
    """returns a cached property that is calculated by function `f`

    Parameters
    ----------
    f : callable

    Returns
    -------
    getter : callable
    """
    assert callable(f), 'f must be callable'
    
    @property
    @functools.wraps(f)
    def getter(self):
        try:
            x = self.__property_cache[f]
        except AttributeError:
            self.__property_cache = {}
            x = self.__property_cache[f] = f(self)
        except KeyError:
            x = self.__property_cache[f] = f(self)
        return x
    return getter

