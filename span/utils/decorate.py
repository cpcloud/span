# decorate.py ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import sys
import functools
import threading


def thunkify(f):
    """Perform `f` using a threaded thunk.

    Adapted from...

    Parameters
    ----------
    f : callable

    Raises
    ------
    AssertionError

    Returns
    -------
    thunked : function
    """
    assert callable(f), '"f" must be callable'

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
        # create an event
        wait_event = threading.Event()

        # default results and exception status
        result = [None]
        exc = [False, None]

        def worker():
            """Function that runs `f`."""

            try:
                # try to run the function
                result[0] = f(*args, **kwargs)
            except Exception:
                # set the except status if any was caught
                exc[0], exc[1] = True, sys.exc_info()
            finally:
                # no matter what happens tell the event that we're done running
                wait_event.set()


        def thunk():
            """The actual thunk.
            Raises
            ------
            Exception
                If any exceptions were caught in the worker thread.

            Returns
            -------
            res : type(f(*args, **kwargs))

            """
            # wait for the event
            wait_event.wait()

            # throw the exception if any
            if exc[0]:
                raise exc[1][0], exc[1][1], exc[1][2]

            return result[0]

        threading.Thread(target=worker).start()

        return thunk

    return thunked


def cached_property(f):
    """Returns a cached property that is calculated by function `f`

    This function allows one to create a computed property that is cached, i.e.,
    created once and then stored. This is useful if you have to perform a
    long running computation that you only need to do once.

    Parameters
    ----------
    f : callable

    Raises
    ------
    AssertionError

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
