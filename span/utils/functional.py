#!/usr/bin/env python

"""Module implementing standard functional programming constructs.
"""

from functools import partial, reduce
from itertools import repeat, imap as map


def flip(f):
    """Return a function that calls `f` on flipped arguments.

    Parameters
    ----------
    f : callable,
    """
    if not callable(f):
        raise TypeError('Cannot flip a non-callable object.')
    def result(*args, **kwargs):
        args = list(args)
        args.reverse()
        return f(*args, **kwargs)
    return result


def ilast(i):
    """
    """
    return reduce(lambda _, x: x, i)


def iscanl(f, v, seq):
    """
    """
    yield v
    for a in seq:
        v = f(v, a)
        yield v


def scanl(*args, **kwargs):
    """
    """
    return list(iscanl(*args, **kwargs))


def foldl(*args, **kwargs):
    """
    """
    return ilast(iscanl(*args, **kwargs))


def iscanr(f, v, seq):
    """
    """
    return iscanl(flip(f), v, seq)


def scanr(*args, **kwargs):
    """
    """
    result = list(iscanr(*args, **kwargs))
    result.reverse()
    return result


def foldr(*args, **kwargs):
    """
    """
    return ilast(iscanr(*args, **kwargs))


def compose2(f, g):
    """Return a function that computes the composition of two functions.

    Parameters
    ----------
    f, g : callable

    Returns
    -------
    h : callable
    """
    if not all(map(callable, (f, g))):
        raise TypeError('f and g must both be callable')
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def compose(*args):
    """Compose an arbitrary number of functions.

    Parameters
    ----------
    args : tuple of callables

    Returns
    -------
    h : callable
        Composition of callables in `args`.
    """
    return partial(reduce, compose2)(args)


def composemap(*args):
    """Compose an arbitrary number of mapped functions.

    Parameters
    ----------
    args : tuple of callables

    Returns
    -------
    h : callable
         
    """
    return reduce(compose2, map(partial, repeat(map, len(args)), args))



