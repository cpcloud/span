from functools import partial, reduce
from itertools import repeat


def flip(f):
    if not callable(f):
        raise TypeError('Cannot flip a non-callable object.')
    def result(*args, **kwargs):
        args = list(args)
        args.reverse()
        return f(*args, **kwargs)
    return result


def ilast(i):
    return reduce(lambda _, x: x, i)


def iscanl(f, v, seq):
    yield v
    for a in seq:
        v = f(v, a)
        yield v


def scanl(*args, **kwargs):
    return list(iscanl(*args, **kwargs))


def foldl(*args, **kwargs):
    return ilast(iscanl(*args, **kwargs))


def iscanr(f, v, seq):
    return iscanl(flip(f), v, seq)


def scanr(*args, **kwargs):
    result = list(iscanr(*args, **kwargs))
    result.reverse()
    return result


def foldr(*args, **kwargs):
    return ilast(iscanr(*args, **kwargs))


def compose2(f, g):
    if not all(map(callable, (f, g))):
        raise TypeError('f and g must both be callable')
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def compose(*args):
    return partial(reduce, compose2)(args)


def composemap(*args):
    nmaps = repeat(map, len(args))
    partial_map = map(partial, nmaps, args)

    # compose2(... compose2(p0, p1), ... pN)
    return reduce(compose2, partial_map)



