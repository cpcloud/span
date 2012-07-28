def astype(a, dtype, order='K', casting='unsafe', subok=True, copy=False):
    try:
        r = a.astype(dtype, order=order, casting=casting, subok=subok,
                     copy=copy)
    except TypeError:
        r = a.astype(dtype)
    return r
    