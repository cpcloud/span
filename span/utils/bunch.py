import tokenize
import keyword
import re
import numpy as np

class Bunch(dict):
    """A class to bundle up a dict and allow key access similar to
    `collection.namedtuple`, i.e., via attributes.

    See Also
    --------
    `collection.namedtuple`
    """
    def __init__(self, **kwargs):
        # check to make sure keys are valid identifiers
        super(Bunch, self).__init__(**kwargs)
        self.__dict__ = self

    def __str__(self):
        members = ', '.join('{0}={1}'.format(k, self.get(k)) for k in self)
        return '{classname}({members})'.format(classname=self.__class__.__name__,
                                                members=members)

    def __repr__(self):
        return '<Bunch object at {0}>'.format(hex(id(self)))


def isidentifier(s):
    """Check to see whether a string is a valid identifier.

    Parameters
    ----------
    s : string
        String to check for identifier-ness.

    Returns
    -------
    is_valid_id : bool
        Whether or not `s` is a valid Python identifier
    """
    is_not_keyword = not keyword.iskeyword(s)
    has_no_whitespace = re.match(r'\s+', s) is None
    is_valid_name = re.match(tokenize.Name, s) is not None
    is_valid_id = is_not_keyword and has_no_whitespace and is_valid_name
    return is_valid_id


def has_valid_keys(d):
    """Check a dict-like object for identifier-like keys.

    Parameters
    ----------
    d : dict_like
        Dictionary to check for keys that are valid identifiers

    Returns
    -------
    where_bad : array_like
        The indicies of the invalid keys
    """
    where_bad, = np.where([not isidentifier(key) for key in d])
    return where_bad.tolist()
