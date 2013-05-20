import os
import sys

import pandas as pd

from span.utils import green, white, red, magenta, puts, bold
from span.spanner.defaults import SPAN_DB_PATH

import sh

git = sh.git


def _get_from_db(dirname, rec_num, method, *args):
    """hash the args and use that number to store the analysis results

    Try to retreive the results of the analysis for the hash of these args
    otherwise perform the analysis and store it for later use if needed.
    """
    full_path = os.path.join(SPAN_DB_PATH, 'h5', dirname, str(rec_num) + '.h5')
    key = str(hash(args))

    with pd.get_store(full_path, mode='a') as raw_store:
        try:
            out = raw_store.get(key)
        except KeyError:
            out = method(*args)
            raw_store.put(key, out)
    return out


def error(msg):
    errmsg = green(os.path.basename(__file__))
    errmsg += '{0} {1}{0} {2}'.format(white(':'), red('error'), magenta(msg))
    puts(bold(errmsg))
    return sys.exit(2)


def _create_new_db(db, path):
    try:
        os.remove(path)
    except OSError:
        # path doesn't exist
        pass

    db.to_csv(path, index_label=db.columns.names)
    dbdir = os.path.dirname(path)
    curdir = os.getcwd()
    git.init(dbdir)
    os.chdir(dbdir)
    git.add(path)

    # something has changed
    _commit_if_changed(version=0)
    os.chdir(curdir)


def _init_db(path, dbcls):
    """initialize the database"""
    if not hasattr(dbcls, 'schema'):
        raise AttributeError('class {0} has no schema attribute'.format(dbcls))
    schema = sorted(dbcls.schema)
    name = 'field'
    columns = pd.Index(schema, name=name)
    empty_db = pd.DataFrame(columns=columns).sort_index(axis=1)

    try:
        current_db = pd.read_csv(path)
        current_db.pop(name)
        current_db.columns.name = name
    except IOError:
        # db doesn't exist create it
        _create_new_db(empty_db, path)
    else:
        if not (current_db.columns != columns).all():
            _create_new_db(empty_db, path)


def _commit_if_changed(version):
    if git('diff-index', '--quiet', 'HEAD', '--').exit_code:
        git.commit(message="'version {0}'".format(version))


def _pop_column_to_name(df, column):
    try:
        df.columns.name = df.pop(column).name
    except KeyError:
        df.columns.name = df.pop(df.columns[column]).name
