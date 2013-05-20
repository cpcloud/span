import os
import tempfile
import subprocess

import numpy as np
import pandas as pd

from span.utils import bold, blue, red
from span.spanner.utils import error
from span.spanner.command import SpanCommand
from span.spanner.defaults import SPAN_DB


def _build_query_index(db, query_dict):
    res = pd.Series(np.ones(db.shape[0], dtype=bool))

    try:
        it = query_dict.iteritems()
    except AttributeError:
        it = query_dict

    for column, value in it:
        if column in db and value is not None:
            res &= getattr(db, column) == value

    return res


def _query_dict_from_args(args):
    return args._get_kwargs()


def _df_prettify(df):
    df = df.copy()
    s = bold(df.to_string())

    for colname in df.columns:
        s = s.replace(colname, str(blue(colname)))
    s = s.replace(df.index.name, str(red(df.index.name)))
    return df.to_string()


def _df_pager(df_string):
    with tempfile.NamedTemporaryFile() as tmpf:
        tmpf.write(df_string)

        try:
            return subprocess.check_call([os.environ.get('PAGER', 'less'),
                                          tmpf.name], stdin=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            return error(e.msg)


class Db(SpanCommand):
    schema = ('artifact_ranges', 'weight', 'between_shank', 'valid_recording',
              'age', 'probe_number', 'site', 'filename', 'within_shank',
              'date', 'animal_type', 'shank_order', 'condition')

    def _parse_filename_and_id(self, args):
        self.db = pd.read_csv(SPAN_DB)
        self.db.columns.name = self.db.pop(self.db.columns[0]).name


class DbCreator(Db):
    """Create a new entry in the recording database"""
    def _run(self, args):
        self.validate_args(args)
        self.append_new_row(self.make_new_row(args))
        self.commit_changes()

    def validate_args(self, args):
        pass

    def append_new_row(self, new_row):
        pass

    def commit_changes(self):
        pass


class DbReader(Db):
    """Read, retrieve, search, or view existing entries"""
    def _run(self, args):
        if self.db.empty:
            return error('no recordings in database named '
                         '"{0}"'.format(SPAN_DB))
        else:
            query_dict = _query_dict_from_args(args)
            indexer = _build_query_index(self.db, query_dict)
            return _df_pager(_df_prettify(self.db[indexer]))


class DbUpdater(Db):

    """Edit an existing entry or entries"""
    def _run(self, args):
        pass


class DbDeleter(Db):

    """Remove an existing entry or entries"""
    def _run(self, args):
        pass
