import os
import glob
import numbers

import pandas as pd
from numpy import nan

from span import ElectrodeMap, TdtTank, NeuroNexusMap
from span.utils import bold, blue, red
from span.spanner.utils import error, _pop_column_to_name
from span.spanner.defaults import SPAN_DB_PATH, SPAN_DB


def _get_fn(id_num_or_filename, db_path=SPAN_DB):
    if db_path is None:
        error('SPAN_DB_PATH environment variable not set, please set via '
              '"export SPAN_DB_PATH=\'path_to_the_span_database\'"')
    db_path = os.path.abspath(db_path)
    db = pd.read_csv(db_path)
    _pop_column_to_name(db, 0)

    if isinstance(id_num_or_filename, numbers.Integral):
        if id_num_or_filename not in db.index:
            error(bold('{0} {1}'.format(blue('"' + str(id_num_or_filename) +
                                             '"'),
                                        red('is not a valid id number'))))
    elif isinstance(id_num_or_filename, basestring):
        if id_num_or_filename not in db.filename.values:
            error(bold('{0} {1}'.format(blue('"' + str(id_num_or_filename) +
                                             '"'),
                                        red('is not a valid filename'))))
        return id_num_or_filename
    return db.filename.ix[id_num_or_filename]


class SpanCommand(object):

    def _parse_filename_and_id(self, args):
        if args.filename is None and args.id is not None:
            self.filename = _get_fn(args.id)
        elif args.filename is not None and args.id is None:
            self.filename = _get_fn(args.filename)
        else:
            return error('Must pass a valid id number or filename')

        paths = glob.glob(self.filename + '*')
        common_prefix = os.path.commonprefix(paths)
        self.filename = common_prefix.strip(os.extsep)

        if not paths:
            return error('No paths match the expression '
                         '"{0}*"'.format(self.filename))

    def run(self, args):
        #self._parse_filename_and_id(args)
        return self._run(args)

    def _run(self, args):
        raise NotImplementedError()

    def _load_data(self):
        #full_path = os.path.join(SPAN_DB_PATH, 'h5', 'raw', str(self.id) +
                                 #'.h5')
        #with pd.get_store(full_path, mode='a') as raw_store:
            #try:
                #spikes = raw_store.get('raw')
                #meta = self._get_meta(self.id)
            #except KeyError:
                #em = ElectrodeMap(NeuroNexusMap.values, 50, 125)
                #tank = TdtTank(os.path.normpath(self.filename), em)
                #spikes = tank.spik
                #raw_store.put('raw', spikes)
                #meta = self._get_meta(tank)
        em = ElectrodeMap(NeuroNexusMap.values, 50, 125)
        tank = TdtTank(os.path.normpath(self.filename), em)
        spikes = tank.spik
        meta = self._get_meta(tank)

        return meta, spikes

    def _get_meta(self, obj):
        #method = {TdtTank: self._get_meta_from_tank,
                  #int: self._get_meta_from_id}[type(obj)]
        #return method(obj)
        keys = ('path', 'name', 'age', 'site', 'time', 'date', 'start', 'end',
                'duration')
        d = dict((k, getattr(obj, k) if hasattr(obj, k) else nan)
                 for k in keys)
        return pd.Series(d)

    def _get_meta_from_id(self, id_num):
        return self.db.iloc[id_num]

    def _get_meta_from_tank(self, tank):
        meta = self._append_to_db_and_commit(tank)
        return meta

    def _get_meta_from_args(self, args):
        pass

    def _append_to_db_and_commit(self, tank):
        row = self._row_from_tank(tank)
        self._append_to_db(row)
        self._commit_to_db()
        return row

    def _append_to_db(self, row):
        pass

    def _commit_to_db(self):
        pass

    def _row_from_tank(self, tank):
        row = _make_row(tank)
        index = self._compute_db_index()
        return pd.Series(row, name=index)

    def _compute_db_index(self):
        pass


def _make_row(tank):
    """
    get the relevant data from the tank into a numpy array
    """
    pass
