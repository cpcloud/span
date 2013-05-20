import os

HOME = os.environ.get('HOME', os.path.expanduser('~'))
SPAN_DB_PATH = os.environ.get('SPAN_DB_PATH', os.path.join(HOME, '.spandb'))
SPAN_DB_NAME = os.environ.get('SPAN_DB_NAME', 'db')
SPAN_DB_EXT = os.environ.get('SPAN_DB_EXT', 'csv')
SPAN_DB = os.path.join(SPAN_DB_PATH, '{0}{1}{2}'.format(SPAN_DB_NAME,
                                                        os.extsep,
                                                        SPAN_DB_EXT))
