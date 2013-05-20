import os
import tarfile
import subprocess

from span.spanner.command import SpanCommand
from span.spanner.converters import _converters, _build_neuroscope_package
from span.spanner.converters import _get_dat_from_tarfile
from span.spanner.utils import error


def _run_neuroscope(tarfile):
    member = _get_dat_from_tarfile(tarfile)
    tarfile.extractall()
    try:
        return subprocess.check_call(['neuroscope', os.path.join(os.curdir,
                                                                 member.name)])
    except OSError:
        return error('could not find neuroscope on the system path, it is '
                     'probably not '
                     'installed\n\nPATH={0}'.format(os.environ.get('PATH')))
    except subprocess.CalledProcessError as e:
        return error(e.msg)


class Viewer(SpanCommand):
    def _run(self, args):
        tank, spikes = self._load_data(return_tank=True)
        base, _ = os.path.splitext(self.filename)
        base = os.path.join(os.curdir, os.path.basename(base))
        outfile = '{base}{extsep}dat'.format(base=base, extsep=os.extsep)
        converter = _converters['neuroscope']('int', 16, tank.datetime)
        args.precision = converter.precision
        zipped_name = '{0}{1}tar{1}{2}'.format(base, os.extsep, args.format)
        _build_neuroscope_package(spikes, converter, base, outfile,
                                  zipped_name, args)
        with tarfile.open(zipped_name,
                          mode='r:{0}'.format(args.format)) as r_package:
            _run_neuroscope(r_package)
