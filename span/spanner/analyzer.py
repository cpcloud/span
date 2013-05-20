import functools

from span.spanner.command import SpanCommand
from span.spanner.utils import _get_from_db, error


class Analyzer(SpanCommand):
    pass


def _compute_xcorr(raw, rec_num, args):
    import span
    from span import spike_xcorr

    detrend = getattr(span, 'detrend_' + args.detrend)

    thr = _get_from_db('thr', rec_num, raw.threshold, args.threshold)
    cleared = _get_from_db('clr', rec_num, functools.partial(thr.clear_refrac,
                                                             inplace=True),
                           args.refractory_period)
    binned = _get_from_db('binned', rec_num, cleared.bin, args.bin_size,
                          args.bin_method)
    xc = _get_from_db('xcorr', rec_num, functools.partial(spike_xcorr, binned),
                      args.max_lags, args.scale_type, detrend, args.nan_auto)
    return xc


def _build_plot_filename(tank):
    raise NotImplementedError()


class CorrelationAnalyzer(Analyzer):
    def _run(self, args):
        meta, spikes = self._load_data(return_meta=True)
        xc = _compute_xcorr(spikes, args.id, args)
        plots = self._build_xcorr_plots(xc)
        plot_filename = _build_plot_filename(meta)
        self._save_plots(plots, plot_filename)

        if args.display:
            self._display_xcorr(plots)

    def _display_xcorr(self, xc, plot_filename):
        pass


class IPythonAnalyzer(Analyzer):
    """Drop into an IPython shell given a filename or database id number"""
    def _run(self, args):
        try:
            from IPython import embed
        except ImportError:
            return error('ipython not installed, please install it with pip '
                         'install ipython')
        else:
            tank, spikes = self._load_data(return_tank=True)
            embed()
        return 0
