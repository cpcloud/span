import pandas as pd

import span.tdt
from span.utils.utils import detrend_mean


class DataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(DataFrame, self).__init__(*args, **kwargs)

    def xcorr(self, maxlags=None, detrend=detrend_mean, scale_type='normalize'):
        return span.tdt.xcorr(self, maxlags=maxlags, detrend=detrend,
                              scale_type=scale_type)
