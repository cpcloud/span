import pandas as pd

import span
from utils import detrend_mean


class Series(pd.Series):
    def __new__(cls, *args, **kwargs):
        return pd.Series.__new__(cls, *args, **kwargs).view(Series)
    
    def __init__(self, *args, **kwargs):
        super(Series, self).__init__(*args, **kwargs)

    def xcorr(self, other=None, maxlags=None, detrend=detrend_mean,
              scale_type='normalize'):
        return span.tdt.xcorr(self, other, maxlags=maxlags, detrend=detrend,
                              scale_type=scale_type)
