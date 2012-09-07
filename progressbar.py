import sys

class ProgressBar(object):
    """
    """
    def __init__(self, start=0, end=10, width=12, fill='#', blank='.',
                 format='[{fill}{blank}] {progress}%', incremental=True):
        """
        """
        super(ProgressBar, self).__init__()
        self.start = start
        self.end = end
        self.width = width
        self.fill = fill
        self.blank = blank
        self.format = format
        self.incremental = incremental
        self.step = 100.0 / float(width)
        self.reset()

    def __iadd__(self, increment):
        """
        """
        increment = self._get_progress(increment)
        if self.progress + increment < 100:
            self.progress += increment
        else:
            self.progress = 100.0
        return self

    def __str__(self):
        """
        """
        progressed = int(self.progress / self.step)
        fill = progressed * self.fill
        blank = (self.width - progressed) * self.blank
        return self.format.format(fill=fill, blank=blank,
                                  progress=int(self.progress))

    __repr__ = __str__

    def _get_progress(self, increment):
        """
        """
        return float(increment * 100.0) / self.end

    def reset(self):
        """
        """
        self.progress = self._get_progress(self.start)
        return self


class AnimatedProgressBar(ProgressBar):
    """
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(AnimatedProgressBar, self).__init__(*args, **kwargs)
        self.stdout = kwargs.get('stdout', sys.stdout)

    def show_progress(self):
        """
        """
        c = '\n'
        if hasattr(self.stdout, 'isatty') and self.stdout.isatty():
            c = '\r'
            
        self.stdout.write(c)
        self.stdout.write(str(self))
        self.stdout.flush()
