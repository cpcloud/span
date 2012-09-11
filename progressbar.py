import sys

class ProgressBar(object):
    """Class encapsulating a progress bar.
    """
    def __init__(self, start=0, end=10, width=12, fill='#', blank='.',
                 fmt='[{fill}{blank}] {progress}%', incremental=True):
        """Constructor for ProgressBar class.

        Parameters
        ----------
        start : int, optional
        end : int, optional
        width : int, optional
        fill : str, optional
        blank : str, optional
        fmt : str, optional
        incremental : bool, optional
        """
        super(ProgressBar, self).__init__()
        self.start = start
        self.end = end
        self.width = width
        self.fill = fill
        self.blank = blank
        self.format = fmt
        self.incremental = incremental
        self.step = 100. / width
        self.reset()

    def __iadd__(self, increment):
        """Increment the amount of progress in place.

        Parameters
        ----------
        increment : int

        Returns
        -------
        self : ProgressBar
        """
        increment = self._get_progress(increment)
        if self.progress + increment < 100.0:
            self.progress += increment
        else:
            self.progress = 100.0
        return self

    def __str__(self):
        """Represent the progress bar as a string.

        Returns
        -------
        formatted : str
            The progress bar formatted as a string
        """
        progressed = int(self.progress / self.step)
        fill = progressed * self.fill
        blank = (self.width - progressed) * self.blank
        return self.format.format(fill=fill, blank=blank,
                                  progress=int(self.progress))

    __repr__ = __str__

    def _get_progress(self, increment):
        """Get the current amount of progress.

        Parameters
        ----------
        increment : int

        Returns
        -------
        prog : float
            The current amount of progress.
        """
        return increment * 100.0 / self.end

    def reset(self):
        """Reset the current amount of progress.

        Returns
        -------
        self : ProgressBar
            The current instance
        """
        self.progress = self._get_progress(self.start)
        return self


class AnimatedProgressBar(ProgressBar):
    """An animated progress bar.

    This class is useful for showing the progress of a process in a terminal.
    """
    def __init__(self, *args, **kwargs):
        """Constructor.

        Parameters
        ----------
        args : tuple, optional
        kwargs : dict, optional
        """
        super(AnimatedProgressBar, self).__init__(*args, **kwargs)
        self.stdout = kwargs.get('stdout', sys.stdout)

    def show_progress(self):
        """Show the current progress, compensating for terminal existence."""
        is_terminal = hasattr(self.stdout, 'isatty') and self.stdout.isatty()
        c = '\r' if is_terminal else '\n'
        self.stdout.write(c)
        self.stdout.write(str(self))
        self.stdout.flush()
