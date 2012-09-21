import unittest

import numpy as np

from span.utils.progressbar import ProgressBar, AnimatedProgressBar


class TestProgressBar(unittest.TestCase):
    def setUp(self):
        self.progbar = ProgressBar()

    def test___init__(self):
        pass

    def test___iadd__(self):
        self.progbar.progress = 0
        self.progbar += 10
        self.progbar.reset()
        self.progbar += 1000

    def test___str__(self):
        s = str(self.progbar)

    def test__get_progress(self):
        pass


class TestAnimatedProgressBar(TestProgressBar):
    def setUp(self):
        self.progbar = AnimatedProgressBar()

    def test_show_progress(self):
        self.progbar += 10
        self.progbar.show_progress()
