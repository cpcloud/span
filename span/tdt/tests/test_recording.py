from unittest import TestCase
import numbers
import itertools as itools
import copy

import numpy as np
from numpy.random import randint

from span.tdt import distance_map, ElectrodeMap
from span.testing import create_spike_df, knownfailure


class TestDistanceMap(TestCase):
    def test_distance_map(self):
        nshanks = tuple(np.r_[:9])
        elecs = copy.deepcopy(nshanks)

        between = 25, 50
        within = copy.deepcopy(between)

        metrics = 'wminkowski',
        ps = 1, 2, np.inf

        arg_sets = itools.product(nshanks, elecs, within, between, metrics, ps)

        def _tester(nsh, nelec, wthn, btwn, metr, p):
            if not (nsh and nelec):
                self.assertRaises(AssertionError, distance_map, nsh, nelec,
                                  wthn, btwn, metr, p)
            else:
                if nelec == 1:
                    nsh = 1

                if nsh == 1:
                    btwn = 0

                if nsh > 1 and nelec > 1:
                    dm = distance_map(nsh, nelec, btwn, wthn, metr, p)
                    self.assertEqual(dm.size, (nsh * nelec) ** 2)

        for arg_set in arg_sets:
            _tester(*arg_set)


class TestElectrodeMap(TestCase):
    def setUp(self):
        self.b = np.arange(1, 101, 25)
        self.w = self.b.copy()
        self.nelecs = np.arange(64) + 1

    def tearDown(self):
        del self.nelecs, self.w, self.b

    def test_nchans(self):
        b, w, nelecs = self.b, self.w, self.nelecs
        arg_sets = itools.product(b, w, nelecs)

        for bb, ww, nelec, in arg_sets:
            nshanks = randint(1, 9)
            a = randint(1, nelec + 1, size=(nelec, nshanks))
            em = ElectrodeMap(a)
            self.assert_(hasattr(em, 'nchans'))
            self.assertIsInstance(em.nchans, numbers.Integral)

    def test_1d_map(self):
        for n in self.nelecs:
            a = randint(1, n + 1, size=n)
            em = ElectrodeMap(a)
            self.assertIsNotNone(em)

    def test_2d_map(self):
        nshanks = xrange(2, 9)
        for nelec, nshank in itools.product(self.nelecs + 1, nshanks):
            a = randint(1, nelec + 1, size=(nelec, nshank))
            em = ElectrodeMap(a)
            self.assertIsNotNone(em)

    @knownfailure
    def test_distance_map_1d(self):
        b, w, nelecs = self.b, self.w, self.nelecs
        arg_sets = itools.product(b, w, nelecs)

        for bb, ww, n in arg_sets:
            a = randint(1, n + 1, size=n)
            em = ElectrodeMap(a)

            if ww and bb and em.values.size > 1:
                dm = em.distance_map(ww, bb)
                self.assertIsNotNone(dm)
            else:
                self.assertRaises(ValueError, em.distance_map, ww, bb)

    def test_distance_map_2d(self):
        nshanks = randint(2, 8)
        b, w, nelecs = self.b, np.zeros_like(self.b), self.nelecs
        arg_sets = itools.product(b, w, nelecs, [nshanks])

        for bb, ww, n, nsh in arg_sets:
            a = randint(1, n + 1, size=(n, nsh))

            em = ElectrodeMap(a)

            if ww and bb:
                dm = em.distance_map(ww, bb)
                self.assertIsNotNone(dm)
            else:
                self.assertRaises(AssertionError, em.distance_map, ww, bb)

            self.assertIsNotNone(em)


class TestDistanceMapWithCrossCorrelation(TestCase):
    def setUp(self):
        sp = create_spike_df()
        thr = sp.threshold(4 * sp.std())
        clr = sp.clear_refrac(thr)
        binned = clr.resample('L', how='sum')
        self.xc = sp.xcorr(binned, maxlags=10)

        rawmap = np.array([1, 3, 2, 6,
                           7, 4, 8, 5,
                           13, 10, 12, 9,
                           14, 16, 11, 15]).reshape(4, 4)
        self.elecmap = ElectrodeMap(rawmap)
        self.dm = self.elecmap.distance_map(50, 125)

    def test_set_index(self):
        xcc = self.xc.T.set_index(self.dm, append=True).T
        lag0_tmp = xcc.ix[0].dropna().sortlevel(level=4)
        lag0 = lag0_tmp.reset_index(level=range(5), drop=True)
        self.assertIsNotNone(lag0)


class TestParseElectrodeConfig(unittest.TestCase):
    def setUp(self):
        # generate some random config files
        pass

    def tearDown(self):
        # delete the random config files
        pass

    def test_parse_electrode_config(self):
        nshanks = 4
        electrodes_per_shank = 4
        nelectrodes = 16
        order = 'lm'
        eid = '4DCE'

        c = ConfigParser.SafeConfigParser()
        sections = 'meta', 'shanks'

        meta = {'nshanks': str(nshanks),
                'electrodes_per_shank': str(electrodes_per_shank),
                'nelectrodes': str(nelectrodes),
                'order': order,
                'id': eid}
        shanks = (np.arange(nelectrodes) + 1).reshape(electrodes_per_shank,
                                                      nshanks)

        shanks_dict = {}

        for i, shank in enumerate(shanks):
            shanks_dict[str(i)] = ', '.join(map(str, shank.tolist()))

        for section, data in zip(sections, (meta, shanks_dict)):
            c.add_section(section)

            for k, v in data.items():
                c.set(section, k, v)

        filename = '.tmp.cfg'

        with open(filename, 'wt') as f:
            c.write(f)

        ecfg = parse_electrode_config(filename)
        self.assertIsNotNone(ecfg)

        os.remove(filename)
