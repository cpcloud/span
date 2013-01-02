import unittest
import itertools as itools
import ConfigParser
import os

import numpy as np

from span.tdt import distance_map, ElectrodeMap, parse_electrode_config
from span.testing import create_spike_df, slow


@slow
class TestDistanceMap(unittest.TestCase):
    def test_distance_map(self):
        nshanks = np.r_[:9]
        elecs = nshanks.copy()

        between = np.r_[25:150:25]
        within = between.copy()

        metrics = 'wminkowski',
        ps = 1, 2, np.inf

        arg_sets = itools.product(nshanks, elecs, within, between, metrics, ps)

        for arg_set in arg_sets:
            nsh, nelec, wthn, btwn, metr, p = arg_set

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


class TestElectrodeMap(unittest.TestCase):
    def setUp(self):
        self.b = np.arange(1, 101, 25)
        self.w = self.b.copy()

        self.nelecs_per_shank = np.arange(64) + 1
        self.nshanks = np.arange(7) + 2

    def tearDown(self):
        del self.nshanks, self.nelecs_per_shank, self.w, self.b

    def test_nchans(self):
        assert False

    def test_1d_map(self):
        nshank = 1
        for nelec_per_shank in self.nelecs_per_shank:
            nelecs = nelec_per_shank * nshank
            a = np.arange(nelecs).reshape(nelec_per_shank, nshank)
            em = ElectrodeMap(a)
            self.assertIsNotNone(em)

    def test_2d_map(self):
        for nelec_per_shank, nshank in itools.product(self.nelecs_per_shank,
                                                      self.nshanks):
            nelecs = nelec_per_shank * nshank
            a = np.arange(nelecs).reshape(nelec_per_shank, nshank)

            em = ElectrodeMap(a)
            self.assertIsNotNone(em)

    def test_distance_map_1d(self):
        nshank = 1
        arg_set = itools.product(self.w, self.b, self.nelecs_per_shank)

        i = 0
        for w, b, nelec_per_shank in arg_set:
            nelecs = nelec_per_shank * nshank
            a = np.arange(nelecs)[:, np.newaxis]

            em = ElectrodeMap(a)
            dm = em.distance_map(w, b)

            if not i:
                print
                print em.nshanks
                print em.shank.unique()
                print w, b
                print dm
                i += 1
            self.assertIsNotNone(em)

    def test_distance_map_2d(self):
        arg_set = itools.product(self.w, self.b, self.nelecs_per_shank,
                                 self.nshanks)

        for w, b, nelec_per_shank, nshank in arg_set:
            nelecs = nelec_per_shank * nshank
            a = np.arange(nelecs).reshape(nelec_per_shank, nshank)

            em = ElectrodeMap(a)
            dm = em.distance_map(w, b)
            self.assertIsNotNone(em)


class TestDistanceMapWithCrossCorrelation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sp = create_spike_df()
        thr = sp.threshold(4 * sp.std())
        clr = sp.clear_refrac(thr)
        binned = sp.bin(clr, binsize=10)
        cls.xc = sp.xcorr(binned, maxlags=10)

        rawmap = np.array([1, 3, 2, 6,
                           7, 4, 8, 5,
                           13, 10, 12, 9,
                           14, 16, 11, 15]).reshape(4, 4)
        cls.elecmap = ElectrodeMap(rawmap, order='lm')
        cls.dm = cls.elecmap.distance_map(50, 125)

    def test_set_index(self):
        xcc = self.xc.T.set_index(self.dm, append=True).T
        lag0_tmp = xcc.ix[0].dropna().sortlevel(level=6)
        lag0 = lag0_tmp.reset_index(level=range(6), drop=True)
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

        os.remove(filename)
