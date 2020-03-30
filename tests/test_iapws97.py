import unittest
from iapws.iapws97.region1 import Region1
from iapws.iapws97.region2 import Region2
from iapws.iapws97.region3 import Region3
from iapws.iapws97._utils import b23, _p_s, State
import numpy as np

# TODO: Maybe increase precision to X after comma with X the number of digits after comma of the data values.
# Due to how Python shows numbers, the best way to get that amount is X = abs(Decimal(string_value).as_tuple().exponent)

class GeneralTests(unittest.TestCase):
    
    def test_b23_eq_5(self):
        T = 0.623_150_000 * 10**3
        p = 0.165_291_643 * 10**2

        p_res = b23(T=T)

        self.assertAlmostEqual(p, p_res, places=6)

    def test_b23_eq_6(self):
        T = 0.623_150_000 * 10**3
        p = 0.165_291_643 * 10**2

        T_res = b23(p=p)

        self.assertAlmostEqual(T, T_res, places=6)
    
    def test_b23_exception(self):
        T = 1
        p = 1

        self.assertRaises(ValueError, b23, T=T, p=p)

    def test_p_s_eq_30(self):
        tees = [300, 500, 600]
        pss = [0.353_658_941e-2, 0.263_889_776e1, 0.123_443_146e2]

        for T, ps in zip(tees, pss):
            self.assertAlmostEqual(_p_s(T), ps)
    
    def test_p_s_exeption(self):
        self.assertRaises(ValueError, _p_s, T=1000)

class TestRegion1(unittest.TestCase):

    def test_range_validity(self):
        s = State(T=300, p=3)
        s1 = State(T=300, p=80)
        s2 = State(T=500, p=3)

        self.assertTrue(s in Region1())
        self.assertTrue(s2 in Region1())
        self.assertTrue(s2 in Region1())

        s = State(T=300, p=3)
        self.assertTrue(s in Region1())


    def test_backwards_t_ph(self):
        pees = [3, 80, 80]
        hs = [500, 500, 1500]
        tees = [0.391_798_509e3, 0.378_108_626e3, 0.611_041_229e3]

        for p, h, T in zip(pees, hs, tees):
            T_calc = Region1().T_ph(p, h)
            self.assertAlmostEqual(T, T_calc, places=6)

    def test_backwards_t_ps(self):
        pees = [3, 80, 80]
        ss = [0.5, 0.5, 3]
        tees = [0.307_842_258e3, 0.309_979_785e3, 0.565_899_909e3]

        for p, s, T in zip(pees, ss, tees):
            T_calc = Region1().T_ps(p, s)
            self.assertAlmostEqual(T, T_calc, places=6)


    def test_backwards_p_hs(self):
        hs = [0.001, 90, 1500]
        ss = [0, 0, 3.4]
        pees = [9.800_980_612e-4, 9.192_954_727e1, 5.868_294_423e1]

        for h, s, p in zip(hs, ss, pees):
            p_calc = Region1().p_hs(h, s)
            self.assertAlmostEqual(p, p_calc, places=6)

    def test_backwards_p_Th(self):
        pees = [3, 80, 80]
        hs = [500, 500, 1500]
        tees = [0.391_798_509e3, 0.378_108_626e3, 0.611_041_229e3]

        for p, h, T in zip(pees, hs, tees):
            p_calc = Region1().p_Th(T, h)
            self.assertAlmostEqual(p, p_calc, places=4)

    def test_backwards_p_Ts(self):
        pees = [3, 80, 80]
        ss = [0.5, 0.5, 3]
        tees = [0.307_842_258e3, 0.309_979_785e3, 0.565_899_909e3]

        for p, s, T in zip(pees, ss, tees):
            p_calc = Region1().p_Ts(T, s)
            self.assertAlmostEqual(p, p_calc, places=4)

    def test_property_accuracy(self):
        """Test the results from Table 5."""
        s = State(T=300, p=3)
        s1 = State(T=300, p=80)
        s2 = State(T=500, p=3)
        states = [s, s1, s2]

        table5 = np.array([[0.100215168e-2, 0.971180894e-3, 0.120241800e-2],
                           [0.115331273e3, 0.184142828e3, 0.975542239e3],
                           [0.112324818e3, 0.106448356e3, 0.971934985e3],
                           [0.392294792, 0.368563852, 0.258041912e1],
                           [0.417301218e1, 0.401008987e1, 0.465580682e1],
                           [0.150773921e4, 0.163469054e4, 0.124071337e4]])
        table5 = table5.T

        for state, properties in zip(states, table5):
            r = Region1(state=state)
            p = [r.v, r.h, r.u, r.s, r.cp, r.w]
            np.testing.assert_almost_equal(properties, p, decimal=5)

class TestRegion2(unittest.TestCase):

    def test_range_validity(self):
        s = State(T=300, p=0.0035)
        s1 = State(T=700, p=0.0035)
        s2 = State(T=700, p=30)

        self.assertTrue(s in Region2())
        self.assertTrue(s2 in Region2())
        self.assertTrue(s2 in Region2())

    def test_b2bc_eq20(self):
        h = 0.3516004323e4
        p = 0.100_000_000e3

        p_res = Region2.b2bc(h=h)

        self.assertAlmostEqual(p, p_res, places=6)

    def test_b2bc_eq21(self):
        h = 0.3516004323e4
        p = 0.100_000_000e3

        h_res = Region2.b2bc(p=p)

        self.assertAlmostEqual(h, h_res, places=6)

    def test_subregion_ph(self):
        # From table 24.
        regions = {'a': {'h': [3000, 3000, 4000], 'p': [0.001, 3, 3]},
                   'b': {'p': [5, 5, 25], 'h': [3500, 4000, 3500]},
                   'c': {'p': [40, 60, 60], 'h': [2700, 2700, 3200]}}

        for reg, vals in regions.items():
            hs = vals['h']
            ps = vals['p']

            for p, h in zip(ps, hs):
                subregion_calc = Region2.subregion(p=p, h=h)
                self.assertEqual(reg, subregion_calc)
    
    def test_subregion_ps(self):
        # From table 29.
        regions = {'a': {'s': [7.5, 8, 8], 'p': [0.1, 0.1, 2.5]},
                   'b': {'p': [8, 8, 90], 's': [6, 7.5, 6]},
                   'c': {'p': [20, 20, 80], 's': [5.75, 5.25, 5.75]}}

        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']

            for p, s in zip(ps, ss):
                subregion_calc = Region2.subregion(p=p, s=s)
                self.assertEqual(reg, subregion_calc)

    def test_subregion_hs(self):
        # From table 29.
        regions = {'a': {'h': [2800, 2800, 4100], 's': [6.5, 9.5, 9.5]},
                   'b': {'h': [2800, 3600, 3600], 's': [6, 6, 7]},
                   'c': {'h': [2800, 2800, 3400], 's': [5.1, 5.8, 5.8]}}

        for reg, vals in regions.items():
            ss = vals['s']
            hs = vals['h']

            for h, s in zip(hs, ss):
                subregion_calc = Region2.subregion(h=h, s=s)
                self.assertEqual(reg, subregion_calc)

    def test_backwards_t_ph(self):
        # From table 24.
        regions = {'a': {'h': [3000, 3000, 4000], 'p': [0.001, 3, 3], 'T': [0.534433241e3, 0.575373370e3, 0.101077577e4]},
                   'b': {'p': [5, 5, 25], 'h': [3500, 4000, 3500], 'T': [0.801299102e3, 0.101531583e4, 0.875279054e3]},
                   'c': {'p': [40, 60, 60], 'h': [2700, 2700, 3200], 'T': [0.743056411e3, 0.791137067e3, 0.882756860e3]}}

        for reg, vals in regions.items():
            hs = vals['h']
            ps = vals['p']
            ts = vals['T']

            for p, h, T in zip(ps, hs, ts):
                T_calc = Region2().T_ph(p=p, h=h)
                self.assertAlmostEqual(T, T_calc, places=4)

    def test_backwards_t_ps(self):
        # From table 29.
        regions = {'a': {'s': [7.5, 8, 8], 'p': [0.1, 0.1, 2.5], 'T': [0.399517097e3, 0.514127081e3, 0.103984917e4]},
            'b': {'p': [8, 8, 90], 's': [6, 7.5, 6], 'T': [0.600484040e3, 0.106495556e4, 0.103801126e4]},
            'c': {'p': [20, 80, 80], 's': [5.75, 5.25, 5.75], 'T': [0.697992849e3, 0.854011484e3, 0.949017998e3]}}
        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']
            ts = vals['T']

            for p, s, T in zip(ps, ss, ts):
                T_calc = Region2().T_ps(p=p, s=s)
                self.assertAlmostEqual(T, T_calc, places=4)

    def test_backwards_p_Th(self):
        # From table 24.
        regions = {'a': {'h': [3000, 3000, 4000], 'p': [0.001, 3, 3], 'T': [0.534433241e3, 0.575373370e3, 0.101077577e4]},
                   'b': {'p': [5, 5, 25], 'h': [3500, 4000, 3500], 'T': [0.801299102e3, 0.101531583e4, 0.875279054e3]},
                   'c': {'p': [40, 60, 60], 'h': [2700, 2700, 3200], 'T': [0.743056411e3, 0.791137067e3, 0.882756860e3]}}

        for reg, vals in regions.items():
            hs = vals['h']
            ps = vals['p']
            ts = vals['T']

            for p, h, T in zip(ps, hs, ts):
                p_calc = Region2().p_Th(T=T, h=h)
                self.assertAlmostEqual(p, p_calc, places=4)

    def test_backwards_p_Ts(self):
        # From table 29.
        regions = {'a': {'s': [7.5, 8, 8], 'p': [0.1, 0.1, 2.5], 'T': [0.399517097e3, 0.514127081e3, 0.103984917e4]},
            'b': {'p': [8, 8, 90], 's': [6, 7.5, 6], 'T': [0.600484040e3, 0.106495556e4, 0.103801126e4]},
            'c': {'p': [20, 80, 80], 's': [5.75, 5.25, 5.75], 'T': [0.697992849e3, 0.854011484e3, 0.949017998e3]}}
        r = Region2()
        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']
            ts = vals['T']

            for p, s, T in zip(ps, ss, ts):
                self.assertRaises(NotImplementedError, r.p_Ts, T=T, s=s)
                # p_calc = r.p_Ts(T=T, s=s)
                # self.assertAlmostEqual(p, p_calc, places=4)

    def test_backwards_p_hs(self):
        # From table 9 of supplement for p(h,s).
        regions = {'a': {'h': [2800, 2800, 4100], 's': [6.5, 9.5, 9.5], 'p': [1.371012767, 1.879743844e-3, 1.024788997e-1]},
                   'b': {'h': [2800, 3600, 3600], 's': [6, 6, 7], 'p': [4.793911442, 8.395519209e1, 7.527161441]},
                   'c': {'h': [2800, 2800, 3400], 's': [5.1, 5.8, 5.8], 'p': [9.439202060e1, 8.414574124, 8.376903879e1]}}
        
        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']
            hs = vals['h']

            for p, h, s in zip(ps, hs, ss):
                p_calc = Region2().p_hs(h=h, s=s)
                self.assertAlmostEqual(p, p_calc, places=4)

    def test_property_accuracy(self):
        """Test the results from Table 15."""
        s = State(T=300, p=0.0035)
        s1 = State(T=700, p=0.0035)
        s2 = State(T=700, p=30)
        states = [s, s1, s2]

        table15 = np.array([[0.394913866e2, 0.923015898e2, 0.542946619e-2],
                           [0.254991145e4, 0.333568375e4, 0.263149474e4],
                           [0.241169160e4, 0.301262819e4, 0.246861076e4],
                           [0.852238967e1, 0.101749996e2, 0.517540298e1],
                           [0.191300162e1, 0.208141274e1, 0.103505092e2],
                           [0.427920172e3, 0.644289068e3, 0.480386523e3]])
        table15 = table15.T

        for state, properties in zip(states, table15):
            r = Region2(state=state)
            p = [r.v, r.h, r.u, r.s, r.cp, r.w]
            np.testing.assert_almost_equal(properties, p, decimal=5)

class TestRegion3(unittest.TestCase):

    def test_h_3ab(self):
        p = 25
        h_calc = Region3.h_3ab(p=p)
        self.assertAlmostEqual(2.095936454e3, h_calc)

    def test_subregion_ph(self):
        # From table 5 in supplementary release 2014.
        regions = {'a': {'h': [1700, 2000, 2100], 'p': [20, 50, 100]},
                   'b': {'p': [20, 50, 100], 'h': [2500, 2400, 2700]}}

        for reg, vals in regions.items():
            hs = vals['h']
            ps = vals['p']

            for p, h in zip(ps, hs):
                subregion_calc = Region3.subregion(p=p, h=h)
                self.assertEqual(reg, subregion_calc)
    
    def test_subregion_ps(self):
        # From table 12 in supplementary release 2014..
        regions = {'a': {'s': [3.8, 3.6, 4.0], 'p': [20, 50, 100]},
                   'b': {'p': [20, 50, 100], 's': [5, 4.5, 5]}}

        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']

            for p, s in zip(ps, ss):
                subregion_calc = Region3.subregion(p=p, s=s)
                self.assertEqual(reg, subregion_calc)

    def test_subregion_hs(self):
        # From table 29.
        regions = {'a': {'h': [2800, 2800, 4100], 's': [6.5, 9.5, 9.5]},
                   'b': {'h': [2800, 3600, 3600], 's': [6, 6, 7]},
                   'c': {'h': [2800, 2800, 3400], 's': [5.1, 5.8, 5.8]}}

        for reg, vals in regions.items():
            ss = vals['s']
            hs = vals['h']

            for h, s in zip(hs, ss):
                subregion_calc = Region2.subregion(h=h, s=s)
                self.assertEqual(reg, subregion_calc)

    def test_range_validity(self):
        assert 1 == 2
        # Table 33.
        s = State(T=650, rho=500)
        s1 = State(T=650, rho=200)
        s2 = State(T=750, rho=500)

        self.assertTrue(s in Region2())
        self.assertTrue(s2 in Region2())
        self.assertTrue(s2 in Region2())

    def test_backwards_t_ph(self):
        # From table 5 supplementary.
        regions = {'a': {'h': [1700, 2000, 2100], 'p': [20, 50, 100], 'T': [6.293083892e2, 6.905718338e2, 7.336163014e2]},
                   'b': {'p': [20, 50, 100], 'h': [2500, 2400, 2700], 'T': [6.418418053e2, 7.351848618e2, 8.420460876e2]}}

        for reg, vals in regions.items():
            hs = vals['h']
            ps = vals['p']
            ts = vals['T']

            for p, h, T in zip(ps, hs, ts):
                T_calc = Region3().T_ph(p=p, h=h)
                self.assertAlmostEqual(T, T_calc, places=4)

    def test_backwards_v_ph(self):
        # From table 8 supplementary.
        regions = {'a': {'h': [1700, 2000, 2100], 'p': [20, 50, 100], 'v': [1.749903962e-3, 1.908139035e-3, 1.676229776e-3]},
                   'b': {'p': [20, 50, 100], 'h': [2500, 2400, 2700], 'v': [6.670547043e-3, 2.801244590e-3, 2.404234998e-3]}}

        for reg, vals in regions.items():
            hs = vals['h']
            ps = vals['p']
            vs = vals['v']

            for p, h, v in zip(ps, hs, vs):
                v_calc = Region3().v_ph(p=p, h=h)
                self.assertAlmostEqual(v, v_calc, places=4)

    def test_backwards_t_ps(self):
        # From table 12 supplementary.
        regions = {'a': {'s': [3.8, 3.6, 4], 'p': [20, 50, 100], 'T': [6.282959869e2, 6.297158726e2, 7.056880237e2]},
                   'b': {'p': [20, 50, 100], 's': [5, 4.5, 5], 'T': [6.401176443e2, 7.163687517e2, 8.474332825e2]}}

        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']
            ts = vals['T']

            for p, s, T in zip(ps, ss, ts):
                T_calc = Region3().T_ps(p=p, s=s)
                self.assertAlmostEqual(T, T_calc, places=4)

    def test_backwards_v_ps(self):
        # From table 15 supplementary.
        regions = {'a': {'s': [3.8, 3.6, 4], 'p': [20, 50, 100], 'v': [1.733791463e-3, 1.469680170e-3, 1.555893131e-3]},
                   'b': {'p': [20, 50, 100], 's': [5, 4.5, 5], 'v': [6.262101987e-3, 2.332634294e-3, 2.449610757e-3]}}

        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']
            vs = vals['v']

            for p, s, v in zip(ps, ss, vs):
                v_calc = Region3().v_ps(p=p, s=s)
                self.assertAlmostEqual(v, v_calc, places=4)
    
    def test_backwards_p_hs(self):
        # From table 5 of supplement for p(h,s) [2].
        regions = {'a': {'h': [1700, 2000, 2100], 's': [3.8, 4.2, 4.3], 'p': [2.555703246e1, 4.540873468e1, 6.078123340e1]},
                   'b': {'h': [2600, 2400, 2700], 's': [5.1, 4.7, 5.0], 'p': [3.434999263e1, 6.363924887e1, 8.839043281e1]}}
        
        for reg, vals in regions.items():
            ss = vals['s']
            ps = vals['p']
            hs = vals['h']

            for p, h, s in zip(ps, hs, ss):
                p_calc = Region3().p_hs(h=h, s=s)
                self.assertAlmostEqual(p, p_calc, places=4)

    def test_v_pt_temp_eqns_boundaries(self):
        r = Region3()
        regs = {'ab': {'P': 40, 'T': 6.930341408e2},
                'cd': {'P': 25, 'T': 6.493659208e2},
                'ef': {'P': 40, 'T': 7.139593992e2},
                'gh': {'P': 23, 'T': 6.498873759e2},
                'ij': {'P': 23, 'T': 6.515778091e2},
                'jk': {'P': 23, 'T': 6.558338344e2},
                'mn': {'P': 22.8, 'T': 6.496054133e2},
                'op': {'P': 22.8, 'T': 6.500106947e2},
                'qu': {'P': 22, 'T': 6.456355027e2},
                'rx': {'P': 22, 'T': 6.482622754e2}}
        for regs, data in regs.items():
            self.assertAlmostEqual(r._T_xx(data['P'], regs), data['T'], places=5)

    def test_property_accuracy(self):
        assert 1 == 2
        """Test the results from Table 15."""
        s = State(T=300, p=0.0035)
        s1 = State(T=700, p=0.0035)
        s2 = State(T=700, p=30)
        states = [s, s1, s2]

        table15 = np.array([[0.394913866e2, 0.923015898e2, 0.542946619e-2],
                           [0.254991145e4, 0.333568375e4, 0.263149474e4],
                           [0.241169160e4, 0.301262819e4, 0.246861076e4],
                           [0.852238967e1, 0.101749996e2, 0.517540298e1],
                           [0.191300162e1, 0.208141274e1, 0.103505092e2],
                           [0.427920172e3, 0.644289068e3, 0.480386523e3]])
        table15 = table15.T

        for state, properties in zip(states, table15):
            r = Region2(state=state)
            p = [r.v, r.h, r.u, r.s, r.cp, r.w]
            np.testing.assert_almost_equal(properties, p, decimal=5)



if __name__ == '__main__':
    unittest.main()
