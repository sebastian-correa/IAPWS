import unittest
from iapws import iapws97
import numpy as np

class GeneralTests(unittest.TestCase):
    
    def test_b23_eq_5(self):
        T = 0.623_150_000 * 10**3
        p = 0.165_291_643 * 10**2

        p_res = iapws97.b23(T=T)

        self.assertAlmostEqual(p, p_res, places=6)

    def test_b23_eq_6(self):
        T = 0.623_150_000 * 10**3
        p = 0.165_291_643 * 10**2

        T_res = iapws97.b23(p=p)

        self.assertAlmostEqual(T, T_res, places=6)
    
    def test_b23_exception(self):
        T = 1
        p = 1

        self.assertRaises(ValueError, iapws97.b23, T=T, p=p)

    def test_p_s_eq_30(self):
        tees = [300, 500, 600]
        pss = [0.353_658_941e-2, 0.263_889_776e1, 0.123_443_146e2]

        for T, ps in zip(tees, pss):
            self.assertAlmostEqual(iapws97._p_s(T), ps)
    
    def test_p_s_exeption(self):
        self.assertRaises(ValueError, iapws97._p_s, T=1000)

class TestRegion1(unittest.TestCase):

    def test_range_validity(self):
        s = iapws97.State(T=300, p=3)
        s1 = iapws97.State(T=300, p=80)
        s2 = iapws97.State(T=500, p=3)

        self.assertTrue(s in iapws97.Region1())
        self.assertTrue(s2 in iapws97.Region1())
        self.assertTrue(s2 in iapws97.Region1())

        s = iapws97.State(T=300, p=3)
        self.assertTrue(s in iapws97.Region1())

    def test_backwards_t_ph(self):
        pees = [3, 80, 80]
        hs = [500, 500, 1500]
        tees = [0.391_798_509e3, 0.378_108_626e3, 0.611_041_229e3]

        for p, h, T in zip(pees, hs, tees):
            T_calc = iapws97.Region1().T_ph(p, h)
            self.assertAlmostEqual(T, T_calc, places=6)

    def test_backwards_t_ps(self):
        pees = [3, 80, 80]
        ss = [0.5, 0.5, 3]
        tees = [0.307_842_258e3, 0.309_979_785e3, 0.565_899_909e3]

        for p, s, T in zip(pees, ss, tees):
            T_calc = iapws97.Region1().T_ps(p, s)
            self.assertAlmostEqual(T, T_calc, places=6)
    
    def test_backwards_p_hs(self):
        hs = [0.001, 90, 1500]
        ss = [0, 0, 3.4]
        pees = [9.800_980_612e-4, 9.192_954_727e1, 5.868_294_423e1]

        for h, s, p in zip(hs, ss, pees):
            p_calc = iapws97.Region1().p_hs(h, s)
            self.assertAlmostEqual(p, p_calc, places=6)

    def test_property_accuracy(self):
        """Test the results from Table 5."""
        s = iapws97.State(T=300, p=3)
        s1 = iapws97.State(T=300, p=80)
        s2 = iapws97.State(T=500, p=3)
        states = [s, s1, s2]

        table5 = np.array([[0.100215168e-2, 0.971180894e-3, 0.120241800e-2],
                           [0.115331273e3, 0.184142828e3, 0.975542239e3],
                           [0.112324818e3, 0.106448356e3, 0.971934985e3],
                           [0.392294792, 0.368563852, 0.258041912e1],
                           [0.417301218e1, 0.401008987e1, 0.465580682e1],
                           [0.150773921e4, 0.163469054e4, 0.124071337e4]])
        table5 = table5.T

        for state, properties in zip(states, table5):
            r = iapws97.Region1(state=state)
            p = [r.v, r.h, r.u, r.s, r.cp, r.w]
            np.testing.assert_almost_equal(properties, p, decimal=5)

if __name__ == '__main__':
    unittest.main()
