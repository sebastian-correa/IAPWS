import unittest
from iapws import iapws97

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



if __name__ == '__main__':
    unittest.main()