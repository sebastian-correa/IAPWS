import numpy as np
from typing import Optional
from collections import defaultdict
from scipy.optimize import  fsolve

from ._utils import State, Region, R, _p_s

class Region1(Region):
    """
    Region1 implements Region1 of the IAPWS97 standard.

    Methods:
        __init__
        __contains__
        base_eqn
        specific_gibbs_free_energy
        base_der_pi_const_tau
        base_der_tau_const_pi
        base_der2_pipi_const_tau
        base_der_tautau_const_pi
        base_der2_pitau

    Class attributes:
        table2
        table6
        table8
        table2_supp

        gamma
        gamma_pi
        gamma_tau
        gamma_pipi
        gamma_tautau
        gamma_pitau
        T
        p
        P
        v
        rho
        u
        s
        h
        cp
        cv
        w
    """
    table2 = {1: {'I': 0, 'J': -2, 'n': 0.146_329_712_131_67},
              18: {'I': 2, 'J': 3, 'n': -0.441_418_453_308_46e-5},
              2: {'I': 0, 'J': -1, 'n': -0.845_481_871_691_14},
              19: {'I': 2, 'J': 17, 'n': -0.726_949_962_975_94e-15},
              3: {'I': 0, 'J': 0, 'n': -0.375_636_036_720_40e1},
              20: {'I': 3, 'J': -4, 'n': -0.316_796_448_450_54e-4},
              4: {'I': 0, 'J': 1, 'n': 0.338_551_691_683_85e1},
              21: {'I': 3, 'J': 0, 'n': -0.282_707_979_853_12e-5},
              5: {'I': 0, 'J': 2, 'n': -0.957_919_633_878_72},
              22: {'I': 3, 'J': 6, 'n': -0.852_051_281_201_03e-9},
              6: {'I': 0, 'J': 3, 'n': 0.157_720_385_132_28},
              23: {'I': 4, 'J': -5, 'n': -0.224_252_819_080_00e-5},
              7: {'I': 0, 'J': 4, 'n': -0.166_164_171_995_01e-1},
              24: {'I': 4, 'J': -2, 'n': -0.651_712_228_956_01e-6},
              8: {'I': 0, 'J': 5, 'n': 0.812_146_299_835_68e-3},
              25: {'I': 4, 'J': 10, 'n': -0.143_417_299_379_24e-12},
              9: {'I': 1, 'J': -9, 'n': 0.283_190_801_238_04e-3},
              26: {'I': 5, 'J': -8, 'n': -0.405_169_968_601_17e-6},
              10: {'I': 1, 'J': -7, 'n': -0.607_063_015_658_74e-3},
              27: {'I': 8, 'J': -11, 'n': -0.127_343_017_416_41e-8},
              11: {'I': 1, 'J': -1, 'n': -0.189_900_682_184_19e-1},
              28: {'I': 8, 'J': -6, 'n': -0.174_248_712_306_34e-9},
              12: {'I': 1, 'J': 0, 'n': -0.325_297_487_705_05e-1},
              29: {'I': 21, 'J': -29, 'n': -0.687_621_312_955_31e-18},
              13: {'I': 1, 'J': 1, 'n': -0.218_417_171_754_14e-1},
              30: {'I': 23, 'J': -31, 'n': 0.144_783_078_285_21e-19},
              14: {'I': 1, 'J': 3, 'n': -0.528_383_579_699_30e-4},
              31: {'I': 29, 'J': -38, 'n': 0.263_357_816_627_95e-22},
              15: {'I': 2, 'J': -3, 'n': -0.471_843_210_732_67e-3},
              32: {'I': 30, 'J': -39, 'n': -0.119_476_226_400_71e-22},
              16: {'I': 2, 'J': 0, 'n': -0.300_017_807_930_26e-3},
              33: {'I': 31, 'J': -40, 'n': 0.182_280_945_814_04e-23},
              17: {'I': 2, 'J': 1, 'n': 0.476_613_939_069_87e-4},
              34: {'I': 32, 'J': -41, 'n': -0.935_370_872_924_58e-25}}

    table6 = {1: {'I': 0, 'J': 0, 'n': -0.238_724_899_245_21e3},
            2: {'I': 0, 'J': 1, 'n': 0.404_211_886_379_45e3},
            3: {'I': 0, 'J': 2, 'n': 0.113_497_468_817_18e3},
            4: {'I': 0, 'J': 6, 'n': -0.584_576_160_480_39e1},
            5: {'I': 0, 'J': 22, 'n': -0.152_854_824_131_40e-3},
            6: {'I': 0, 'J': 32, 'n': -0.108_667_076_953_77e-5},
            7: {'I': 1, 'J': 0, 'n': -0.133_917_448_726_02e2},
            8: {'I': 1, 'J': 1, 'n': 0.432_110_391_835_59e2},
            9: {'I': 1, 'J': 2, 'n': -0.540_100_671_705_06e2},
            10: {'I': 1, 'J': 3, 'n': 0.305_358_922_039_16e2},
            11: {'I': 1, 'J': 4, 'n': -0.659_647_494_236_38e1},
            12: {'I': 1, 'J': 10, 'n': 0.939_654_008_783_63e-2},
            13: {'I': 1, 'J': 32, 'n': 0.115_736_475_053_40e-6},
            14: {'I': 2, 'J': 10, 'n': -0.258_586_412_820_73e-4},
            15: {'I': 2, 'J': 32, 'n': -0.406_443_630_847_99e-8},
            16: {'I': 3, 'J': 10, 'n': 0.664_561_861_916_35e-7},
            17: {'I': 3, 'J': 32, 'n': 0.806_707_341_030_27e-10},
            18: {'I': 4, 'J': 32, 'n': -0.934_777_712_139_47e-12},
            19: {'I': 5, 'J': 32, 'n': 0.582_654_420_206_01e-14},
            20: {'I': 6, 'J': 32, 'n': -0.150_201_859_535_03e-16}}

    table8 = {1: {'I': 0, 'J': 0, 'n': 0.174_782_680_583_07e3},
              2: {'I': 0, 'J': 1, 'n': 0.348_069_308_928_73e2},
              3: {'I': 0, 'J': 2, 'n': 0.652_925_849_784_55e1},
              4: {'I': 0, 'J': 3, 'n': 0.330_399_817_754_89},
              5: {'I': 0, 'J': 11, 'n': -0.192_813_829_231_96e-6},
              6: {'I': 0, 'J': 31, 'n': -0.249_091_972_445_73e-22},
              7: {'I': 1, 'J': 0, 'n': -0.261_076_364_893_32},
              8: {'I': 1, 'J': 1, 'n': 0.225_929_659_815_86},
              9: {'I': 1, 'J': 2, 'n': -0.642_564_633_952_26e-1},
              10: {'I': 1, 'J': 3, 'n': 0.788_762_892_705_26e-2},
              11: {'I': 1, 'J': 12, 'n': 0.356_721_106_073_66e-9},
              12: {'I': 1, 'J': 31, 'n': 0.173_324_969_948_95e-23},
              13: {'I': 2, 'J': 0, 'n': 0.566_089_006_548_37e-3},
              14: {'I': 2, 'J': 1, 'n': -0.326_354_831_397_17e-3},
              15: {'I': 2, 'J': 2, 'n': 0.447_782_866_906_32e-4},
              16: {'I': 2, 'J': 9, 'n': -0.513_221_569_085_07e-9},
              17: {'I': 2, 'J': 31, 'n': -0.425_226_570_422_07e-25},
              18: {'I': 3, 'J': 10, 'n': 0.264_004_413_606_89e-12},
              19: {'I': 3, 'J': 32, 'n': 0.781_246_004_597_23e-28},
              20: {'I': 4, 'J': 32, 'n': -0.307_321_999_036_68e-30}}

    table2_supp = {1: {'I': 0, 'J': 0, 'n': -0.691997014660582},
                   2: {'I': 0, 'J': 1, 'n': -1.83612548787560e1},
                   3: {'I': 0, 'J': 2, 'n': -9.28332409297335e0},
                   4: {'I': 0, 'J': 4, 'n': 6.59639569909906e1},
                   5: {'I': 0, 'J': 5, 'n': -1.62060388912024e1},
                   6: {'I': 0, 'J': 6, 'n': 4.50620017338667e2},
                   7: {'I': 0, 'J': 8, 'n': 8.54680678224170e2},
                   8: {'I': 0, 'J': 14, 'n': 6.07523214001162e3},
                   9: {'I': 1, 'J': 0, 'n': 3.26487682621856e1},
                   10: {'I': 1, 'J': 1, 'n': -2.69408844582931e1},
                   11: {'I': 1, 'J': 4, 'n': -3.19947848334300e2},
                   12: {'I': 1, 'J': 6, 'n': -9.28354307043320e2},
                   13: {'I': 2, 'J': 0, 'n': 3.03634537455249e1},
                   14: {'I': 2, 'J': 1, 'n': -6.50540422444146e1},
                   15: {'I': 2, 'J': 10, 'n': -4.30991316516130e3},
                   16: {'I': 3, 'J': 4, 'n': -7.47512324096068e2},
                   17: {'I': 4, 'J': 1, 'n': 7.30000345529245e2},
                   18: {'I': 4, 'J': 4, 'n': 1.14284032569021e3},
                   19: {'I': 5, 'J': 0, 'n': -4.36407041874559e2}}

    def __init__(self, T: Optional[float] = None, p: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None, state: Optional[State] = None):
        """
        If all parameters are None (their default), then the point (p, T) = (3, 300) is instanciated. This point is chosen from Table 5 as a reference point.
        """
        params = [p, T, h, s]
        if state is not None and all(param is None for param in params):
            # Case: Only state is given. To handle: convert to normal case.
            p = state.p
            T = state.T
            h = state.h
            s = state.s
        elif state is not None and any(param is None for param in params):
            raise ValueError('If state is given, no values for p, t, h and s can be given.')

        params = [p, T, h, s]
        calc = True
        self._state = State()

        # Cases are handled such that after this if/elif/else block, p and T are always determined.
        if all(param is None for param in params) and state is None:
            calc = False
            # Let the class instantiate so that someone can perform a `State in Region1()` check.
        elif p and T:
            self._state.T = T
            self._state.p = p
        elif p and h:
            self._state.T = self.T_ph(p, h)
            self._state.p = p
            self._state.h = h
        elif p and s:
            self._state.T = self.T_ps(p, s)
            self._state.p = p
            self._state.s = s
        elif T and h:
            self._state.T = T
            self._state.p = self.p_Th(T=T, h=h)
            self._state.h = h
        elif T and s:
            self._state.T = T
            self._state.p = self.p_Ts(T=T, s=s)
            self._state.s = s
        elif h and s:
            self._state.p = self.p_hs(h, s)
            self._state.T = self.T_ph(p, h)
            self._state.s = s
            self._state.h = h
        else:
            raise ValueError('You should only pass one of the following combinations to determine a state in Reg1: (p,T) (p, h), (p, s), (T, h), (T,s), (h, s).')


        if calc:
            tau = 1386 / T
            _pi = p / 16.53

            if not self._state in self:
                # Find region number and return it.
                pass

            gg = Region1.base_eqn(T=T, p=p)
            gp = Region1.base_der_pi_const_tau(T=T, p=p)
            gt = Region1.base_der_tau_const_pi(T=T, p=p)
            gpp = Region1.base_der2_pipi_const_tau(T=T, p=p)
            gtt = Region1.base_der2_tautau_const_pi(T=T, p=p)
            gpt = Region1.base_der2_pitau(T=T, p=p)

            self._state.ders = defaultdict(float, gamma=gg, gamma_pi=gp, gamma_tau=gt, gamma_pipi=gpp, gamma_tautau=gtt, gamma_pitau=gpt)

            self._state.v = _pi * gp * R * T / p / 1000  # R*T/p has units of 1000 m^3/kg.
            self._state.rho = 1 / self._state.v
            self._state.u = R * T * (tau*gt - _pi*gp)
            self._state.s = self._state.s if self._state.s is not None else R * (tau*gt - gg)
            self._state.h = self._state.h if self._state.h is not None else R * T * tau * gt
            self._state.cp = R * -tau**2 * gtt
            self._state.cv = R * (-tau**2 * gtt + (gp-tau*gpt)**2 / gpp)
            self._state.w = np.sqrt(1000 * R * T * gp**2 / ((gp-tau*gpt)**2 / (tau**2 * gtt) - gpp))  # 1000 is a conversion factor: sqrt(kJ/kg) = sqrt(1000 m/s) -> sqrt(1000) m/s
        else:
            self._state = State()

    def __contains__(self, other: State) -> bool:
        """
        Overrides the behaviour of the `in` operator to facilitate a `State in Region` query.
        """
        if not isinstance(other, State):
            return False
        else:
            return 273.15 <= other.T <= 623.15 and _p_s(T=other.T) <= other.p <= 100

    def __repr__(self) -> str:
        return f'Region1(p={self.p}, T={self.T})'

    @staticmethod
    def base_eqn(T: float, p: float) -> float:
        """
        Dimensionless specific Gibbs free energy (eq. 7).
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Dimensionless specific Gibbs free energy.
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(entry['n'] * (7.1 - _pi)**entry['I'] * (tau - 1.222)**entry['J'] for entry in Region1.table2.values())

    @staticmethod
    def specific_gibbs_free_energy(T: float, p: float) -> float:
        """Alias for `self.base_eqn`"""
        return Region1.base_eqn(T, p) * R * T

    #############################################################
    ################## FIRST ORDER DERIVATIVES ##################
    #############################################################
    @staticmethod
    def base_der_pi_const_tau(T: float, p: float) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        Also known as gamma_pi.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(- entry['n'] * entry['I'] * (7.1 - _pi)**(entry['I'] - 1) * (tau - 1.222)**entry['J'] for entry in Region1.table2.values())

    @staticmethod
    def base_der_tau_const_pi(T: float, p: float) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`.
        Also known as gamma_tau.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(entry['n'] * (7.1 - _pi)**entry['I'] * entry['J'] * (tau - 1.222)**(entry['J'] - 1) for entry in Region1.table2.values())

    #############################################################
    ################# SECOND ORDER DERIVATIVES ##################
    #############################################################
    @staticmethod
    def base_der2_pipi_const_tau(T: float, p: float) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(entry['n'] * entry['I'] * (entry['I'] - 1) * (7.1 - _pi)**(entry['I'] - 2) * (tau - 1.222)**entry['J'] for entry in Region1.table2.values())

    @staticmethod
    def base_der2_tautau_const_pi(T: float, p: float) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(entry['n'] * (7.1 - _pi)**entry['I'] * entry['J'] * (entry['J'] -1) * (tau - 1.222)**(entry['J'] - 2) for entry in Region1.table2.values())

    @staticmethod
    def base_der2_pitau(T: float, p: float) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(- entry['n'] * entry['I'] * (7.1 - _pi)**(entry['I'] - 1) * entry['J'] * (tau - 1.222)**(entry['J'] - 1) for entry in Region1.table2.values())

    #############################################################
    ####################### Properties ##########################
    #############################################################
    @property
    def gamma(self) -> float:
        """Dimensionless specific Gibbs free energy (eq. 7)."""
        return self._state.ders['gamma']

    @property
    def gamma_pi(self) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`"""
        return self._state.ders['gamma_pi']

    @property
    def gamma_tau(self) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`"""
        return self._state.ders['gamma_tau']

    @property
    def gamma_pipi(self) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`"""
        return self.self._state.ders['gamma_pipi']

    @property
    def gamma_tautau(self) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`"""
        return self.self._state.ders['gamma_tautau']

    @property
    def gamma_pitau(self) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`"""
        return self.self._state.ders['gamma_pitau']

    @property
    def T(self) -> float:
        """Temperature of state (K)"""
        return self._state.T

    @property
    def p(self) -> float:
        """Pressure of state (MPa)"""
        return self._state.p

    @property
    def P(self) -> float:
        """Pressure of state (MPa)"""
        return self._state.p

    @property
    def v(self) -> float:
        """Specific volume in m^3/kg"""
        return self._state.v

    @property
    def rho(self) -> float:
        """Density in kg/m^3"""
        return self._state.rho

    @property
    def u(self) -> float:
        """Specific internal energy in kJ/kg"""
        return self._state.u

    @property
    def s(self) -> float:
        """Specific entropy in kJ/kg/K"""
        return self._state.s

    @property
    def h(self) -> float:
        """Specific enthalpy in kJ/kg"""
        return self._state.h

    @property
    def cp(self) -> float:
        """Specific isobaric heat capacity kJ/kg/K"""
        return self._state.cp

    @property
    def cv(self) -> float:
        """Specific isochoric heat capacity kJ/kg/K"""
        return self._state.cv

    @property
    def w(self) -> float:
        """Speed of sound in m/s"""
        return self._state.w

    #############################################################
    ####################### Backwards ###########################
    #############################################################
    def T_ph(self, p: float, h: float) -> float:
        """
        Backwards equation 11 for calculating Temperature as a function of pressure and enthalpy.
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
        Returns:
            Temperature (K).
        """
        eta = h/2500
        T = sum(entry['n'] * p**entry['I']*(eta + 1)**entry['J'] for entry in Region1.table6.values())
        if State(p=p, T=T) in self:
            return T
        else:
            raise ValueError(f'State out of bounds. {T}')

    def T_ps(self, p: float, s: float) -> float:
        """
        Backwards equation 13 for calculating Temperature as a function of pressure and entropy.
        Args:
            p: Pressure (MPa).
            s: Entropy (kJ/kg/K).
        Returns:
            Temperature (K).
        """
        T = sum(entry['n'] * p**entry['I'] * (s + 2)**entry['J'] for entry in Region1.table8.values())
        if State(p=p, T=T) in self:
            return T
        else:
            #TODO: Suggest a region.
            raise ValueError(f'State out of bounds. {T}')

    def T_hs(self, h: float, s: float) -> float:
        """
        Backwards equation for calculating Temperature as a function of enthalpy and entropy.
        Args:
            h: Enthalpy (kJ/kg).
            s: Entropy (kJ/kg/K).
        Returns:
            Temperature (K).
        References:
            http://www.iapws.org/relguide/Supp-VPT3-2016.pdf
        """
        p = self.p_hs(h, s)
        return self.T_ph(p, h)

    def p_hs(self, h: float, s: float) -> float:
        """
        Backwards equation 1 from [1] for calculating pressure as a function of enthalpy and entropy.
        Args:
            h: Enthalpy (kJ/kg).
            s: Entropy (kJ/kg/K).
        Returns:
            Pressure (MPa).
        References:
            http://www.iapws.org/relguide/Supp-VPT3-2016.pdf
        """
        eta = h / 3400
        sigma = s / 7.6

        p = 100 * sum(entry['n'] * (eta + 0.05)**entry['I'] * (sigma + 0.05)**entry['J'] for entry in Region1.table2_supp.values())
        T = self.T_ps(p, s)
        if State(p=p, T=T) in self:
            return p
        else:
            raise ValueError(f'State out of bounds.')

    def p_Th(self, T: float, h: float) -> float:
        """
        Backwards equation for calculating pressure as a function of Temperature and enthalpy.
        Beware that this calculation might be time consuming as it is performing iteration (no backwards equation is provided by IAPWS).
        Args:
            T: Temperature (K).
            h: Enthalpy (kJ/kg).
        Returns:
            Pressure (MPa).
        """

        def f(p):
            return self.T_ph(p, h) - T

        p0 = (_p_s(T=T) + 100) / 2
        p = fsolve(f, p0)[0]  # initial p guess from region boundaries (see __contains__).

        if State(p=p, T=T) in self:
            return p
        else:
            raise ValueError(f'State out of bounds.')

    def p_Ts(self, T: float, s: float) -> float:
        """
        Backwards equation for calculating pressure as a function of Temperature and Entropy.
        Beware that this calculation might be time consuming as it is performing iteration (no backwards equation is provided by IAPWS).
        Args:
            T: Temperature (K).
            s: Entropy (kJ/kg/K).
        Returns:
            Pressure (MPa).
        """

        def f(p):
            return self.T_ps(p, s) - T

        p0 = (_p_s(T=T) + 100) / 2
        p = fsolve(f, p0)[0]  # initial p guess from region boundaries (see __contains__).

        if State(p=p, T=T) in self:
            return p
        else:
            raise ValueError(f'State out of bounds.')
