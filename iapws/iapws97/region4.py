import warnings

import numpy as np
from typing import Optional
from scipy.optimize import newton
import math

from ._utils import State, Region, R, s_c

hp = 1.670858218e3
hpp = 2.563592004e3
sp = 3.778281340
spp = 5.210887825


class Region4(Region):
    """
    Region1 implements Region1 of the IAPWS97 standard.

    Methods:

    Class attributes:

    """
    table34 = {1: 0.116_705_214_527_67e4,
               2: -0.724_213_167_032_06e6,
               3: -0.170_738_469_400_92e2,
               4: 0.120_208_247_024_70e5,
               5: -0.323_255_503_223_33e7,
               6: 0.149_151_086_135_30e2,
               7: -0.482_326_573_615_91e4,
               8: 0.405_113_405_420_57e6,
               9: -0.238_555_575_678_49,
               10: 0.650_175_348_447_98e3}

    table17_supp_ref4 = {1: {'I': 0, 'J': 0, 'n': 0.600073641753024},
                         2: {'I': 1, 'J': 1, 'n': -0.936203654849857e1},
                         3: {'I': 1, 'J': 3, 'n': 0.246590798594147e2},
                         4: {'I': 1, 'J': 4, 'n': -0.107014222858224e3},
                         5: {'I': 1, 'J': 36, 'n': -0.915821315805768e14},
                         6: {'I': 5, 'J': 3, 'n': -0.862332011700662e4},
                         7: {'I': 7, 'J': 0, 'n': -0.235837344740032e2},
                         8: {'I': 8, 'J': 24, 'n': 0.252304969384128e18},
                         9: {'I': 14, 'J': 16, 'n': -0.389718771997719e19},
                         10: {'I': 20, 'J': 16, 'n': -0.333775713645296e23},
                         11: {'I': 22, 'J': 3, 'n': 0.356499469636328e11},
                         12: {'I': 24, 'J': 18, 'n': -0.148547544720641e27},
                         13: {'I': 28, 'J': 8, 'n': 0.330611514838798e19},
                         14: {'I': 36, 'J': 24, 'n': 0.813641294467829e38}}

    table19_supp_ref4 = {1: {'I': 0, 'J': 0, 'n': 0.639767553612785},
                         2: {'I': 1, 'J': 1, 'n': -0.129727445396014e2},
                         3: {'I': 1, 'J': 32, 'n': -0.224595125848403e16},
                         4: {'I': 4, 'J': 7, 'n': 0.177466741801846e7},
                         5: {'I': 12, 'J': 4, 'n': 0.717079349571538e10},
                         6: {'I': 12, 'J': 14, 'n': -0.378829107169011e18},
                         7: {'I': 16, 'J': 36, 'n': -0.955586736431328e35},
                         8: {'I': 24, 'J': 10, 'n': 0.187269814676188e24},
                         9: {'I': 28, 'J': 0, 'n': 0.119254746466473e12},
                         10: {'I': 32, 'J': 18, 'n': 0.110649277244882e37}}

    table28_supp_ref5 = {0: {'I': 0, 'J': 0, 'n': 0.179882673606601},
                         1: {'I': 0, 'J': 3, 'n': -0.267507455199603},
                         2: {'I': 0, 'J': 12, 'n': 1.162767226126},
                         3: {'I': 1, 'J': 0, 'n': 0.147545428713616},
                         4: {'I': 1, 'J': 1, 'n': -0.512871635973248},
                         5: {'I': 1, 'J': 2, 'n': 0.421333567697984},
                         6: {'I': 1, 'J': 5, 'n': 0.56374952218987},
                         7: {'I': 2, 'J': 0, 'n': 0.429274443819153},
                         8: {'I': 2, 'J': 5, 'n': -3.3570455214214},
                         9: {'I': 2, 'J': 8, 'n': 10.8890916499278},
                         10: {'I': 3, 'J': 0, 'n': -0.248483390456012},
                         11: {'I': 3, 'J': 2, 'n': 0.30415322190639},
                         12: {'I': 3, 'J': 3, 'n': -0.494819763939905},
                         13: {'I': 3, 'J': 4, 'n': 1.07551674933261},
                         14: {'I': 4, 'J': 0, 'n': 0.0733888415457688},
                         15: {'I': 4, 'J': 1, 'n': 0.0140170545411085},
                         16: {'I': 5, 'J': 1, 'n': -0.106110975998808},
                         17: {'I': 5, 'J': 2, 'n': 0.0168324361811875},
                         18: {'I': 5, 'J': 4, 'n': 1.25028363714877},
                         19: {'I': 5, 'J': 16, 'n': 1013.16840309509},
                         20: {'I': 6, 'J': 6, 'n': -1.51791558000712},
                         21: {'I': 6, 'J': 8, 'n': 52.4277865990866},
                         22: {'I': 6, 'J': 22, 'n': 23049.5545563912},
                         23: {'I': 8, 'J': 1, 'n': 0.0249459806365456},
                         24: {'I': 10, 'J': 20, 'n': 2107964.67412137},
                         25: {'I': 10, 'J': 36, 'n': 366836848.613065},
                         26: {'I': 12, 'J': 24, 'n': -144814105.365163},
                         27: {'I': 14, 'J': 1, 'n': -0.0017927637300359},
                         28: {'I': 14, 'J': 28, 'n': 4899556021.00459},
                         29: {'I': 16, 'J': 12, 'n': 471.262212070518},
                         30: {'I': 16, 'J': 32, 'n': -82929439019.8652},
                         31: {'I': 18, 'J': 14, 'n': -1715.45662263191},
                         32: {'I': 18, 'J': 22, 'n': 3557776.82973575},
                         33: {'I': 18, 'J': 36, 'n': 586062760258.436},
                         34: {'I': 20, 'J': 24, 'n': -12988763.5078195},
                         35: {'I': 28, 'J': 36, 'n': 31724744937.1057}}


    def __init__(self, x: Optional[float] = None, T: Optional[float] = None, p: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None, state: Optional[State] = None):
        """
        If all parameters are None (their default), then an empty instance is instanciated. This is to that a `State in Region4` check can be performed easily.
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
            # Let the class instantiate so that someone can perform a `State in Region4()` check.
        elif p and T:
            self._state.T = T
            self._state.p = p
        elif p and h:
            self._state.T = self.T_sat(p)
            self._state.p = p
            self._state.h = h
        elif p and s:
            self._state.T = self.T_sat(p)
            self._state.p = p
            self._state.s = s
        elif T and h:
            self._state.T = T
            self._state.p = self.p_sat(T=T)
            self._state.h = h
        elif T and s:
            self._state.T = T
            self._state.p = self.p_sat(T=T)
            self._state.s = s
        elif h and s:
            self._state.p = self.p_sat(h=h)
            self._state.T = self.T_sat(p)
            self._state.s = s
            self._state.h = h
        else:
            raise ValueError('You should only pass one of the following combinations to determine a state in Reg1: (p,T) (p, h), (p, s), (T, h), (T,s), (h, s).')


        if calc:
            if not self._state in self:
                # Find region number and return it.
                pass

            self._state.ders = None

            self._state.v = None
            self._state.rho = 1 / self._state.v
            self._state.u = R * T * (tau*gt - _pi*gp)
            self._state.s = self._state.s if self._state.s is not None else None
            self._state.h = self._state.h if self._state.h is not None else None
            self._state.cp = None
            self._state.cv = None
            self._state.w = None
        else:
            self._state = State()

    def __contains__(self, other: State) -> bool:
        """
        Overrides the behaviour of the `in` operator to facilitate a `State in Region` query.
        In enthalpy, the range is [1.670858218e3, 2.563592004e3] according to [4].
        In entropy, the range is  [3.778281340, 5.210887825] according to [4].
        """
        if not isinstance(other, State):
            return False
        else:
            return math.isclose(self.p_sat(T=T), other.p) and math.isclose(self.T_sat(p=p), other.T)

    def __repr__(self) -> str:
        return f'Region4(p={self.p}, T={self.T})'

    @staticmethod
    def base_eqn(T: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None) -> float:
        """
        Equation for saturation pressure as a function of temperature (equation 30), enthalpy (eqn.10 [4]), entropy (eqn 11 [4]) or enthalpy and entropy (eqn 9 [5]).
        Args:
            T: Temperature (K).
            h: Enthalpy (kJ/kg).
            s: Entropy (kJ/kg/K).
        Returns:
            The saturation pressure at the given temperature/enthalpy/entropy in MPa.
        References:
            [1], [4], [5].
        """
        if T is not None and h is None and s is None:
            if not 273.15 <= T <= 647.096:
                warnings.warn(f'T must be in the range [273.15, 647.096]. {T} given.', RuntimeWarning)
            z = T + Region4.table34[9] / (T - Region4.table34[10])
            A = z ** 2 + Region4.table34[1] * z + Region4.table34[2]
            B = Region4.table34[3] * z ** 2 + Region4.table34[4] * z + Region4.table34[5]
            C = Region4.table34[6] * z ** 2 + Region4.table34[7] * z + Region4.table34[8]
            p = 2 * C / (-B + np.sqrt(B ** 2 - 4 * A * C))
            return p ** 4
        elif h is not None and T is None and s is None:
            if hp <= h <= hpp:
                eta = h / 2600
                return 22 * sum(entry['n'] * (eta - 1.02)**entry['I'] * (eta - 0.608)**entry['J'] for entry in Region4.table17_supp_ref4.values())
            else:
                raise NotImplementedError('Try also supplying a value for s.')
        elif s is not None and T is None and h is None:
            if sp <= s <= spp:
                sigma = s / 5.2
                return 22 * sum(entry['n'] * (sigma - 1.03)**entry['I'] * (sigma - 0.699)**entry['J'] for entry in Region4.table19_supp_ref4.values())
            else:
                raise NotImplementedError('Try also supplying a value for h.')
        elif s is not None and h is not None and T is None:
            T = Region4.T_sat(h=h, s=s)
            return Region4.base_eqn(T=T)


    @staticmethod
    def p_sat(T: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None) -> float:
        """Alias for `self.base_eqn`"""
        return Region4.base_eqn(T, h, s)

    #############################################################
    ####################### Properties ##########################
    #############################################################
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
    @staticmethod
    def T_sat(p: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None) -> float:
        """
        Backwards equation for calculating Saturation Temperature as a function of pressure or enthalpy and entropy.
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
            s: Entropy (kJ/kg/K).
        Returns:
            Temperature (K).
        References:
            [1], [5].
        """
        if p is not None and (h is None and s is None):
            beta = p**(1/4)
            g = Region4.table34[2] * beta**2 + Region4.table34[5] * beta + Region4.table34[8]
            f = Region4.table34[1] * beta**2 + Region4.table34[4] * beta + Region4.table34[7]
            e = beta ** 2 + Region4.table34[3] * beta + Region4.table34[6]
            d = 2 * g / (-f - np.sqrt(f**2 - 4*e*g))
            ts = 1/2 * (Region4.table34[10] + d - np.sqrt( (Region4.table34[10] + d)**2 - 4*(Region4.table34[9] + Region4.table34[10]*d) ))
        elif (h is not None and s is not None) and p is None:
            # Eqn 9 [5].
            if s >= spp:
                eta = h / 2800
                sigma = s / 9.2
                ts = 550 * sum(entry['n'] * (eta - 0.119)**entry['I'] * (sigma - 1.07)**entry['J'] for entry in Region4.table28_supp_ref5.values())
            else:
                raise NotImplementedError(f's should be >= {spp}. {s} given.')
        return ts

    def h_sat(self, x: int = 0, p: Optional[float] = None, T: Optional[float] = None) -> float:
        """
        Calculate the saturation enthalpy from either pressure or Temperature.
        Args:
            x: Specify if the saturation enthalpy of 'liquid' (x=0) or 'steam' (x=1) should be calculated.
            p: Pressure (MPa).
            T: Temperature (K).
        Returns:
            Enthalpy of saturation in kJ/kg.
        References:
            [4].
        Notes:
            This function iterates on p_sat or T_sat. In order to try to find the correct enthalpy (as a root an equation),
            the initial point is chosen to be the average of h_c and h'' for saturated vapor or the average of h_c and h'
            for saturated liquid.
            h_c taken from Wolfram|Alpha because it doesn't have to bee too exact.
            Please see Figure 3 in [4].
        """
        if T is not None and p is None:
            p = self.p_sat(T=T)

        def f(h):
            return self.p_sat(h=h) - p

        if x == 0:
            h0 = (2.084e3 + hp) / 2
        elif x == 1:
            h0 = (2.084e3 + hpp) / 2
        else:
            warnings.warn('Quality (x) should only be 0 or 1. Otherwise, water is not saturated.', RuntimeWarning)
        return newton(f, h0)

    def s_sat(self, x: int = 0, p: Optional[float] = None, T: Optional[float] = None) -> float:
        """
        Calculate the saturation entropy from either pressure or Temperature.
        Args:
            x: Specify if the saturation enthalpy of 'liquid' (x=0) or 'steam' (x=1) should be calculated.
            p: Pressure (MPa).
            T: Temperature (K).
        Returns:
            Entropy of saturation in kJ/kg/K.
        References:
            [4].
        Notes:
            This function iterates on p_sat or T_sat. In order to try to find the correct enthalpy (as a root an equation),
            the initial point is chosen to be the average of s_c and s'' for saturated vapor or the average of s_c and s'
            for saturated liquid.
            Please see Figure 4 in [4].
        """
        if T is not None and p is None:
            p = self.p_sat(T=T)

        def f(s):
            return self.p_sat(s=s) - p

        if x == 0:
            s0 = (s_c + sp) / 2
        elif x == 1:
            s0 = (s_c + spp) / 2
        else:
            warnings.warn('Quality (x) should only be 0 or 1. Otherwise, water is not saturated.', RuntimeWarning)
        return newton(f, s0)
