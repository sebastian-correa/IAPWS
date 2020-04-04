import warnings

import numpy as np
from typing import Optional
from scipy.optimize import newton
import math

from ._utils import State, Region, R, _p_s


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


    def __init__(self, T: Optional[float] = None, p: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None, state: Optional[State] = None):
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
            # Let the class instantiate so that someone can perform a `State in Region1()` check.
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
        Equation for saturation pressure as a function of temperature (equation 30), enthalpy (eqn.10 [4]) or entropy (eqn 11 [4]).
        Args:
            T: Temperature (K).
            h: Enthalpy (kJ/kg).
            s: Entropy (kJ/kg/K).
        Returns:
            The saturation pressure at the given temperature/enthalpy/entropy in MPa.
        References:
            [1], [4].
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
            eta = h / 2600
            return 22 * sum(entry['n'] * (eta - 1.02)**entry['I'] * (eta - 0.608)**entry['J'] for entry in Region4.table17_supp_ref4.values())
        elif s is not None and T is None and h is None:
            sigma = s / 5.2
            return 22 * sum(entry['n'] * (sigma - 1.03)**entry['I'] * (sigma - 0.699)**entry['J'] for entry in Region4.table19_supp_ref4.values())

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
    def T_sat(self, p: float) -> float:
        """
        Backwards equation for calculating Saturation Temperature as a function of pressure and enthalpy.
        Args:
            p: Pressure (MPa).
        Returns:
            Temperature (K).
        """
        beta = p**(1/4)
        g = Region4.table34[2] * beta**2 + Region4.table34[5] * beta + Region4.table34[8]
        f = Region4.table34[1] * beta**2 + Region4.table34[4] * beta + Region4.table34[7]
        e = beta ** 2 + Region4.table34[3] * beta + Region4.table34[6]
        d = 2 * g / (-f - np.sqrt(f**2 - 4*e*g))
        ts = 1/2 * (Region4.table34[10] + d - np.sqrt( (Region4.table34[10] + d)**2 - 4*(Region4.table34[9] + Region4.table34[10]*d) ))
        return ts

    def h_sat(self, x: int = 0, p: Optional[float] = None, T: Optional[float] = None) -> float:
        """
        Calculate the saturation enthalpy from either pressure or Temperature.
        Args:
            phase: Specify if the saturation enthalpy of 'liquid' (x=0) or 'steam' (x=1) should be calculated.
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
            h0 = (2.084e3 + 1.670858218e3) / 2
        elif x == 1:
            h0 = (2.084e3 + 2.563592004e3) / 2
        else:
            warnings.warn('Quality (x) should only be 0 or 1. Otherwise, water is not saturated.', RuntimeWarning)
        return newton(f, h0)


r = Region4()
hs = [1700, 2000, 2400]
pss = [1.724175718e1, 2.193442957e1, 2.018090839e1]
xx = [0,0,1]

for h, ps, x in zip(hs, pss, xx):
    print(h, r.h_sat(x=x, p=ps))
