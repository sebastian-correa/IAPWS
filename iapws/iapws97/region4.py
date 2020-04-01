import warnings

import numpy as np
from typing import Optional
from collections import defaultdict
from scipy.optimize import fsolve, newton

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
            return False  # TODO: What?

    def __repr__(self) -> str:
        return f'Region4(p={self.p}, T={self.T})'

    @staticmethod
    def base_eqn(T: float) -> float:
        """
        Equation for saturation pressure as a function of temperature (equation 30).
        Args:
            T: Temperature in K.
        Returns:
            The saturation pressure at the given temperature in MPa.
        """
        if not 273.15 <= T <= 647.096:
            warnings.warn(f'T must be in the range [273.15, 647.096]. {T} given.', RuntimeWarning)
        z = T + Region4.table34[9] / (T - Region4.table34[10])
        A = z ** 2 + Region4.table34[1] * z + Region4.table34[2]
        B = Region4.table34[3] * z ** 2 + Region4.table34[4] * z + Region4.table34[5]
        C = Region4.table34[6] * z ** 2 + Region4.table34[7] * z + Region4.table34[8]
        p = 2 * C / (-B + np.sqrt(B ** 2 - 4 * A * C))
        return p ** 4

    @staticmethod
    def p_sat(T: float) -> float:
        """Alias for `self.base_eqn`"""
        return Region4.base_eqn(T)

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
