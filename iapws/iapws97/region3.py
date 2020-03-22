import numpy as np
from typing import Optional, Dict
from collections import defaultdict

from ._utils import State, Region, R, _p_s, rho_c, T_c

class Region3(Region):
    """
    Region3 implements Region3 of the IAPWS97 standard.

    Methods:


    Class attributes:

    """
    b23_const = {1: 0.348_051_856_289_69e3,
                2: -0.116_718_598_799_75e1,
                3: 0.101_929_700_393_26e-2,
                4: 0.572_544_598_627_46e3,
                5: 0.139_188_397_788_70e2}

    b23bc_const = {1: 0.90584278514723e3,
                   2: -0.67955786399241,
                   3: 0.12809002730136e-3,
                   4: 0.26526571908428e4,
                   5: 0.45257578905948e1}

    table30 = {1: {'I': None, 'J': None, 'n': 0.10658070028513e1},
               2: {'I': 0, 'J': 0, 'n': -0.15732845290239e2},
               3: {'I': 0, 'J': 1, 'n': 0.20944396974307e2},
               4: {'I': 0, 'J': 2, 'n': -0.76867707878716e1},
               5: {'I': 0, 'J': 7, 'n': 0.26185947787954e1},
               6: {'I': 0, 'J': 10, 'n': -0.28080781148620e1},
               7: {'I': 0, 'J': 12, 'n': 0.12053369696517e1},
               8: {'I': 0, 'J': 23, 'n': -0.84566812812502e-2},
               9: {'I': 1, 'J': 2, 'n': -0.12654315477714e1},
               10: {'I': 1, 'J': 6, 'n': -0.11524407806681e1},
               11: {'I': 1, 'J': 15, 'n': 0.88521043984318},
               12: {'I': 1, 'J': 17, 'n': -0.64207765181607},
               13: {'I': 2, 'J': 0, 'n': 0.38493460186671},
               14: {'I': 2, 'J': 2, 'n': -0.85214708824206},
               15: {'I': 2, 'J': 6, 'n': 0.48972281541877e1},
               16: {'I': 2, 'J': 7, 'n': -0.30502617256965e1},
               17: {'I': 2, 'J': 22, 'n': 0.39420536879154e-1},
               18: {'I': 2, 'J': 26, 'n': 0.12558408424308},
               19: {'I': 3, 'J': 0, 'n': -0.27999329698710},
               20: {'I': 3, 'J': 2, 'n': 0.13899799569460e1},
               21: {'I': 3, 'J': 4, 'n': -0.20189915023570e1},
               22: {'I': 3, 'J': 16, 'n': -0.82147637173963e-2},
               23: {'I': 3, 'J': 26, 'n': -0.47596035734923},
               24: {'I': 4, 'J': 0, 'n': 0.43984074473500e-1},
               25: {'I': 4, 'J': 2, 'n': -0.44476435428739},
               26: {'I': 4, 'J': 4, 'n': 0.90572070719733},
               27: {'I': 4, 'J': 26, 'n': 0.70522450087967},
               28: {'I': 5, 'J': 1, 'n': 0.10770512626332},
               29: {'I': 5, 'J': 3, 'n': -0.32913623258954},
               30: {'I': 5, 'J': 26, 'n': -0.50871062041158},
               31: {'I': 6, 'J': 0, 'n': -0.22175400873096e-1},
               32: {'I': 6, 'J': 2, 'n': 0.94260751665092e-1},
               33: {'I': 6, 'J': 26, 'n': 0.16436278447961},
               34: {'I': 7, 'J': 2, 'n': -0.13503372241348e-1},
               35: {'I': 8, 'J': 26, 'n': -0.14834345352472e-1},
               36: {'I': 9, 'J': 2, 'n': 0.57922953628084e-3},
               37: {'I': 9, 'J': 26, 'n': 0.32308904703711e-2},
               38: {'I': 10, 'J': 0, 'n': 0.80964802996215e-4},
               39: {'I': 10, 'J': 1, 'n': -0.16557679795037e-3},
               40: {'I': 11, 'J': 26, 'n': -0.44923899061815e-4}}



    def __init__(self, p: Optional[float] = None, T: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None, state: Optional[State] = None):
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
            pass
        elif T and s:
            pass
        elif h and s:
            self._state.p = self.p_hs(h, s)
            self._state.T = self.T_ph(p, h)
            self._state.s = s
            self._state.h = h
        else:
            raise ValueError('You should only pass one of the following combinations to determine a state in Reg1: (p,T) (p, h), (p, s), (T, h), (T,s), (h, s).')
        
        
        if calc:
            tau = 540 / T
            _pi = p / 1

            if not self._state in self:
                # Find region number and return it.
                pass

            ggO, ggR = Region2.base_eqn_id_gas(T=T, p=p), Region2.base_eqn_residual(T=T, p=p)
            gg = ggO + ggR

            gpO, gpR = Region2.base_id_gas_der_pi_const_tau(T=T, p=p), Region2.base_residual_der_pi_const_tau(T=T, p=p)
            gp = gpO + gpR

            gtO, gtR = Region2.base_id_gas_der_tau_const_pi(T=T, p=p), Region2.base_residual_der_tau_const_pi(T=T, p=p)
            gt = gtO + gtR

            gppO, gppR = Region2.base_id_gas_der2_pipi_const_tau(T=T, p=p), Region2.base_residual_der2_pipi_const_tau(T=T, p=p)
            gpp = gppO + gppR

            gttO, gttR = Region2.base_id_gas_der2_tautau_const_pi(T=T, p=p), Region2.base_residual_der2_tautau_const_pi(T=T, p=p)
            gtt = gttO + gttR

            gptO, gptR = Region2.base_id_gas_der2_pitau(T=T, p=p), Region2.base_residual_der2_pitau(T=T, p=p)
            gpt = gptO + gptR
            
            self._state.ders = defaultdict(float,
                                           gammaO=ggO, gammaR=ggR, gamma=gg,
                                           gammaO_pi=gpO, gammaR_pi=gpO, gamma_pi=gp,
                                           gammaO_tau=gtO, gammaR_tau=gtR, gamma_tau=gt,
                                           gammaO_pipi=gppO, gammaR_pipi=gppR, gamma_pipi=gpp,
                                           gammaO_tautau=gttO, gammaR_tautau=gttR, gamma_tautau=gtt,
                                           gammaO_pitau=gptO, gammaR_pitau=gptR, gamma_pitau=gpt)

            self._state.v = _pi * gp * R * T / p / 1000  # R*T/p has units of 1000 m^3/kg.
            self._state.u = R * T * (tau*gt - _pi*gp)
            self._state.s = self._state.s if self._state.s is not None else R * (tau*gt - gg)
            self._state.h = self._state.h if self._state.h is not None else R * T * tau * gt
            self._state.cp = R * -tau**2 * gtt
            self._state.cv = R * (-tau**2 * gtt - (1 + _pi * gpR - tau * _pi * gptR)**2 / (1 - _pi**2 * gppR))
            self._state.w = np.sqrt(1000 * R * T * gp**2 / ((gp-tau*gpt)**2 / (tau**2 * gtt) - gpp))  # 1000 is a conversion factor: sqrt(kJ/kg) = sqrt(1000 m/s) -> sqrt(1000) m/s
        else:
            self._state = State()

    @staticmethod
    def p_b23(T: float) -> float:
        """
        Implements the equation for the boundary between 2 and 3 as p = p(T)
        Args:
            T: Temperature (K).
        Returns:
            The value p at the boundary between 2 and 3.
        Raises:
            ValueError if T is not in [623.15, 863.15]
        """
        if not 623.15 <= T <= 863.15:
            raise ValueError(f'T must be in the range [623.15, 863.15]. {T}K supplied.')
        return  Region2.b23_const[1] + Region2.b23_const[2] * T + Region2.b23_const[3] * T**2
    
    @staticmethod
    def T_b23(p: float) -> float:
        """
        Implements the equation for the boundary between 2 and 3 as T = T(p)
        Args:
            p: Pressure (MPa).
        Returns:
            The value T at the boundary between 2 and 3.
        Raises:
            ValueError if p is not in [16.5292, 100]
        """
        if not 16.5292 <= p <= 100:
            raise ValueError(f'T must be in the range [623.15, 863.15]. {T}K supplied.')
        return  Region2.b23_const[4] + np.sqrt( (p - Region2.b23_const[5]) / Region2.b23_const[3] )

    @staticmethod
    def b2bc(p:Optional[float] = None, h: Optional[float] = None) -> str:
        """
        Implements the equation for the boundaries in subregions of region 2.
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
        Returns:
            The value of p if h is given or the value of h if p is given.
        Raises:
            ValueError if both p and h are supplied.
        """
        # TODO: Page 21 limits?
        if h is not None and p is None:
            _pi = Region2.b23bc_const[1] + Region2.b23bc_const[2] * h + Region2.b23bc_const[3] * h**2
            return  _pi
        elif p is not None and h is None:
            eta = Region2.b23bc_const[4] + np.sqrt( (p - Region2.b23bc_const[5]) / Region2.b23bc_const[3] )
            return  eta
        else:
            raise ValueError('Pass only T or P, not both.')
    
    @staticmethod
    def h_2ab(s: float) -> float:
        """
        Used to determine wheter to use region 2a or 2b. Eq.2 from supplementary release.
        Args:
            s: Entryopy (kJ/kg/K)
        Returns:
            The value of the enthalpy in region 2 for a given entropy.
        """
        return Region2.table5_supp[1] + Region2.table5_supp[2] * s + Region2.table5_supp[3] * s**2 + Region2.table5_supp[4] * s**3

    @staticmethod
    def subregion(p: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None) -> str:
        """
        Returns 'a', 'b' or 'c' depending on the subregion in region 2 given a (p, h), (p, s) or (h, s) pair.
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
            s: Entropy (kJ/kg/K)
        Returns:
            The subregion in region 2.
        Raises:
            ValueError if an erroneous pair is provided.
        """
        #TODO: Probably check if value is in fact in region2?
        if h is not None and s is not None and p is None:
            if s < 5.85:
                return 'c'
            else:
                h_calc = Region2.h_2ab(s)
                if h <= h_calc:
                    return 'a'
                else:
                    return 'b'
        elif h is not None and p is not None and s is None:
            if p <= 4:
                return 'a'
            else:
                p_calc = Region2.b2bc(h=h)
                if p <= p_calc:
                    return 'b'
                else:
                    return 'c'
        elif s is not None and p is not None and h is None:
            if p <= 4:
                return 'a'
            else:
                if s >= 5.85:
                    return 'b'
                else:
                    return 'c'
        else:
            raise ValueError('Please supply only one of the following data pairs: (p, h), (p, s) or (h, s).')

    def __contains__(self, other: State) -> bool:
        """
        Overrides the behaviour of the `in` operator to facilitate a `State in Region` query.
        """
        if not isinstance(other, State):
            return False
        else:
            # Lower bounds for p from page 5 of Revised Supplementary Release on Backward Equations for Pressure as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the IAPWS...
            cond1 = 273.15 <= other.T <= 623.15 and 611.213e-6 <= other.p <= _p_s(T=other.T)
            cond2 = 623.15 <= other.T <= 863.15 and 611.213e-6 <= other.p <= Region2.p_b23(T=other.T)
            cond3 = 863.15 <= other.T <= 1073.15 and 611.213e-6 <= other.p <= 100
            return cond1 or cond2 or cond3

    def __repr__(self) -> str:
        return f'Region3(p={self.p}, T={self.T})'

    @staticmethod
    def base_eqn(T: float, rho: float) -> float:
        """
        Dimensionless specific Helmholtz free energy (eq. 28).
        Args:
            T: Temperature (K)
            rho: rho (kg/m^3)
        Returns:
            Dimensionless specific Helmholtz free energy.
        """
        delta = rho / rho_c
        tau = T / T_c
        _sum = sum(entry['n'] * delta**entry['I'] * tau**entry['J'] for entry in Region3.table30.values() if entry['I'] is not None)
        other_term = Region3.table30[1]['n'] * np.log(delta)
        return R * T * (other_term + _sum)

    @staticmethod
    def specific_helmholtz_free_energy(T: float, rho: float) -> float:
        """Alias for `self.base_eqn`"""
        return Region3.base_eqn(T, p) * R * T

    #############################################################
    ################## FIRST ORDER DERIVATIVES ##################
    #############################################################
    @staticmethod
    def base_id_gas_der_pi_const_tau(T: float, p: float) -> float:
        """Derivative of Ideal gas part of dimensionless specific Gibbs free energy (`gammaO`) with respect to `pi` with consant `tau`
        Also known as gammaO_pi.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of Ideal gas part of dimensionless specific Gibbs free energy (`gammaO`) with respect to `pi` with consant `tau`
        """
        return 1/p

    @staticmethod
    def base_residual_der_pi_const_tau(T: float, p: float) -> float:
        """Derivative of residual of dimensionless specific Gibbs free energy (`gammaR`) with respect to `pi` with consant `tau`
        Also known as gammaR_pi.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of residual part of dimensionless specific Gibbs free energy (`gammaR`) with respect to `pi` with consant `tau`
        """
        tau = 540 / T
        return sum(entry['n'] * entry['I'] * p**(entry['I'] - 1) * (tau - 0.5)**entry['J'] for entry in Region2.table11.values())

    @staticmethod
    def base_id_gas_der_tau_const_pi(T: float, p: float) -> float:
        """Derivative of Ideal gas part of dimensionless specific Gibbs free energy (`gammaO`) with respect to `tau` with consant `pi`
        Also known as gammaO_tau.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of Ideal gas part of dimensionless specific Gibbs free energy (`gammaO`) with respect to `tau` with consant `pi`
        """
        tau = 540 / T
        return sum(entry['n'] * entry['J'] * tau**(entry['J'] - 1) for entry in Region2.table10.values())
    
    @staticmethod
    def base_residual_der_tau_const_pi(T: float, p: float) -> float:
        """Derivative of residual part of dimensionless specific Gibbs free energy (`gammaR`) with respect to `tau` with consant `pi`
        Also known as gammaR_tau.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of residual part of dimensionless specific Gibbs free energy (`gammaR`) with respect to `tau` with consant `pi`
        """
        tau = 540 / T
        return sum(entry['n'] * p**entry['I'] * entry['J'] * (tau - 0.5)**(entry['J'] - 1) for entry in Region2.table11.values())

    #############################################################
    ################# SECOND ORDER DERIVATIVES ##################
    #############################################################
    @staticmethod
    def base_id_gas_der2_pipi_const_tau(T: float, p: float) -> float:
        """Second order derivative of Ideal gas part of Dimensionless specific Gibbs free energy (`gammaO`) with respect to `pi` with consant `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Ideal gas part of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        """
        return -1/p**2
    
    @staticmethod
    def base_id_gas_der2_tautau_const_pi(T: float, p: float) -> float:
        """Second order derivative of Ideal gas part of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Ideal gas part of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        """
        tau = 540 / T
        return sum(entry['n'] * entry['J'] * (entry['J'] -1) * tau**(entry['J'] - 2) for entry in Region2.table10.values())
    
    @staticmethod
    def base_id_gas_der2_pitau(T: float, p: float) -> float:
        """Second order derivative of Ideal gas part of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Ideal gas part of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`
        """
        return 0

    @staticmethod
    def base_residual_der2_pipi_const_tau(T: float, p: float) -> float:
        """Second order derivative of residual of Dimensionless specific Gibbs free energy (`gammaR`) with respect to `pi` with consant `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of residual of Dimensionless specific Gibbs free energy (`gammaR`) with respect to `pi` with consant `tau`
        """
        tau = 540 / T
        return sum(entry['n'] * entry['I'] * (entry['I'] - 1) * p **(entry['I'] - 2) * (tau - 0.5)**entry['J'] for entry in Region2.table11.values())
    
    @staticmethod
    def base_residual_der2_tautau_const_pi(T: float, p: float) -> float:
        """Second order derivative of residual of Dimensionless specific Gibbs free energy (`gammaR`) with respect to `tau` with consant `pi`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of residual of Dimensionless specific Gibbs free energy (`gammaR`) with respect to `tau` with consant `pi`
        """
        tau = 540 / T
        return sum(entry['n'] * p**entry['I'] * entry['J'] * (entry['J'] - 1) * (tau - 0.5)**(entry['J'] - 2) for entry in Region2.table11.values())
    
    @staticmethod
    def base_residual_der2_pitau(T: float, p: float) -> float:
        """Second order derivative of residual of Dimensionless specific Gibbs free energy (`gammaR`) with respect to `pi` and then `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of residual of Dimensionless specific Gibbs free energy (`gammaR`) with respect to `pi` and then `tau`
        """
        tau = 540 / T
        return sum(entry['n'] * entry['I'] * p**(entry['I'] - 1) * entry['J'] * (tau - 0.5)**(entry['J'] - 1) for entry in Region2.table11.values())

    #############################################################
    ####################### Properties ##########################
    #############################################################
    @property
    def gamma(self) -> float:
        """Dimensionless specific Gibbs free energy (eq. 15)."""
        return self._state.ders['gamma']

    @property
    def gammaO(self) -> float:
        """Dimensionless specific Gibbs free energy ideal gas(eq. 16)."""
        return self._state.ders['gammaO']
    
    @property
    def gammaR(self) -> float:
        """Dimensionless specific Gibbs free energy residual (eq. 17)."""
        return self._state.ders['gammaO']

    @property
    def gamma_pi(self) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`"""
        return self._state.ders['gamma_pi']
    
    @property
    def gammaO_pi(self) -> float:
        """Ideal gas part of the derivative of gamma with respect to pi."""
        return self._state.ders['gammaO_pi']

    @property
    def gammaR_pi(self) -> float:
        """Residual part of the derivative of gamma with respect to pi."""
        return self._state.ders['gammaR_pi']

    @property
    def gamma_tau(self) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`"""
        return self._state.ders['gamma_tau']

    @property
    def gammaO_tau(self) -> float:
        """Ideal gas part of the derivative of gamma with respect to tau."""
        return self._state.ders['gammaO_tau']

    @property
    def gammaR_tau(self) -> float:
        """Residual part of the derivative of gamma with respect to tau."""
        return self._state.ders['gammaR_tau']

    @property
    def gamma_pipi(self) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`"""
        return self.self._state.ders['gamma_pipi']

    @property
    def gammaO_pipi(self) -> float:
        """Ideal gas part of the second order derivative of gamma with respect to pi twice."""
        return self.self._state.ders['gammaO_pipi']

    @property
    def gammaR_pipi(self) -> float:
        """Residual part of the second order derivative of gamma with respect to pi twice."""
        return self.self._state.ders['gammaR_pipi']

    @property
    def gamma_tautau(self) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`"""
        return self.self._state.ders['gamma_tautau']
    
    @property
    def gammaO_tautau(self) -> float:
        """Idel gas part of the second order derivative of gamma with respect to pi twice."""
        return self.self._state.ders['gammaO_tautau']
    @property
    def gammaR_tautau(self) -> float:
        """Residual part of the second order derivative of gamma with respect to tau twice."""
        return self.self._state.ders['gammaR_tautau']

    @property
    def gamma_pitau(self) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`"""
        return self.self._state.ders['gamma_pitau']

    @property
    def gammaO_pitau(self) -> float:
        """Ideal gas part of the second order derivative of gamma with respect to pi and then tau."""
        return self.self._state.ders['gammaO_pitau']

    @property
    def gammaR_pitau(self) -> float:
        """Residual part of the second order derivative of gamma with respect to pi and then tau."""
        return self.self._state.ders['gammaR_pitau']

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
        return 1 / self._state.v
    
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
        Backwards equations 22, 23 and 23 for calculating Temperature as a function of pressure and enthalpy.
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
        Returns:
            Temperature (K).
        """
        eta = h / 2000
        reg = self.subregion(p=p, h=h)
        if reg == 'a':
            T = sum(entry['n'] * p**entry['I'] * (eta - 2.1)**entry['J'] for entry in Region2.table20.values())
        elif reg == 'b':
            T = sum(entry['n'] * (p - 2)**entry['I'] * (eta - 2.6)**entry['J'] for entry in Region2.table21.values())
        elif reg == 'c':
            T = sum(entry['n'] * (p + 25)**entry['I'] * (eta - 1.8)**entry['J'] for entry in Region2.table22.values())

        if State(p=p, T=T) in self:
            return T
        else:
            raise ValueError(f'State out of bounds. {T}')

    def T_ps(self, p: float, s: float) -> float:
        """
        Backwards equations 25, 26 and 27 for calculating Temperature as a function of pressure and entropy.
        Args:
            p: Pressure (MPa).
            s: Entropy (kJ/kg/K).
        Returns:
            Temperature (K).
        """
        reg = self.subregion(p=p, s=s)
        if reg == 'a':
            sigma = s / 2
            T = sum(entry['n'] * p**entry['I'] * (sigma - 2)**entry['J'] for entry in Region2.table25.values())
        elif reg == 'b':
            sigma = s / 0.7853
            T = sum(entry['n'] * p**entry['I'] * (10 - sigma)**entry['J'] for entry in Region2.table26.values())
        elif reg == 'c':
            sigma = s / 2.9251
            T = sum(entry['n'] * p**entry['I'] * (2 - sigma)**entry['J'] for entry in Region2.table27.values())

        if State(p=p, T=T) in self:
            return T
        else:
            raise ValueError(f'State out of bounds. {p},{T}')

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
        reg = self.subregion(h=h, s=s)
        if reg == 'a':
            sigma = s / 12
            eta = h / 4200
            _pi = sum(entry['n'] * (eta - 0.5)**entry['I'] * (sigma - 1.2)**entry['J'] for entry in Region2.table6_supp.values()) **4
            p = 4 * _pi
        elif reg == 'b':
            sigma = s / 7.9
            eta = h / 4100
            _pi = sum(entry['n'] * (eta - 0.6)**entry['I'] * (sigma - 1.01)**entry['J'] for entry in Region2.table7_supp.values()) **4
            p = 100 * _pi
        elif reg == 'c':
            sigma = s / 5.9
            eta = h / 3500
            _pi = sum(entry['n'] * (eta - 0.7)**entry['I'] * (sigma - 1.1)**entry['J'] for entry in Region2.table8_supp.values()) **4
            p = 100 * _pi

        T = self.T_ph(p=p, h=h)
        if State(p=p, T=T) in self:
            return p
        else:
            raise ValueError(f'State out of bounds. {p},{T}')
