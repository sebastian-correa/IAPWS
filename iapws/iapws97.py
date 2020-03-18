import numpy as np
from typing import Optional, NamedTuple
from abc import ABC

R = 0.461526        # kJ/(kg*K)
T_c = 647.096       # K
p_c = 22.064        # MPa
rho_c = 322         # kg/m^3

b23_const = {1: 0.348_051_856_289_69e3,
             2: -0.116_718_598_799_75e1,
             3: 0.101_929_700_393_26e-2,
             4: 0.572_544_598_627_46e3,
             5: 0.139_188_397_788_70e2}

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

def b23(p: Optional[float] = None, T: Optional[float] = None) -> float:
    """
    Implements the equation for the boundary between 2 and 3.
    Args:
        p: Pressure (MPa).
        T: Temperature (K).
    Returns:
        The value of T if p is given or the value of p is T is given.
    Raises:
        ValueError if both p and T are supplied.
    """
    if T is not None and p is None:
        _pi = b23_const[1] + b23_const[2] * T + b23_const[3] * T**2
        return  _pi
    elif p is not None and T is None:
        theta = b23_const[4] + np.sqrt( (p - b23_const[5]) / b23_const[3] )
        return  theta
    else:
        raise ValueError('Pass only T or P, not both.')

def _p_s(T: float) -> float:
    """
    Implements equation 30 for the boundary between regions 4 and 1,2.
    Args:
        T: Temperature in K.
    Returns:
        The saturation pressure at the given temperature in MPa.
    Raises:
        ValueError if T is out of bounds (bounds: [273.15, 647.096])
    """
    if not 273.15 <= T <= 647.096:
        raise ValueError(f'T must be in the range [273.15, 647.096]. {T} given.')
    z = T + table34[9]/(T-table34[10])
    A = z**2 + table34[1] * z + table34[2]
    B = table34[3] * z**2 + table34[4] * z + table34[5]
    C = table34[6] * z**2 + table34[7] * z + table34[8]
    p = 2*C / (-B + np.sqrt(B**2 - 4*A*C))
    return p**4

def region(p: float, T: float) -> int:
    """
    Implements the equation for the boundary between 2 and 3.
    Args:
        p: Pressure (MPa).
        T: Temperature (K).
    Returns:
        The region number.
    Raises:
        ValueError is p and T combination are out of bounds.
    """
    if 1073.15 <= T <= 2273.15 and 0 <= p <= 50:
        return 5
    elif 273.15 <= T <= 1073.15 and 0 <= p <= 100:
        T_b23 = b23(p=p)
        p_b23 = b23(T=T)

        if T >= T_b23 and p <= p_b23:
            return 2
        elif 623.15 <= T <= T_b23 and p > p_b23:
            return 3
        elif 273.15 <= T <= 623.15 and p > p_b23:
            return 1
        else:
            raise ValueError('Parameters out of bounds.')
    else:
        raise ValueError('Parameters out of bounds.')

class State(NamedTuple):
    T: float = None
    p: float = None
    v: float = None
    u: float = None
    h: float = None
    cp: float = None
    cv: float = None
    w: float = None

class Region(ABC):
    """
    Region Abstract Base Class detailing how a region should be implemented.

    Available methods vary by region due to the different implementations in the standard. 
    
    However, each region has a base equation from which all properties are derived, access to the necessary first and second order partial derivatives and all backwards equations.
    Further, a region must override the `__contains__` method to facilitate a `State in Region` type of query.

    A Region can also have many constants at a class level if those constants are used only inside said region. Otherwise, a module level constant is favoured.
    """
    def __contains__(self, other: State) -> bool:
        """
        Overrides the behaviour of the `in` operator to facilitate a `State in Region` query.
        """
    
    def base_eqn(self):
        """
        Implements the base equation.
        """

class Region1(Region):
    """
    Region1 implements Region1 of the IAPWS97 standard.

    Methods:
        __contains__
        base_eqn

    Class attributes:

    Instance attributes:

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

    def __init__(self, state: State = None):
        if not state in self:
            # Find region number and return it.
            pass
        else:
            self._state = state

    def __contains__(self, other: State) -> bool:
        """
        Overrides the behaviour of the `in` operator to facilitate a `State in Region` query.
        """
        if other is None:
            return False
        else:
            return 273.15 <= other.T <= 623.15 and _p_s(T=other.T) <= other.p <= 100
    
    def base_eqn(self, T: float, p: float) -> float:
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
        return sum(entry['n'] * (7.1 - _pi)**entry['I'] * (tau - 1.222)**entry['J'] for entry in Region1.table2.items())

    def specific_gibbs_free_energy(self, T: float, p: float) -> float:
        """
        Specific Gibbs free energy (eq. 7).
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Specific Gibbs free energy.
        """
        return self.base_eqn(T, p) * R * T
    
    def gamma(self, T: float, p: float) -> float:
        """Alias for `self.base_eqn`"""
        return self.base_eqn(T, p)
    
    #############################################################
    ################## FIRST ORDER DERIVATIVES ##################
    #############################################################
    def base_der_pi_const_tau(self, T: float, p: float) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(- entry['n'] * entry['I'] * (7.1 - _pi)**(entry['I'] - 1) * (tau - 1.222)**entry['J'] for entry in Region1.table2.items())
    
    def gamma_pi(self, T: float, p: float) -> float:
        """Alias for `base_der_pi_const_tau`."""
        return self.base_der_pi_const_tau(T, p)

    def base_der_tau_const_pi(self, T: float, p: float) -> float:
        """Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(- entry['n'] * (7.1 - _pi)**(entry['I'] - 1) * entry['J'] * (tau - 1.222)**(entry['J'] - 1) for entry in Region1.table2.items())
    
    def gamma_tau(self, T: float, p: float) -> float:
        """Alias for `base_der_tau_const_pi`."""
        return self.base_der_tau_const_pi(T, p)

    #############################################################
    ################# SECOND ORDER DERIVATIVES ##################
    #############################################################
    def base_der2_pipi_const_tau(self, T: float, p: float) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` with consant `tau`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(entry['n'] * entry['I'] * (entry['I'] - 1) * (7.1 - _pi)**(entry['I'] - 2) * (tau - 1.222)**entry['J'] for entry in Region1.table2.items())
    
    def gamma_pipi(self, T: float, p: float) -> float:
        """Alias for `base_der2_pipi_const_tau`."""
        return self.base_der2_pipi_const_tau(T, p)

    def base_der_tautau_const_pi(self, T: float, p: float) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `tau` with consant `pi`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(entry['n'] * (7.1 - _pi)**entry['I'] * entry['J'] * (entry['J'] -1) * (tau - 1.222)**(entry['J'] - 2) for entry in Region1.table2.items())
    
    def gamma_tautau(self, T: float, p: float) -> float:
        """Alias for `base_der_tautau_const_pi`."""
        return self.base_der_tautau_const_pi(T, p)
    
    def base_der_pitau(self, T: float, p: float) -> float:
        """Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Second order derivative of Dimensionless specific Gibbs free energy (`gamma`) with respect to `pi` and then `tau`
        """
        tau = 1386 / T
        _pi = p / 16.53
        return sum(- entry['n'] * entry['I'] * (7.1 - _pi)**(entry['I'] - 1) * entry['J'] * (tau - 1.222)**(entry['J'] - 1) for entry in Region1.table2.items())
    
    def gamma_pitau(self, T: float, p: float) -> float:
        """Alias for `base_der_pitau`."""
        return self.base_der_pitau(T, p)

    #############################################################
    ####################### Properties ##########################
    #############################################################
    @property
    def v(self) -> float:
        return self._state.h
    
    @property
    def h(self) -> float:
        return self._state.h

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
        if 273.15 <= T <= 623.15 and _p_s(T) <= p <= 100:
            return T
        else:
            raise ValueError(f'State out of bounds. {T}')
