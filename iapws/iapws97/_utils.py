import numpy as np
from typing import Optional, Dict
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

R = 0.461526        # kJ/(kg*K)
T_c = 647.096       # K
p_c = 22.064        # MPa
rho_c = 322         # kg/m^3
s_c = 4.41202148223476  # kJ/kg/K

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
    pass

@dataclass
class State(object):
    T: float = None
    p: float = None
    v: float = None
    u: float = None
    s: float = None
    h: float = None
    cp: float = None
    cv: float = None
    w: float = None
    ders: Dict[str, float] = None

class Region(ABC):
    """
    Region Abstract Base Class detailing how a region should be implemented.

    Available methods vary by region due to the different implementations in the standard. 
    
    However, each region has a base equation from which all properties are derived, access to the necessary first and second order partial derivatives and all backwards equations.
    Further, a region must override the `__contains__` method to facilitate a `State in Region` type of query.

    Partial derivatives must be named `base_der_VAR_const_CONST` or `base_der2_VAR1VAR2_const_CONST`.

    It is recommended that a Region calculates the properties from the data given to its constructor and stores the restuts in a `self._state` attribute. 
    
    The properties must be accessible via @property decorated functions that access the `self._state` attribute.

    The constructor must be able to calculate the basic properties and populate the `self._state` attribute with the results and the derivatives. An instance must have 2 ways of instantiating: either giving a State or a combination of keyword argument properties that define the state.
    It must also have the possibility to be instantiated empty (with None in all keyword arguments) so that the `State in Region1()` is easy.

    A Region can also have many constants at a class level if those constants are used only inside said region. Otherwise, a module level constant is favoured.
    """
    @abstractmethod
    def __contains__(self, other: State) -> bool:
        """
        Overrides the behaviour of the `in` operator to facilitate a `State in Region` query.
        """
    
    @staticmethod
    @abstractmethod
    def base_eqn(T: float, p: float) -> float:
        """
        Implements the base equation.
        """
    
    @abstractmethod
    def __repr__(self) -> str:
        return ''
