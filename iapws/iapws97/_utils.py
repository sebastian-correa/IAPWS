import numpy as np
from typing import Optional, Dict
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

R = 0.461526  # kJ/(kg*K)
T_c = 647.096  # K
p_c = 22.064  # MPa
rho_c = 322  # kg/m^3
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

table16_supp_ref2 = {1: {'I': 1, 'J': 8, 'n': -524.581170928788},
                     2: {'I': 1, 'J': 24, 'n': -9269472.18142218},
                     3: {'I': 2, 'J': 4, 'n': -237.385107491666},
                     4: {'I': 2, 'J': 32, 'n': 21077015581.2776},
                     5: {'I': 4, 'J': 1, 'n': -23.9494562010986},
                     6: {'I': 4, 'J': 2, 'n': 221.802480294197},
                     7: {'I': 7, 'J': 7, 'n': -5104725.33393438},
                     8: {'I': 8, 'J': 5, 'n': 1249813.96109147},
                     9: {'I': 8, 'J': 12, 'n': 2000084369.96201},
                     10: {'I': 10, 'J': 1, 'n': -815.158509791035},
                     11: {'I': 12, 'J': 0, 'n': -157.612685637523},
                     12: {'I': 12, 'J': 7, 'n': -11420042233.2791},
                     13: {'I': 18, 'J': 10, 'n': 6623646807768720.0},
                     14: {'I': 20, 'J': 12, 'n': -2.27622818296144e+18},
                     15: {'I': 24, 'J': 32, 'n': -1.71048081348406e+31},
                     16: {'I': 28, 'J': 8, 'n': 6607887669380910.0},
                     17: {'I': 28, 'J': 12, 'n': 1.66320055886021e+22},
                     18: {'I': 28, 'J': 20, 'n': -2.18003784381501e+29},
                     19: {'I': 28, 'J': 22, 'n': -7.87276140295618e+29},
                     20: {'I': 28, 'J': 24, 'n': 1.51062329700346e+31},
                     21: {'I': 32, 'J': 2, 'n': 7957321.70300541},
                     22: {'I': 32, 'J': 7, 'n': 1319576473553470.0},
                     23: {'I': 32, 'J': 12, 'n': -3.2509706829914e+23},
                     24: {'I': 32, 'J': 14, 'n': -4.18600611419248e+25},
                     25: {'I': 32, 'J': 24, 'n': 2.97478906557467e+34},
                     26: {'I': 36, 'J': 10, 'n': -9.53588761745473e+19},
                     27: {'I': 36, 'J': 12, 'n': 1.66957699620939e+24},
                     28: {'I': 36, 'J': 20, 'n': -1.75407764869978e+32},
                     29: {'I': 36, 'J': 22, 'n': 3.47581490626396e+34},
                     30: {'I': 36, 'J': 28, 'n': -7.10971318427851e+38}}

table17_supp_ref2 = {1: {'I': 0, 'J': 0, 'n': 1.04351280732769},
                     2: {'I': 0, 'J': 3, 'n': -2.27807912708513},
                     3: {'I': 0, 'J': 4, 'n': 1.80535256723202},
                     4: {'I': 1, 'J': 0, 'n': 0.420440834792042},
                     5: {'I': 1, 'J': 12, 'n': -105721.24483466},
                     6: {'I': 5, 'J': 36, 'n': 4.36911607493884e+24},
                     7: {'I': 6, 'J': 12, 'n': -328032702839.753},
                     8: {'I': 7, 'J': 16, 'n': -6786867608042700.0},
                     9: {'I': 8, 'J': 2, 'n': 7439.57464645363},
                     10: {'I': 8, 'J': 20, 'n': -3.56896445355761e+19},
                     11: {'I': 12, 'J': 32, 'n': 1.67590585186801e+31},
                     12: {'I': 16, 'J': 36, 'n': -3.55028625419105e+37},
                     13: {'I': 22, 'J': 2, 'n': 396611982166.538},
                     14: {'I': 22, 'J': 32, 'n': -4.14716268484468e+40},
                     15: {'I': 24, 'J': 7, 'n': 3.59080103867382e+18},
                     16: {'I': 36, 'J': 20, 'n': -1.16994334851995e+40}}

table23_supp_ref2 = {1: {'I': 0, 'J': 0, 'n': 0.913965547600543},
                     2: {'I': 1, 'J': -2, 'n': -4.30944856041991e-05},
                     3: {'I': 1, 'J': 2, 'n': 60.3235694765419},
                     4: {'I': 3, 'J': -12, 'n': 1.17518273082168e-18},
                     5: {'I': 5, 'J': -4, 'n': 0.220000904781292},
                     6: {'I': 6, 'J': -3, 'n': -69.0815545851641}}

table25_supp_ref2 = {1: {'I': -12, 'J': 10, 'n': 0.00062909626082981},
                     2: {'I': -10, 'J': 8, 'n': -0.000823453502583165},
                     3: {'I': -8, 'J': 3, 'n': 5.15446951519474e-08},
                     4: {'I': -4, 'J': 4, 'n': -1.17565945784945},
                     5: {'I': -3, 'J': 3, 'n': 3.48519684726192},
                     6: {'I': -2, 'J': -6, 'n': -5.07837382408313e-12},
                     7: {'I': -2, 'J': 2, 'n': -2.84637670005479},
                     8: {'I': -2, 'J': 3, 'n': -2.36092263939673},
                     9: {'I': -2, 'J': 4, 'n': 6.01492324973779},
                     10: {'I': 0, 'J': 0, 'n': 1.48039650824546},
                     11: {'I': 1, 'J': -3, 'n': 0.000360075182221907},
                     12: {'I': 1, 'J': -2, 'n': -0.0126700045009952},
                     13: {'I': 1, 'J': 10, 'n': -1221843.32521413},
                     14: {'I': 3, 'J': -2, 'n': 0.149276502463272},
                     15: {'I': 3, 'J': -1, 'n': 0.698733471798484},
                     16: {'I': 5, 'J': -5, 'n': -0.0252207040114321},
                     17: {'I': 6, 'J': -6, 'n': 0.0147151930985213},
                     18: {'I': 6, 'J': -3, 'n': -1.08618917681849},
                     19: {'I': 8, 'J': -8, 'n': -0.000936875039816322},
                     20: {'I': 8, 'J': -2, 'n': 81.9877897570217},
                     21: {'I': 8, 'J': -1, 'n': -182.041861521835},
                     22: {'I': 12, 'J': -12, 'n': 2.61907376402688e-06},
                     23: {'I': 12, 'J': -1, 'n': -29162.6417025961},
                     24: {'I': 14, 'J': -12, 'n': 1.40660774926165e-05},
                     25: {'I': 14, 'J': 1, 'n': 7832370.62349385}}

table9_supp_ref2 = {1: {'I': 0, 'J': 14, 'n': 0.332171191705237},
                    2: {'I': 0, 'J': 36, 'n': 0.000611217706323496},
                    3: {'I': 1, 'J': 3, 'n': -8.82092478906822},
                    4: {'I': 1, 'J': 16, 'n': -0.45562819254325},
                    5: {'I': 2, 'J': 0, 'n': -2.63483840850452e-05},
                    6: {'I': 2, 'J': 5, 'n': -22.3949661148062},
                    7: {'I': 3, 'J': 4, 'n': -4.28398660164013},
                    8: {'I': 3, 'J': 36, 'n': -0.616679338856916},
                    9: {'I': 4, 'J': 4, 'n': -14.682303110404},
                    10: {'I': 4, 'J': 16, 'n': 284.523138727299},
                    11: {'I': 4, 'J': 24, 'n': -113.398503195444},
                    12: {'I': 5, 'J': 18, 'n': 1156.71380760859},
                    13: {'I': 5, 'J': 24, 'n': 395.551267359325},
                    14: {'I': 7, 'J': 1, 'n': -1.54891257229285},
                    15: {'I': 8, 'J': 4, 'n': 19.4486637751291},
                    16: {'I': 12, 'J': 2, 'n': -3.57915139457043},
                    17: {'I': 12, 'J': 4, 'n': -3.35369414148819},
                    18: {'I': 14, 'J': 1, 'n': -0.66442679633246},
                    19: {'I': 14, 'J': 22, 'n': 32332.1885383934},
                    20: {'I': 16, 'J': 10, 'n': 3317.66744667084},
                    21: {'I': 20, 'J': 12, 'n': -22350.1257931087},
                    22: {'I': 20, 'J': 28, 'n': 5739538.75852936},
                    23: {'I': 22, 'J': 8, 'n': 173.226193407919},
                    24: {'I': 24, 'J': 3, 'n': -0.0363968822121321},
                    25: {'I': 28, 'J': 0, 'n': 8.34596332878346e-07},
                    26: {'I': 32, 'J': 6, 'n': 5.03611916682674},
                    27: {'I': 32, 'J': 8, 'n': 65.5444787064505}}

table10_supp_ref2 = {1: {'I': 0, 'J': 1, 'n': 0.822673364673336},
                     2: {'I': 0, 'J': 4, 'n': 0.181977213534479},
                     3: {'I': 0, 'J': 10, 'n': -0.0112000260313624},
                     4: {'I': 0, 'J': 16, 'n': -0.000746778287048033},
                     5: {'I': 2, 'J': 1, 'n': -0.179046263257381},
                     6: {'I': 3, 'J': 36, 'n': 0.0424220110836657},
                     7: {'I': 4, 'J': 3, 'n': -0.341355823438768},
                     8: {'I': 4, 'J': 16, 'n': -2.09881740853565},
                     9: {'I': 5, 'J': 20, 'n': -8.22477343323596},
                     10: {'I': 5, 'J': 36, 'n': -4.99684082076008},
                     11: {'I': 6, 'J': 4, 'n': 0.191413958471069},
                     12: {'I': 7, 'J': 2, 'n': 0.0581062241093136},
                     13: {'I': 7, 'J': 28, 'n': -1655.05498701029},
                     14: {'I': 7, 'J': 32, 'n': 1588.70443421201},
                     15: {'I': 10, 'J': 14, 'n': -85.0623535172818},
                     16: {'I': 10, 'J': 32, 'n': -31771.4386511207},
                     17: {'I': 10, 'J': 36, 'n': -94589.0406632871},
                     18: {'I': 32, 'J': 0, 'n': -1.3927384708869e-06},
                     19: {'I': 32, 'J': 6, 'n': 0.63105253224098}}


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
        _pi = b23_const[1] + b23_const[2] * T + b23_const[3] * T ** 2
        return _pi
    elif p is not None and T is None:
        theta = b23_const[4] + np.sqrt((p - b23_const[5]) / b23_const[3])
        return theta
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
    z = T + table34[9] / (T - table34[10])
    A = z ** 2 + table34[1] * z + table34[2]
    B = table34[3] * z ** 2 + table34[4] * z + table34[5]
    C = table34[6] * z ** 2 + table34[7] * z + table34[8]
    p = 2 * C / (-B + np.sqrt(B ** 2 - 4 * A * C))
    return p ** 4


def _hp_1(s: float) -> float:
    """Define the saturated line boundary between Region 1 and 4.

    Args
        s: Entropy (kJ/kg/K)

    Returns
        h : Enthalpy (kJ/kg)

    Raises
        `NotImplementedError` if input isn't in limit: [-1.545495919e-4, 3.77828134]

    References:
        [2] Eq 3
    """
    if not -1.545495919e-4 <= s <= 3.77828134:
        raise NotImplementedError(f's should be -1.545495919e-4 <= s <= 3.77828134. {s} provided.')

    sigma = s / 3.8
    return 1700 * sum(entry['n'] * (sigma - 1.09) ** entry['I'] * (sigma + 0.366e-4) ** entry['J'] for entry in
                      table9_supp_ref2.values())

def _hp_3a(s: float) -> float:
    """Define the saturated line boundary between Region 4 and 3a.

    Args
        s: Entropy (kJ/kg/K)

    Returns
        h : Enthalpy (kJ/kg)

    Raises
        `NotImplementedError` if input isn't in limit: [3.778281340, s_c]

    References:
        [2] Eq 4
    """
    if not 3.778281340 <= s <= s_c:
        raise NotImplementedError(f's should be 3.778281340 <= s <= {s_c}. {s} provided.')

    sigma = s/3.8
    return 1700 * sum(entry['n'] * (sigma - 1.09) ** entry['I'] * (sigma + 0.366e-4) ** entry['J'] for entry in
                      table10_supp_ref2.values())


def _hpp_2ab(s: float) -> float:
    """Define the saturated line boundary between Region 4 and 2a and 2b.

    Args
        s: Entropy (kJ/kg/K)

    Returns
        h : Enthalpy (kJ/kg)

    Raises
        `NotImplementedError` if input isn't in limit: [5.85, s''(273.15K)]

    References:
        [2] Eq 5
    """
    if not 5.85 <= s <= 9.155759395:
        raise NotImplementedError(f's should be 5.85 <= s <= 9.155759395. {s} provided.')

    sigma_1 = s / 5.21
    sigma_2 = s / 9.2
    return 2800 * np.exp(sum(
        entry['n'] * (1 / sigma_1 - 0.513) ** entry['I'] * (sigma_2 - 0.524) ** entry['J'] for entry in
        table16_supp_ref2.values()))


def _hpp_2c3b(s: float) -> float:
    """Define the saturated line boundary between Region 4 and 2c-3b.

    Args
        s: Entropy (kJ/kg/K)

    Returns
        h : Enthalpy (kJ/kg)

    Raises
        `NotImplementedError` if input isn't in limit: [s_c, 5.85].

    References
        [2] Eqn 6.
    """
    if not s_c <= s <= 5.85:
        raise NotImplementedError(f's should be {s_c} <= s <= 5.85. {s} provided.')

    sigma = s / 5.9
    return 2800 * sum(entry['n'] * (sigma - 1.02) ** entry['I'] * (sigma - 0.726) ** entry['J'] for entry in
                      table17_supp_ref2.values()) ** 4


def _h_b13(s: float) -> float:
    """Define the boundary between Region 1 and 3.

    Args
        s: Entropy (kJ/kg/K)

    Returns
        h : Enthalpy (kJ/kg)

    Raises
        `NotImplementedError` if input isn't in limit: [3.397782955, 3.77828134].

    References
        [2] Eqn 6.
    """
    if not 3.397782955 <= s <= 3.77828134:
        raise NotImplementedError(f's should be 3.77828134 <= s <= 3.397782955. {s} provided.')

    sigma = s / 3.8

    return 1700 * sum(entry['n'] * (sigma - 0.884) ** entry['I'] * (sigma - 0.864) ** entry['J'] for entry in
                      table23_supp_ref2.values())


def _T_b23(h: float, s: float) -> float:
    """Define the boundary between Region 2 and 3.

    Args
        h : Enthalpy (kJ/kg)
        s: Entropy (kJ/kg/K)

    Returns
        Temperature at the boundary between regions 2 and 3.

    Raises
        `NotImplementedError` if input isn't in limit: 5.048096828 <= s <= 5.260578707 and 2.563592004e3 <= h <= 2.812942061e3

    References
        [2] Eq 8
    """
    if not 5.048096828 <= s <= 5.260578707:
        raise NotImplementedError(f's should be 5.048096828 <= s <= 5.260578707. {s} provided.')
    elif not 2.563592004e3 <= h <= 2.812942061e3:
        raise NotImplementedError(f'h should be 2.563592004e3 <= h <= 2.812942061e3. {h} provided.')

    nu = h / 3000
    sigma = s / 5.3

    return 900 * sum(
        entry['n'] * (nu - 0.727) ** entry['I'] * (sigma - 0.864) ** entry['J'] for entry in table25_supp_ref2.values())


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
    rho: float = None
    u: float = None
    s: float = None
    h: float = None
    cp: float = None
    cv: float = None
    w: float = None
    ders: Dict[str, float] = None
    x: float = None


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


class IAPWS97(object):
    pass
