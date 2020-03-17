import numpy as np

R = 0.461526        # kJ/(kg*K)
T_c = 647.096       # K
p_c = 22.064        # MPa
rho_c = 322         # kg/m^3

p_star = 1          # MPa
T_star = 1          # K

b23_const = {1: 0.348_051_856_289_69 * 10**3,
             2: -0.116_718_598_799_75 * 10,
             3: 0.101_929_700_393_26 * 10**-2,
             4: 0.572_544_598_627_46 * 10**3,
             5: 0.139_188_397_788_70 * 10**2}

def b23(T: float = None, p: float = None) -> float:
    if T is not None and p is None:
        theta = T/T_star
        pi = b23_const[1] + b23_const[2] * theta + b23_const[3] * theta**2
        return  p_star * pi
    elif p is not None and T is None:
        pi = p/p_star
        theta = b23_const[4] + np.sqrt( (pi - b23_const[5]) / b23_const[3] )
        return  T_star * theta
    else:
        raise ValueError('Pass only T or P, not both.')

