import numpy as np
from typing import Optional, Dict
from collections import defaultdict

from ._utils import State, Region, R, _p_s, rho_c, T_c, s_c

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

    table1_supp = {1: 0.201_464_004_206_875e4, 2: 0.374_696_550_136_983e1, 3: -0.219_921_901_054_187e-1, 4: 0.875_131_686_009_950e-4}

    table3_supp = {1: {'I': -12, 'J': 0, 'n': -0.133645667811215e-6},
                   2: {'I': -12, 'J': 1, 'n': 0.455912656802978e-5},
                   3: {'I': -12, 'J': 2, 'n': -0.146294640700979e-4},
                   4: {'I': -12, 'J': 6, 'n': 0.639341312970080e-2},
                   5: {'I': -12, 'J': 14, 'n': 0.372783927268847e3},
                   6: {'I': -12, 'J': 16, 'n': -0.718654377460447e4},
                   7: {'I': -12, 'J': 20, 'n': 0.573494752103400e6},
                   8: {'I': -12, 'J': 22, 'n': -0.267569329111439e7},
                   9: {'I': -10, 'J': 1, 'n': -0.334066283302614e-4},
                   10: {'I': -10, 'J': 5, 'n': -0.245479214069597e-1},
                   11: {'I': -10, 'J': 12, 'n': 0.478087847764996e2},
                   12: {'I': -8, 'J': 0, 'n': 0.764664131818904e-5},
                   13: {'I': -8, 'J': 2, 'n': 0.128350627676972e-2},
                   14: {'I': -8, 'J': 4, 'n': 0.171219081377331e-1},
                   15: {'I': -8, 'J': 10, 'n': -0.851007304583213e1},
                   16: {'I': -5, 'J': 2, 'n': -0.136513461629781e-1},
                   17: {'I': -3, 'J': 0, 'n': -0.384460997596657e-5},
                   18: {'I': -2, 'J': 1, 'n': 0.337423807911655e-2},
                   19: {'I': -2, 'J': 3, 'n': -0.551624873066791},
                   20: {'I': -2, 'J': 4, 'n': 0.729202277107470},
                   21: {'I': -1, 'J': 0, 'n': -0.992522757376041e-2},
                   22: {'I': -1, 'J': 2, 'n': -0.119308831407288},
                   23: {'I': 0, 'J': 0, 'n': 0.793929190615421},
                   24: {'I': 0, 'J': 1, 'n': 0.454270731799386},
                   25: {'I': 1, 'J': 1, 'n': 0.209998591259910},
                   26: {'I': 3, 'J': 0, 'n': -0.642109823904738e-2},
                   27: {'I': 3, 'J': 1, 'n': -0.235155868604540e-1},
                   28: {'I': 4, 'J': 0, 'n': 0.252233108341612e-2},
                   29: {'I': 4, 'J': 3, 'n': -0.764885133368119e-2},
                   30: {'I': 10, 'J': 4, 'n': 0.136176427574291e-1},
                   31: {'I': 12, 'J': 5, 'n': -0.133027883575669e-1}}
 
    table4_supp = {1: {'I': -12, 'J': 0, 'n': 0.323254573644920e-4},
                   2: {'I': -12, 'J': 1, 'n': -0.127575556587181e-3},
                   3: {'I': -10, 'J': 0, 'n': -0.475851877356068e-3},
                   4: {'I': -10, 'J': 1, 'n': 0.156183014181602e-2},
                   5: {'I': -10, 'J': 5, 'n': 0.105724860113781},
                   6: {'I': -10, 'J': 10, 'n': -0.858514221132534e2},
                   7: {'I': -10, 'J': 12, 'n': 0.724140095480911e3},
                   8: {'I': -8, 'J': 0, 'n': 0.296475810273257e-2},
                   9: {'I': -8, 'J': 1, 'n': -0.592721983365988e-2},
                   10: {'I': -8, 'J': 2, 'n': -0.126305422818666e-1},
                   11: {'I': -8, 'J': 4, 'n': -0.115716196364853},
                   12: {'I': -8, 'J': 10, 'n': 0.849000969739595e2},
                   13: {'I': -6, 'J': 0, 'n': -0.108602260086615e-1},
                   14: {'I': -6, 'J': 1, 'n': 0.154304475328851e-1},
                   15: {'I': -6, 'J': 2, 'n': 0.750455441524466e-1},
                   16: {'I': -4, 'J': 0, 'n': 0.252520973612982e-1},
                   17: {'I': -4, 'J': 1, 'n': -0.602507901232996e-1},
                   18: {'I': -3, 'J': 5, 'n': -0.307622221350501e1},
                   19: {'I': -2, 'J': 0, 'n': -0.574011959864879e-1},
                   20: {'I': -2, 'J': 4, 'n': 0.503471360939849e1},
                   21: {'I': -1, 'J': 2, 'n': -0.925081888584834},
                   22: {'I': -1, 'J': 4, 'n': 0.391733882917546e1},
                   23: {'I': -1, 'J': 6, 'n': -0.773146007130190e2},
                   24: {'I': -1, 'J': 10, 'n': 0.949308762098587e4},
                   25: {'I': -1, 'J': 14, 'n': -0.141043719679409e7},
                   26: {'I': -1, 'J': 16, 'n': 0.849166230819026e7},
                   27: {'I': 0, 'J': 0, 'n': 0.861095729446704},
                   28: {'I': 0, 'J': 2, 'n': 0.323346442811720},
                   29: {'I': 1, 'J': 1, 'n': 0.873281936020439},
                   30: {'I': 3, 'J': 1, 'n': -0.436653048526683},
                   31: {'I': 5, 'J': 1, 'n': 0.286596714529479},
                   32: {'I': 6, 'J': 1, 'n': -0.131778331276228},
                   33: {'I': 8, 'J': 1, 'n': 0.676682064330275e-2}}

    table6_supp = {1: {'I': -12, 'J': 6, 'n': 0.529944062966028e-2},
                   2: {'I': -12, 'J': 8, 'n': -0.170099690234461},
                   3: {'I': -12, 'J': 12, 'n': 0.111323814312927e2},
                   4: {'I': -12, 'J': 18, 'n': -0.217898123145125e4},
                   5: {'I': -10, 'J': 4, 'n': -0.506061827980875e-3},
                   6: {'I': -10, 'J': 7, 'n': 0.556495239685324},
                   7: {'I': -10, 'J': 10, 'n': -0.943672726094016e1},
                   8: {'I': -8, 'J': 5, 'n': -0.297856807561527},
                   9: {'I': -8, 'J': 12, 'n': 0.939353943717186e2},
                   10: {'I': -6, 'J': 3, 'n': 0.192944939465981e-1},
                   11: {'I': -6, 'J': 4, 'n': 0.421740664704763},
                   12: {'I': -6, 'J': 22, 'n': -0.368914126282330e7},
                   13: {'I': -4, 'J': 2, 'n': -0.737566847600639e-2},
                   14: {'I': -4, 'J': 3, 'n': -0.354753242424366},
                   15: {'I': -3, 'J': 7, 'n': -0.199768169338727e1},
                   16: {'I': -2, 'J': 3, 'n': 0.115456297059049e1},
                   17: {'I': -2, 'J': 16, 'n': 0.568366875815960e4},
                   18: {'I': -1, 'J': 0, 'n': 0.808169540124668e-2},
                   19: {'I': -1, 'J': 1, 'n': 0.172416341519307},
                   20: {'I': -1, 'J': 2, 'n': 0.104270175292927e1},
                   21: {'I': -1, 'J': 3, 'n': -0.297691372792847},
                   22: {'I': 0, 'J': 0, 'n': 0.560394465163593},
                   23: {'I': 0, 'J': 1, 'n': 0.275234661176914},
                   24: {'I': 1, 'J': 0, 'n': -0.148347894866012},
                   25: {'I': 1, 'J': 1, 'n': -0.651142513478515e-1},
                   26: {'I': 1, 'J': 2, 'n': -0.292468715386302e1},
                   27: {'I': 2, 'J': 0, 'n': 0.664876096952665e-1},
                   28: {'I': 2, 'J': 2, 'n': 0.352335014263844e1},
                   29: {'I': 3, 'J': 0, 'n': -0.146340792313332e-1},
                   30: {'I': 4, 'J': 2, 'n': -0.224503486668184e1},
                   31: {'I': 5, 'J': 2, 'n': 0.110533464706142e1},
                   32: {'I': 8, 'J': 2, 'n': -0.408757344495612e-1}}

    table7_supp = {1: {'I': -12, 'J': 0, 'n': -0.225196934336318e-8},
                   2: {'I': -12, 'J': 1, 'n': 0.140674363313486e-7},
                   3: {'I': -8, 'J': 0, 'n': 0.233784085280560e-5},
                   4: {'I': -8, 'J': 1, 'n': -0.331833715229001e-4},
                   5: {'I': -8, 'J': 3, 'n': 0.107956778514318e-2},
                   6: {'I': -8, 'J': 6, 'n': -0.271382067378863},
                   7: {'I': -8, 'J': 7, 'n': 0.107202262490333e1},
                   8: {'I': -8, 'J': 8, 'n': -0.853821329075382},
                   9: {'I': -6, 'J': 0, 'n': -0.215214194340526e-4},
                   10: {'I': -6, 'J': 1, 'n': 0.769656088222730e-3},
                   11: {'I': -6, 'J': 2, 'n': -0.431136580433864e-2},
                   12: {'I': -6, 'J': 5, 'n': 0.453342167309331},
                   13: {'I': -6, 'J': 6, 'n': -0.507749535873652},
                   14: {'I': -6, 'J': 10, 'n': -0.100475154528389e3},
                   15: {'I': -4, 'J': 3, 'n': -0.219201924648793},
                   16: {'I': -4, 'J': 6, 'n': -0.321087965668917e1},
                   17: {'I': -4, 'J': 10, 'n': 0.607567815637771e3},
                   18: {'I': -3, 'J': 0, 'n': 0.557686450685932e-3},
                   19: {'I': -3, 'J': 2, 'n': 0.187499040029550},
                   20: {'I': -2, 'J': 1, 'n': 0.905368030448107e-2},
                   21: {'I': -2, 'J': 2, 'n': 0.285417173048685},
                   22: {'I': -1, 'J': 0, 'n': 0.329924030996098e-1},
                   23: {'I': -1, 'J': 1, 'n': 0.239897419685483},
                   24: {'I': -1, 'J': 4, 'n': 0.482754995951394e1},
                   25: {'I': -1, 'J': 5, 'n': -0.118035753702231e2},
                   26: {'I': 0, 'J': 0, 'n': 0.169490044091791},
                   27: {'I': 1, 'J': 0, 'n': -0.179967222507787e-1},
                   28: {'I': 1, 'J': 1, 'n': 0.371810116332674e-1},
                   29: {'I': 2, 'J': 2, 'n': -0.536288335065096e-1},
                   30: {'I': 2, 'J': 6, 'n': 0.160697101092520e1}}


    def __init__(self, T: Optional[float] = None, rho: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None, state: Optional[State] = None):
        """
        If all parameters are None (their default), then an empty instance is instanciated. This is to that a `State in Region3` check can be performed easily.
        """
        p = rho
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
                                           phi=gg,
                                           phi_pi=gp, phi_tau=gt,
                                           phi_pipi=gpp, phi_tautau=gtt, phi_pitau=gpt)

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
        return  Region3.b23_const[1] + Region3.b23_const[2] * T + Region3.b23_const[3] * T**2
    
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
            raise ValueError(f'p must be in the range [16.5292, 100]. {p} MPa supplied.')
        return  Region3.b23_const[4] + np.sqrt( (p - Region3.b23_const[5]) / Region3.b23_const[3] )

    @staticmethod
    def h_3ab(p: float) -> float:
        """
        Used to determine wheter to use region 3a or 3b. Eq.1 from supplementary release.
        Args:
            p: Pressure (MPa)
        Returns:
            The value of the enthalpy in region 2 for a given pressure.
        """
        return Region3.table1_supp[1] + Region3.table1_supp[2] * p + Region3.table1_supp[3] * p**2 + Region3.table1_supp[4] * p**3

    @staticmethod
    def subregion(p: Optional[float] = None, h: Optional[float] = None, s: Optional[float] = None) -> str:
        """
        Returns 'a' or 'b' depending on the subregion in region 3 given a (p, h), (p, s) or (h, s) pair.
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
            s: Entropy (kJ/kg/K)
        Returns:
            The subregion in region 3.
        Raises:
            ValueError if an erroneous pair is provided.
        """
        #TODO: Probably check if value is in fact in region3?
        if (s is not None and h is not None and p is None) or (s is not None and p is not None and h is None):
            if s >= s_c:
                return 'b'
            else:
                return 'a'
        elif h is not None and p is not None and s is None:
            h_calc = Region3.h_3ab(p=p)
            if h <= h_calc:
                return 'a'
            else:
                return 'b'
        else:
            raise ValueError('Please supply only one of the following data pairs: (p, h), (p, s) or (h, s).')

    def __contains__(self, other: State) -> bool:
        """
        Overrides the behaviour of the `in` operator to facilitate a `State in Region` query.
        """
        if not isinstance(other, State):
            return False
        else:
            return 623.15 <= other.T <= Region3.T_b23(other.p) and Region3.p_b23(other.T) <= other.p <= 100

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
    def base_der_delta_const_tau(T: float, rho: float) -> float:
        """Derivative of dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` with consant `tau`
        Also known as phi_delta.
        Args:
            T: Temperature (K)
            rho: rho (kg/m^3)
        Returns:
            Derivative of Ideal gas part of dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` with consant `tau`
        """
        delta = rho / rho_c
        tau = T / T_c
        _sum = sum(entry['n'] * entry['I'] * delta**(entry['I'] - 1) * tau**entry['J'] for entry in Region3.table30.values() if entry['I'] is not None)
        other_term = Region3.table30[1]['n'] / delta
        return other_term + _sum

    @staticmethod
    def base_der_tau_const_delta(T: float, rho: float) -> float:
        """Derivative of dimensionless specific Helmholtz free energy (`phi`) with respect to `tau` with consant `delta`
        Also known as phi_tau.
        Args:
            T: Temperature (K)
            rho: rho (kg/m^3)
        Returns:
            Derivative of Ideal gas part of dimensionless specific Helmholtz free energy (`phi`) with respect to `tau` with consant `delta`
        """
        delta = rho / rho_c
        tau = T / T_c
        _sum = sum(entry['n'] * delta**entry['I'] * entry['J'] * tau**(entry['J'] - 1) for entry in Region3.table30.values() if entry['I'] is not None)
        return _sum

    #############################################################
    ################# SECOND ORDER DERIVATIVES ##################
    #############################################################
    @staticmethod
    def base_der2_deltadelta_const_tau(T: float, rho: float) -> float:
        """Second order derivative of Ideal gas part of Dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` with consant `tau`
        Args:
            T: Temperature (K)
            rho: rho (kg/m^3)
        Returns:
            Second order derivative of Ideal gas part of Dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` with consant `tau`
        """
        delta = rho / rho_c
        tau = T / T_c
        _sum = sum(entry['n'] * entry['I'] * (entry['I'] - 1) * delta**(entry['I'] - 2) * tau**entry['J'] for entry in Region3.table30.values() if entry['I'] is not None)
        other_term = - Region3.table30[1]['n'] / delta**2
        return other_term + _sum
    
    @staticmethod
    def base_der2_tautau_const_pi(T: float, rho: float) -> float:
        """Second order derivative of Ideal gas part of Dimensionless specific Helmholtz free energy (`phi`) with respect to `tau` with consant `delta`
        Args:
            T: Temperature (K)
            rho: rho (kg/m^3)
        Returns:
            Second order derivative of Ideal gas part of Dimensionless specific Helmholtz free energy (`phi`) with respect to `tau` with consant `delta`
        """
        delta = rho / rho_c
        tau = T / T_c
        _sum = sum(entry['n'] * delta**entry['I'] * entry['J'] * (entry['J'] - 1) * tau**(entry['J'] - 2) for entry in Region3.table30.values() if entry['I'] is not None)
        return _sum
    
    @staticmethod
    def base_der2_deltatau(T: float, rho: float) -> float:
        """Second order derivative of Ideal gas part of Dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` and then `tau`
        Args:
            T: Temperature (K)
            rho: rho (kg/m^3)
        Returns:
            Second order derivative of Ideal gas part of Dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` and then `tau`
        """
        delta = rho / rho_c
        tau = T / T_c
        _sum = sum(entry['n'] * entry['I'] * delta**(entry['I'] - 1) * entry['J'] * tau**(entry['J'] - 1) for entry in Region3.table30.values() if entry['I'] is not None)
        return _sum

    #############################################################
    ####################### Properties ##########################
    #############################################################
    @property
    def phi(self) -> float:
        """Dimensionless specific Helmholtz free energy (eq. 15)."""
        return self._state.ders['phi']
    
    @property
    def phi_delta(self) -> float:
        """Derivative of Dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` with consant `tau`"""
        return self._state.ders['phi_delta']

    @property
    def phi_tau(self) -> float:
        """Derivative of Dimensionless specific Helmholtz free energy (`phi`) with respect to `tau` with consant `delta`"""
        return self._state.ders['phi_tau']

    @property
    def phi_deltadelta(self) -> float:
        """Second order derivative of Dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` with consant `tau`"""
        return self.self._state.ders['phi_deltadelta']

    @property
    def phi_tautau(self) -> float:
        """Second order derivative of Dimensionless specific Helmholtz free energy (`phi`) with respect to `tau` with consant `delta`"""
        return self.self._state.ders['phi_tautau']
    
    @property
    def phi_deltatau(self) -> float:
        """Second order derivative of Dimensionless specific Helmholtz free energy (`phi`) with respect to `delta` and then `tau`"""
        return self.self._state.ders['phi_deltatau']

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
    def v_pT(self, p:float, T: float) -> float:
        """
        Backwards equations 22, 23 and 23 for calculating Temperature as a function of pressure and enthalpy.
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
        Returns:
            Temperature (K).
        """

    def v_ph(self, p: float, h: float) -> float:
        """
        Backwards equations 2 and 3 for calculating Specific Volume as a function of pressure and enthalpy (supplementary release 2014).
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
        Returns:
            Specific Volume (m^3/kg).
        """
        reg = self.subregion(p=p, h=h)
        if reg == 'a':
            _pi = p / 100
            eta = h / 2100
            v = 0.0028 * sum(entry['n'] * (_pi + 0.128)**entry['I'] * (eta - 0.727)**entry['J'] for entry in Region3.table6_supp.values())
        elif reg == 'b':
            _pi = p / 100
            eta = h / 2800
            v = 0.0088 * sum(entry['n'] * (_pi + 0.0661)**entry['I'] * (eta - 0.72)**entry['J'] for entry in Region3.table7_supp.values())

        return v
        # TODO: Check if state is in region:
        #if State(p=p, T=T) in self:
        #    return T
        #else:
        #    raise ValueError(f'State out of bounds. {T}')

    def rho_ph(self, p: float, h: float) -> float:
        """
        Backwards equations 2 and 3 for calculating density as a function of pressure and enthalpy (supplementary release 2014).
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
        Returns:
            Density (kg/m^3).
        """
        return 1 / self.v_ph(p, h)

    def T_ph(self, p: float, h: float) -> float:
        """
        Backwards equations 2 and 3 for calculating Temperature as a function of pressure and enthalpy (supplementary release 2014).
        Args:
            p: Pressure (MPa).
            h: Enthalpy (kJ/kg).
        Returns:
            Temperature (K).
        """
        reg = self.subregion(p=p, h=h)
        if reg == 'a':
            _pi = p / 100
            eta = h / 2300
            T = 760 * sum(entry['n'] * (_pi + 0.24)**entry['I'] * (eta - 0.615)**entry['J'] for entry in Region3.table3_supp.values())
        elif reg == 'b':
            _pi = p / 100
            eta = h / 2800
            T = 860 * sum(entry['n'] * (_pi + 0.298)**entry['I'] * (eta - 0.72)**entry['J'] for entry in Region3.table4_supp.values())

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
