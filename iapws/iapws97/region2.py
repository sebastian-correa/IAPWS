import numpy as np
from typing import Optional, Dict
from collections import defaultdict

from ._utils import State, Region, R, _p_s

class Region2(Region):
    """
    Region2 implements Region2 of the IAPWS97 standard.

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
    # ROADMAP:
        #TODO: Metastable region.

    table10 = {1: {'J': 0, 'n': -0.96927686500217e1},
               2: {'J': 1, 'n': 0.10086655968018e2},
               3: {'J': -5, 'n': -0.56087911283020e-2},
               4: {'J': -4, 'n': 0.71452738081455e-1},
               5: {'J': -3, 'n': -0.40710498223928},
               6: {'J': -2, 'n': 0.14240819171444e1},
               7: {'J': -1, 'n': -0.43839511319450e1},
               8: {'J': 2, 'n': -0.28408632460772},
               9: {'J': 3, 'n': 0.21268463753307e-1}}

    table11 = {1: {'I': 1, 'J': 0, 'n': -0.17731742473213e-2},
                2: {'I': 1, 'J': 1, 'n': -0.17834862292358e-1},
                3: {'I': 1, 'J': 2, 'n': -0.45996013696365e-1},
                4: {'I': 1, 'J': 3, 'n': -0.57581259083432e-1},
                5: {'I': 1, 'J': 6, 'n': -0.50325278727930e-1},
                6: {'I': 2, 'J': 1, 'n': -0.33032641670203e-4},
                7: {'I': 2, 'J': 2, 'n': -0.18948987516315e-3},
                8: {'I': 2, 'J': 4, 'n': -0.39392777243355e-2},
                9: {'I': 2, 'J': 7, 'n': -0.43797295650573e-1},
                10: {'I': 2, 'J': 36, 'n': -0.26674547914087e-4},
                11: {'I': 3, 'J': 0, 'n': 0.20481737692309e-7},
                12: {'I': 3, 'J': 1, 'n': 0.43870667284435e-6},
                13: {'I': 3, 'J': 3, 'n': -0.32277677238570e-4},
                14: {'I': 3, 'J': 6, 'n': -0.15033924542148e-2},
                15: {'I': 3, 'J': 35, 'n': -0.40668253562649e-1},
                16: {'I': 4, 'J': 1, 'n': -0.78847309559367e-9},
                17: {'I': 4, 'J': 2, 'n': 0.12790717852285e-7},
                18: {'I': 4, 'J': 3, 'n': 0.48225372718507e-6},
                19: {'I': 5, 'J': 7, 'n': 0.22922076337661e-5},
                20: {'I': 6, 'J': 3, 'n': -0.16714766451061e-10},
                21: {'I': 6, 'J': 16, 'n': -0.21171472321355e-2},
                22: {'I': 6, 'J': 35, 'n': -0.23895741934104e2},
                23: {'I': 7, 'J': 0, 'n': -0.59059564324270e-17},
                24: {'I': 7, 'J': 11, 'n': -0.12621808899101e-5},
                25: {'I': 7, 'J': 25, 'n': -0.38946842435739e-1},
                26: {'I': 8, 'J': 8, 'n': 0.11256211360459e-10},
                27: {'I': 8, 'J': 36, 'n': -0.82311340897998e1},
                28: {'I': 9, 'J': 13, 'n': 0.19809712802088e-7},
                29: {'I': 10, 'J': 4, 'n': 0.10406965210174e-18},
                30: {'I': 10, 'J': 10, 'n': -0.10234747095929e-12},
                31: {'I': 10, 'J': 14, 'n': -0.10018179379511e-8},
                32: {'I': 16, 'J': 29, 'n': -0.80882908646985e-10},
                33: {'I': 16, 'J': 50, 'n': 0.10693031879409},
                34: {'I': 18, 'J': 57, 'n': -0.33662250574171},
                35: {'I': 20, 'J': 20, 'n': 0.89185845355421e-24},
                36: {'I': 20, 'J': 35, 'n': 0.30629316876232e-12},
                37: {'I': 20, 'J': 48, 'n': -0.42002467698208e-5},
                38: {'I': 21, 'J': 21, 'n': -0.59056029685639e-25},
                39: {'I': 22, 'J': 53, 'n': 0.37826947613457e-5},
                40: {'I': 23, 'J': 39, 'n': -0.12768608934681e-14},
                41: {'I': 24, 'J': 26, 'n': 0.73087610595061e-28},
                42: {'I': 24, 'J': 40, 'n': 0.55414715350778e-16},
                43: {'I': 24, 'J': 58, 'n': -0.94369707241210e-6}}

    table_10_meta = {1: {'J': 0, 'n': -0.96937268393049e1},
                2: {'J': 1, 'n': 0.10087275970006e2},
                3: {'J': -5, 'n': -0.56087911283020e-2},
                4: {'J': -4, 'n': 0.71452738081455e-1},
                5: {'J': -3, 'n': -0.40710498223928},
                6: {'J': -2, 'n': 0.14240819171444e1},
                7: {'J': -1, 'n': -0.43839511319450e1},
                8: {'J': 2, 'n': -0.28408632460772},
                9: {'J': 3, 'n': 0.21268463753307e-1}}

    table16 = {1: {'I': 1, 'J': 0, 'n': -0.73362260186506e-2},
               2: {'I': 1, 'J': 2, 'n': -0.88223831943146e-1},
               3: {'I': 1, 'J': 5, 'n': -0.72334555213245e-1},
               4: {'I': 1, 'J': 11, 'n': -0.40813178534455e-2},
               5: {'I': 2, 'J': 1, 'n': 0.20097803380207e-2},
               6: {'I': 2, 'J': 7, 'n': -0.53045921898642e-1},
               7: {'I': 2, 'J': 16, 'n': -0.76190409086970e-2},
               8: {'I': 3, 'J': 4, 'n': -0.63498037657313e-2},
               9: {'I': 3, 'J': 16, 'n': -0.86043093028588e-1},
               10: {'I': 4, 'J': 7, 'n': 0.75321581522770e-2},
               11: {'I': 4, 'J': 10, 'n': -0.79238375446139e-2},
               12: {'I': 5, 'J': 9, 'n': -0.22888160778447e-3},
               13: {'I': 5, 'J': 10, 'n': -0.26456501482810e-2}}

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

    table20 = {1: {'I': 0, 'J': 0, 'n': 0.10898952318288e4},
               2: {'I': 0, 'J': 1, 'n': 0.84951654495535e3},
               3: {'I': 0, 'J': 2, 'n': -0.10781748091826e3},
               4: {'I': 0, 'J': 3, 'n': 0.33153654801263e2},
               5: {'I': 0, 'J': 7, 'n': -0.74232016790248e1},
               6: {'I': 0, 'J': 20, 'n': 0.11765048724356e2},
               7: {'I': 1, 'J': 0, 'n': 0.18445749355790e1},
               8: {'I': 1, 'J': 1, 'n': -0.41792700549624e1},
               9: {'I': 1, 'J': 2, 'n': 0.62478196935812e1},
               10: {'I': 1, 'J': 3, 'n': -0.17344563108114e2},
               11: {'I': 1, 'J': 7, 'n': -0.20058176862096e3},
               12: {'I': 1, 'J': 9, 'n': 0.27196065473796e3},
               13: {'I': 1, 'J': 11, 'n': -0.45511318285818e3},
               14: {'I': 1, 'J': 18, 'n': 0.30919688604755e4},
               15: {'I': 1, 'J': 44, 'n': 0.25226640357872e6},
               16: {'I': 2, 'J': 0, 'n': -0.61707422868339e-2},
               17: {'I': 2, 'J': 2, 'n': -0.31078046629583},
               18: {'I': 2, 'J': 7, 'n': 0.11670873077107e2},
               19: {'I': 2, 'J': 36, 'n': 0.12812798404046e9},
               20: {'I': 2, 'J': 38, 'n': -0.98554909623276e9},
               21: {'I': 2, 'J': 40, 'n': 0.28224546973002e10},
               22: {'I': 2, 'J': 42, 'n': -0.35948971410703e10},
               23: {'I': 2, 'J': 44, 'n': 0.17227349913197e10},
               24: {'I': 3, 'J': 24, 'n': -0.13551334240775e5},
               25: {'I': 3, 'J': 44, 'n': 0.12848734664650e8},
               26: {'I': 4, 'J': 12, 'n': 0.13865724283226e1},
               27: {'I': 4, 'J': 32, 'n': 0.23598832556514e6},
               28: {'I': 4, 'J': 44, 'n': -0.13105236545054e8},
               29: {'I': 5, 'J': 32, 'n': 0.73999835474766e4},
               30: {'I': 5, 'J': 36, 'n': -0.55196697030060e6},
               31: {'I': 5, 'J': 42, 'n': 0.37154085996233e7},
               32: {'I': 6, 'J': 34, 'n': 0.19127729239660e5},
               33: {'I': 6, 'J': 44, 'n': -0.41535164835634e6},
               34: {'I': 7, 'J': 28, 'n': -0.62459855192507e2}}

    table21 = {1: {'I': 0, 'J': 0, 'n': 0.14895041079516e4},
               2: {'I': 0, 'J': 1, 'n': 0.74307798314034e3},
               3: {'I': 0, 'J': 2, 'n': -0.97708318797837e2},
               4: {'I': 0, 'J': 12, 'n': 0.24742464705674e1},
               5: {'I': 0, 'J': 18, 'n': -0.63281320016026},
               6: {'I': 0, 'J': 24, 'n': 0.11385952129658e1},
               7: {'I': 0, 'J': 28, 'n': -0.47811863648625},
               8: {'I': 0, 'J': 40, 'n': 0.85208123431544e-2},
               9: {'I': 1, 'J': 0, 'n': 0.93747147377932},
               10: {'I': 1, 'J': 2, 'n': 0.33593118604916e1},
               11: {'I': 1, 'J': 6, 'n': 0.33809355601454e1},
               12: {'I': 1, 'J': 12, 'n': 0.16844539671904},
               13: {'I': 1, 'J': 18, 'n': 0.73875745236695},
               14: {'I': 1, 'J': 24, 'n': -0.47128737436186},
               15: {'I': 1, 'J': 28, 'n': 0.15020273139707},
               16: {'I': 1, 'J': 40, 'n': -0.21764114219750e-2},
               17: {'I': 2, 'J': 2, 'n': -0.21810755324761e-1},
               18: {'I': 2, 'J': 8, 'n': -0.10829784403677},
               19: {'I': 2, 'J': 18, 'n': -0.46333324635812e-1},
               20: {'I': 2, 'J': 40, 'n': 0.71280351959551e-4},
               21: {'I': 3, 'J': 1, 'n': 0.11032831789999e-3},
               22: {'I': 3, 'J': 2, 'n': 0.18955248387902e-3},
               23: {'I': 3, 'J': 12, 'n': 0.30891541160537e-2},
               24: {'I': 3, 'J': 24, 'n': 0.13555504554949e-2},
               25: {'I': 4, 'J': 2, 'n': 0.28640237477456e-6},
               26: {'I': 4, 'J': 12, 'n': -0.10779857357512e-4},
               27: {'I': 4, 'J': 18, 'n': -0.76462712454814e-4},
               28: {'I': 4, 'J': 24, 'n': 0.14052392818316e-4},
               29: {'I': 4, 'J': 28, 'n': -0.31083814331434e-4},
               30: {'I': 4, 'J': 40, 'n': -0.10302738212103e-5},
               31: {'I': 5, 'J': 18, 'n': 0.28217281635040e-6},
               32: {'I': 5, 'J': 24, 'n': 0.12704902271945e-5},
               33: {'I': 5, 'J': 40, 'n': 0.73803353468292e-7},
               34: {'I': 6, 'J': 28, 'n': -0.11030139238909e-7},
               35: {'I': 7, 'J': 2, 'n': -0.81456365207833e-13},
               36: {'I': 7, 'J': 28, 'n': -0.25180545682962e-10},
               37: {'I': 9, 'J': 1, 'n': -0.17565233969407e-17},
               38: {'I': 9, 'J': 40, 'n': 0.86934156344163e-14}}

    table22 = {1: {'I': -7, 'J': 0, 'n': -0.32368398555242e13},
                2: {'I': -7, 'J': 4, 'n': 0.73263350902181e13},
                3: {'I': -6, 'J': 0, 'n': 0.35825089945447e12},
                4: {'I': -6, 'J': 2, 'n': -0.58340131851590e12},
                5: {'I': -5, 'J': 0, 'n': -0.10783068217470e11},
                6: {'I': -5, 'J': 2, 'n': 0.20825544563171e11},
                7: {'I': -2, 'J': 0, 'n': 0.61074783564516e6},
                8: {'I': -2, 'J': 1, 'n': 0.85977722535580e6},
                9: {'I': -1, 'J': 0, 'n': -0.25745723604170e5},
                10: {'I': -1, 'J': 2, 'n': 0.31081088422714e5},
                11: {'I': 0, 'J': 0, 'n': 0.12082315865936e4},
                12: {'I': 0, 'J': 1, 'n': 0.48219755109255e3},
                13: {'I': 1, 'J': 4, 'n': 0.37966001272486e1},
                14: {'I': 1, 'J': 8, 'n': -0.10842984880077e2},
                15: {'I': 2, 'J': 4, 'n': -0.45364172676660e-1},
                16: {'I': 6, 'J': 0, 'n': 0.14559115658698e-12},
                17: {'I': 6, 'J': 1, 'n': 0.11261597407230e-11},
                18: {'I': 6, 'J': 4, 'n': -0.17804982240686e-10},
                19: {'I': 6, 'J': 10, 'n': 0.12324579690832e-6},
                20: {'I': 6, 'J': 12, 'n': -0.11606921130984e-5},
                21: {'I': 6, 'J': 16, 'n': 0.27846367088554e-4},
                22: {'I': 6, 'J': 20, 'n': -0.59270038474176e-3},
                23: {'I': 6, 'J': 22, 'n': 0.12918582991878e-2}}

    table25 = {1: {'I': -1.5, 'J': -24, 'n': -0.39235983861984e6},
               2: {'I': -1.5, 'J': -23, 'n': 0.51526573827270e6},
               3: {'I': -1.5, 'J': -19, 'n': 0.40482443161048e5},
               4: {'I': -1.5, 'J': -13, 'n': -0.32193790923902e3},
               5: {'I': -1.5, 'J': -11, 'n': 0.96961424218694e2},
               6: {'I': -1.5, 'J': -10, 'n': -0.22867846371773e2},
               7: {'I': -1.25, 'J': -19, 'n': -0.44942914124357e6},
               8: {'I': -1.25, 'J': -15, 'n': -0.50118336020166e4},
               9: {'I': -1.25, 'J': -6, 'n': 0.35684463560015},
               10: {'I': -1.0, 'J': -26, 'n': 0.44235335848190e5},
               11: {'I': -1.0, 'J': -21, 'n': -0.13673388811708e5},
               12: {'I': -1.0, 'J': -17, 'n': 0.42163260207864e6},
               13: {'I': -1.0, 'J': -16, 'n': 0.22516925837475e5},
               14: {'I': -1.0, 'J': -9, 'n': 0.47442144865646e3},
               15: {'I': -1.0, 'J': -8, 'n': -0.14931130797647e3},
               16: {'I': -0.75, 'J': -15, 'n': -0.19781126320452e6},
               17: {'I': -0.75, 'J': -14, 'n': -0.23554399470760e5},
               18: {'I': -0.5, 'J': -26, 'n': -0.19070616302076e5},
               19: {'I': -0.5, 'J': -13, 'n': 0.55375669883164e5},
               20: {'I': -0.5, 'J': -9, 'n': 0.38293691437363e4},
               21: {'I': -0.5, 'J': -7, 'n': -0.60391860580567e3},
               22: {'I': -0.25, 'J': -27, 'n': 0.19363102620331e4},
               23: {'I': -0.25, 'J': -25, 'n': 0.42660643698610e4},
               24: {'I': -0.25, 'J': -11, 'n': -0.59780638872718e4},
               25: {'I': -0.25, 'J': -6, 'n': -0.70401463926862e3},
               26: {'I': 0.25, 'J': 1, 'n': 0.33836784107553e3},
               27: {'I': 0.25, 'J': 4, 'n': 0.20862786635187e2},
               28: {'I': 0.25, 'J': 8, 'n': 0.33834172656196e-1},
               29: {'I': 0.25, 'J': 11, 'n': -0.43124428414893e-4},
               30: {'I': 0.5, 'J': 0, 'n': 0.16653791356412e3},
               31: {'I': 0.5, 'J': 1, 'n': -0.13986292055898e3},
               32: {'I': 0.5, 'J': 5, 'n': -0.78849547999872},
               33: {'I': 0.5, 'J': 6, 'n': 0.72132411753872e-1},
               34: {'I': 0.5, 'J': 10, 'n': -0.59754839398283e-2},
               35: {'I': 0.5, 'J': 14, 'n': -0.12141358953904e-4},
               36: {'I': 0.5, 'J': 16, 'n': 0.23227096733871e-6},
               37: {'I': 0.75, 'J': 0, 'n': -0.10538463566194e2},
               38: {'I': 0.75, 'J': 4, 'n': 0.20718925496502e1},
               39: {'I': 0.75, 'J': 9, 'n': -0.72193155260427e-1},
               40: {'I': 0.75, 'J': 17, 'n': 0.20749887081120e-6},
               41: {'I': 1, 'J': 7, 'n': -0.18340657911379e-1},
               42: {'I': 1, 'J': 18, 'n': 0.29036272348696e-6},
               43: {'I': 1.25, 'J': 3, 'n': 0.21037527893619},
               44: {'I': 1.25, 'J': 15, 'n': 0.25681239729999e-3},
               45: {'I': 1.5, 'J': 5, 'n': -0.12799002933781e-1},
               46: {'I': 1.5, 'J': 18, 'n': -0.82198102652018e-5}}

    table26 = {1: {'I': -6, 'J': 0, 'n': 0.31687665083497e6},
               2: {'I': -6, 'J': 11, 'n': 0.20864175881858e2},
               3: {'I': -5, 'J': 0, 'n': -0.39859399803599e6},
               4: {'I': -5, 'J': 11, 'n': -0.21816058518877e2},
               5: {'I': -4, 'J': 0, 'n': 0.22369785194242e6},
               6: {'I': -4, 'J': 1, 'n': -0.27841703445817e4},
               7: {'I': -4, 'J': 11, 'n': 0.99207436071480e1},
               8: {'I': -3, 'J': 0, 'n': -0.75197512299157e5},
               9: {'I': -3, 'J': 1, 'n': 0.29708605951158e4},
               10: {'I': -3, 'J': 11, 'n': -0.34406878548526e1},
               11: {'I': -3, 'J': 12, 'n': 0.38815564249115},
               12: {'I': -2, 'J': 0, 'n': 0.17511295085750e5},
               13: {'I': -2, 'J': 1, 'n': -0.14237112854449e4},
               14: {'I': -2, 'J': 6, 'n': 0.10943803364167e1},
               15: {'I': -2, 'J': 10, 'n': 0.89971619308495},
               16: {'I': -1, 'J': 0, 'n': -0.33759740098958e4},
               17: {'I': -1, 'J': 1, 'n': 0.47162885818355e3},
               18: {'I': -1, 'J': 5, 'n': -0.19188241993679e1},
               19: {'I': -1, 'J': 8, 'n': 0.41078580492196},
               20: {'I': -1, 'J': 9, 'n': -0.33465378172097},
               21: {'I': 0, 'J': 0, 'n': 0.13870034777505e4},
               22: {'I': 0, 'J': 1, 'n': -0.40663326195838e3},
               23: {'I': 0, 'J': 2, 'n': 0.41727347159610e2},
               24: {'I': 0, 'J': 4, 'n': 0.21932549434532e1},
               25: {'I': 0, 'J': 5, 'n': -0.10320050009077e1},
               26: {'I': 0, 'J': 6, 'n': 0.35882943516703},
               27: {'I': 0, 'J': 9, 'n': 0.52511453726066e-2},
               28: {'I': 1, 'J': 0, 'n': 0.12838916450705e2},
               29: {'I': 1, 'J': 1, 'n': -0.28642437219381e1},
               30: {'I': 1, 'J': 2, 'n': 0.56912683664855},
               31: {'I': 1, 'J': 3, 'n': -0.99962954584931e-1},
               32: {'I': 1, 'J': 7, 'n': -0.32632037778459e-2},
               33: {'I': 1, 'J': 8, 'n': 0.23320922576723e-3},
               34: {'I': 2, 'J': 0, 'n': -0.15334809857450},
               35: {'I': 2, 'J': 1, 'n': 0.29072288239902e-1},
               36: {'I': 2, 'J': 5, 'n': 0.37534702741167e-3},
               37: {'I': 3, 'J': 0, 'n': 0.17296691702411e-2},
               38: {'I': 3, 'J': 1, 'n': -0.38556050844504e-3},
               39: {'I': 3, 'J': 3, 'n': -0.35017712292608e-4},
               40: {'I': 4, 'J': 0, 'n': -0.14566393631492e-4},
               41: {'I': 4, 'J': 1, 'n': 0.56420857267269e-5},
               42: {'I': 5, 'J': 0, 'n': 0.41286150074605e-7},
               43: {'I': 5, 'J': 1, 'n': -0.20684671118824e-7},
               44: {'I': 5, 'J': 2, 'n': 0.16409393674725e-8}}

    table27 = {1: {'I': -2, 'J': 0, 'n': 0.90968501005365e3},
               2: {'I': -2, 'J': 1, 'n': 0.24045667088420e4},
               3: {'I': -1, 'J': 0, 'n': -0.59162326387130e3},
               4: {'I': 0, 'J': 0, 'n': 0.54145404128074e3},
               5: {'I': 0, 'J': 1, 'n': -0.27098308411192e3},
               6: {'I': 0, 'J': 2, 'n': 0.97976525097926e3},
               7: {'I': 0, 'J': 3, 'n': -0.46966772959435e3},
               8: {'I': 1, 'J': 0, 'n': 0.14399274604723e2},
               9: {'I': 1, 'J': 1, 'n': -0.19104204230429e2},
               10: {'I': 1, 'J': 3, 'n': 0.53299167111971e1},
               11: {'I': 1, 'J': 4, 'n': -0.21252975375934e2},
               12: {'I': 2, 'J': 0, 'n': -0.31147334413760},
               13: {'I': 2, 'J': 1, 'n': 0.60334840894623},
               14: {'I': 2, 'J': 2, 'n': -0.42764839702509e-1},
               15: {'I': 3, 'J': 0, 'n': 0.58185597255259e-2},
               16: {'I': 3, 'J': 1, 'n': -0.14597008284753e-1},
               17: {'I': 3, 'J': 5, 'n': 0.56631175631027e-2},
               18: {'I': 4, 'J': 0, 'n': -0.76155864584577e-4},
               19: {'I': 4, 'J': 1, 'n': 0.22440342919332e-3},
               20: {'I': 4, 'J': 4, 'n': -0.12561095013413e-4},
               21: {'I': 5, 'J': 0, 'n': 0.63323132660934e-6},
               22: {'I': 5, 'J': 1, 'n': -0.20541989675375e-5},
               23: {'I': 5, 'J': 2, 'n': 0.36405370390082e-7},
               24: {'I': 6, 'J': 0, 'n': -0.29759897789215e-8},
               25: {'I': 6, 'J': 1, 'n': 0.10136618529763e-7},
               26: {'I': 7, 'J': 0, 'n': 0.59925719692351e-11},
               27: {'I': 7, 'J': 1, 'n': -0.20677870105164e-10},
               28: {'I': 7, 'J': 3, 'n': -0.20874278181886e-10},
               29: {'I': 7, 'J': 4, 'n': 0.10162166825089e-9},
               30: {'I': 7, 'J': 5, 'n': -0.16429828281347e-9}}

    table5_supp = {1: -0.349898083432139e4, 2: 0.257560716905876e4,
                   3: -0.421073558227969e3, 4: 0.276349063799944e2}

    table6_supp = {1: {'I': 0, 'J': 1, 'n': -0.182575361923032e-1},
                   2: {'I': 0, 'J': 3, 'n': -0.125229548799536},
                   3: {'I': 0, 'J': 6, 'n': 0.592290437320145},
                   4: {'I': 0, 'J': 16, 'n': 0.604769706185122e1},
                   5: {'I': 0, 'J': 20, 'n': 0.238624965444474e3},
                   6: {'I': 0, 'J': 22, 'n': -0.298639090222922e3},
                   7: {'I': 1, 'J': 0, 'n': 0.512250813040750e-1},
                   8: {'I': 1, 'J': 1, 'n': -0.437266515606486},
                   9: {'I': 1, 'J': 2, 'n': 0.413336902999504},
                   10: {'I': 1, 'J': 3, 'n': -0.516468254574773e1},
                   11: {'I': 1, 'J': 5, 'n': -0.557014838445711e1},
                   12: {'I': 1, 'J': 6, 'n': 0.128555037824478e2},
                   13: {'I': 1, 'J': 10, 'n': 0.114144108953290e2},
                   14: {'I': 1, 'J': 16, 'n': -0.119504225652714e3},
                   15: {'I': 1, 'J': 20, 'n': -0.284777985961560e4},
                   16: {'I': 1, 'J': 22, 'n': 0.431757846408006e4},
                   17: {'I': 2, 'J': 3, 'n': 0.112894040802650e1},
                   18: {'I': 2, 'J': 16, 'n': 0.197409186206319e4},
                   19: {'I': 2, 'J': 20, 'n': 0.151612444706087e4},
                   20: {'I': 3, 'J': 0, 'n': 0.141324451421235e-1},
                   21: {'I': 3, 'J': 2, 'n': 0.585501282219601},
                   22: {'I': 3, 'J': 3, 'n': -0.297258075863012e1},
                   23: {'I': 3, 'J': 6, 'n': 0.594567314847319e1},
                   24: {'I': 3, 'J': 16, 'n': -0.623656565798905e4},
                   25: {'I': 4, 'J': 16, 'n': 0.965986235133332e4},
                   26: {'I': 5, 'J': 3, 'n': 0.681500934948134e1},
                   27: {'I': 5, 'J': 16, 'n': -0.633207286824489e4},
                   28: {'I': 6, 'J': 3, 'n': -0.558919224465760e1},
                   29: {'I': 7, 'J': 1, 'n': 0.400645798472063e-1}}

    table7_supp = {1: {'I': 0, 'J': 0, 'n': 0.801496989929495e-1},
                   2: {'I': 0, 'J': 1, 'n': -0.543862807146111},
                   3: {'I': 0, 'J': 2, 'n': 0.337455597421283},
                   4: {'I': 0, 'J': 4, 'n': 0.890555451157450e1},
                   5: {'I': 0, 'J': 8, 'n': 0.313840736431485e3},
                   6: {'I': 1, 'J': 0, 'n': 0.797367065977789},
                   7: {'I': 1, 'J': 1, 'n': -0.121616973556240e1},
                   8: {'I': 1, 'J': 2, 'n': 0.872803386937477e1},
                   9: {'I': 1, 'J': 3, 'n': -0.169769781757602e2},
                   10: {'I': 1, 'J': 5, 'n': -0.186552827328416e3},
                   11: {'I': 1, 'J': 12, 'n': 0.951159274344237e5},
                   12: {'I': 2, 'J': 1, 'n': -0.189168510120494e2},
                   13: {'I': 2, 'J': 6, 'n': -0.433407037194840e4},
                   14: {'I': 2, 'J': 18, 'n': 0.543212633012715e9},
                   15: {'I': 3, 'J': 0, 'n': 0.144793408386013},
                   16: {'I': 3, 'J': 1, 'n': 0.128024559637516e3},
                   17: {'I': 3, 'J': 7, 'n': -0.672309534071268e5},
                   18: {'I': 3, 'J': 12, 'n': 0.336972380095287e8},
                   19: {'I': 4, 'J': 1, 'n': -0.586634196762720e3},
                   20: {'I': 4, 'J': 16, 'n': -0.221403224769889e11},
                   21: {'I': 5, 'J': 1, 'n': 0.171606668708389e4},
                   22: {'I': 5, 'J': 12, 'n': -0.570817595806302e9},
                   23: {'I': 6, 'J': 1, 'n': -0.312109693178482e4},
                   24: {'I': 6, 'J': 8, 'n': -0.207841384633010e7},
                   25: {'I': 6, 'J': 18, 'n': 0.305605946157786e13},
                   26: {'I': 7, 'J': 1, 'n': 0.322157004314333e4},
                   27: {'I': 7, 'J': 16, 'n': 0.326810259797295e12},
                   28: {'I': 8, 'J': 1, 'n': -0.144104158934487e4},
                   29: {'I': 8, 'J': 3, 'n': 0.410694867802691e3},
                   30: {'I': 8, 'J': 14, 'n': 0.109077066873024e12},
                   31: {'I': 8, 'J': 18, 'n': -0.247964654258893e14},
                   32: {'I': 12, 'J': 10, 'n': 0.188801906865134e10},
                   33: {'I': 14, 'J': 16, 'n': -0.123651009018773e15}}

    table8_supp = {1: {'I': 0, 'J': 0, 'n': 0.112225607199012},
                   2: {'I': 0, 'J': 1, 'n': -0.339005953606712e1},
                   3: {'I': 0, 'J': 2, 'n': -0.320503911730094e2},
                   4: {'I': 0, 'J': 3, 'n': -0.197597305104900e3},
                   5: {'I': 0, 'J': 4, 'n': -0.407693861553446e3},
                   6: {'I': 0, 'J': 8, 'n': 0.132943775222331e5},
                   7: {'I': 1, 'J': 0, 'n': 0.170846839774007e1},
                   8: {'I': 1, 'J': 2, 'n': 0.373694198142245e2},
                   9: {'I': 1, 'J': 5, 'n': 0.358144365815434e4},
                   10: {'I': 1, 'J': 8, 'n': 0.423014446424664e6},
                   11: {'I': 1, 'J': 14, 'n': -0.751071025760063e9},
                   12: {'I': 2, 'J': 2, 'n': 0.523446127607898e2},
                   13: {'I': 2, 'J': 3, 'n': -0.228351290812417e3},
                   14: {'I': 2, 'J': 7, 'n': -0.960652417056937e6},
                   15: {'I': 2, 'J': 10, 'n': -0.807059292526074e8},
                   16: {'I': 2, 'J': 18, 'n': 0.162698017225669e13},
                   17: {'I': 3, 'J': 0, 'n': 0.772465073604171},
                   18: {'I': 3, 'J': 5, 'n': 0.463929973837746e5},
                   19: {'I': 3, 'J': 8, 'n': -0.137317885134128e8},
                   20: {'I': 3, 'J': 16, 'n': 0.170470392630512e13},
                   21: {'I': 3, 'J': 18, 'n': -0.251104628187308e14},
                   22: {'I': 4, 'J': 18, 'n': 0.317748830835520e14},
                   23: {'I': 5, 'J': 1, 'n': 0.538685623675312e2},
                   24: {'I': 5, 'J': 4, 'n': -0.553089094625169e5},
                   25: {'I': 5, 'J': 6, 'n': -0.102861522421405e7},
                   26: {'I': 5, 'J': 14, 'n': 0.204249418756234e13},
                   27: {'I': 6, 'J': 8, 'n': 0.273918446626977e9},
                   28: {'I': 6, 'J': 18, 'n': -0.263963146312685e16},
                   29: {'I': 10, 'J': 7, 'n': -0.107890854108088e10},
                   30: {'I': 12, 'J': 7, 'n': -0.296492620980124e11},
                   31: {'I': 16, 'J': 10, 'n': -0.111754907323424e16}}

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
        return f'Region2(p={self.p}, T={self.T})'

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
        return R * T * (Region2.base_eqn_id_gas(T, p) + Region2.base_eqn_residual(T, p))

    @staticmethod
    def specific_gibbs_free_energy(T: float, p: float) -> float:
        """Alias for `self.base_eqn`"""
        return Region2.base_eqn(T, p) * R * T

    @staticmethod
    def base_eqn_id_gas(T: float, p: float) -> float:
        """
        Ideal gas part (`gammaO`) of the dimensionless Gibbs free energy.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Ideal gas part of the dimensionless specific Gibbs free energy.
        """
        tau = 540 / T
        return np.log(p) + sum(entry['n'] * tau**entry['J'] for entry in Region2.table10.values())

    @staticmethod
    def base_eqn_residual(T: float, p: float) -> float:
        """
        Residual part (`gammaR`) of the dimensionless Gibbs free energy.
        Args:
            T: Temperature (K)
            p: Pressure (MPa)
        Returns:
            Residual part of the dimensionless specific Gibbs free energy.
        """
        tau = 540 / T
        return sum(entry['n'] * p**entry['I'] * (tau - 0.5)**entry['J'] for entry in Region2.table11.values())
    
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
