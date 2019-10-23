# -*- coding: utf-8 -*-
##################################################################################################### 
# Python function to hold expressions for calculating rate coefficients for a given equation number # 
#    Copyright (C) 2017  David Topping : david.topping@manchester.ac.uk                             # 
#                                      : davetopp80@gmail.com                                       # 
#    Personal website: davetoppingsci.com                                                           # 
#                                                                                                   # 
#    This program does not have a license, meaning the deault copyright law applies.                # 
#    Only users who have access to the private repository that holds this file may                  # 
#    use it or develop it, but may not distribute it.                                               # 
#                                                                                                   # 
#                                                                                                   # 
##################################################################################################### 
# Minor modified by XSX
# File Created as 2019-10-21 11:58:04.207923

import numpy

def evaluate_rates(RO2, H2O, TEMP, lightm):
    # mcm_constants_dict: given by mcm_constants.py
    # RO2: specified by the chemical scheme. eg: subset of MCM
    # H2O, TEMP: given by the user
    # lightm: given by the user and is 0 for lights off and 1 for on
    # Creating reference to constant values used in rate expressions
    # mcm_constants_dict: given by MCM_constants.py
    KRO2NO = 9.031312353614072e-12 
    KRO2HO2 = 2.277786778370343e-11 
    KAPHO2 = 1.3915587832143401e-11 
    KAPNO = 1.9837375038963997e-11 
    KRO2NO3 = 2.3e-12 
    KNO3AL = 2.733975964953383e-15 
    KDEC = 1000000.0 
    KROPRIM = 9.140096052746274e-15 
    KROSEC = 9.140096052746274e-15 
    KCH3O2 = 3.5035432056788604e-13 
    K298CH3O2 = 3.5e-13 
    KD0 = 0.2858654787161464 
    KDI = 0.0003865354977050049 
    KRD = 739.5581529081514 
    FCD = 0.3 
    NCD = 1.4140560065060288 
    FD = 0.7903214205531268 
    KBPAN = 0.0003050747741265704 
    KC0 = 6.944423033246356e-09 
    KCI = 1.2066992504885919e-11 
    KRC = 575.4891312341963 
    FCC = 0.3 
    NC = 1.4140560065060288 
    FC = 0.7785522770353273 
    KFPAN = 9.378487940710399e-12 
    K10 = 2.4859762715193917e-12 
    K1I = 2.994437979215886e-11 
    KR1 = 0.08301979499239324 
    FC1 = 0.85 
    NC1 = 0.8396379643428482 
    F1 = 0.9406666674267817 
    KMT01 = 2.1592172418682586e-12 
    K20 = 3.2297706766030638e-12 
    K2I = 2.2965879943120184e-11 
    KR2 = 0.1406334390235544 
    FC2 = 0.6 
    NC2 = 1.0317479120127726 
    F2 = 0.7380514176422049 
    KMT02 = 2.0898360024993884e-12 
    K30 = 9.088989095412873e-11 
    K3I = 1.8976508649648676e-12 
    KR3 = 47.89600270111406 
    FC3 = 0.35 
    NC3 = 1.32903358367515 
    F3 = 0.667632587690726 
    KMT03 = 1.2410227777423264e-12 
    K40 = 3.1017403629106415 
    K4I = 0.07031265481707286 
    KR4 = 44.1135435858624 
    FC4 = 0.35 
    NC4 = 1.32903358367515 
    F4 = 0.660504313532291 
    KMT04 = 0.04541236902085684 
    KMT05 = 2.283940589846278e-13 
    KMT06 = 2.035860025518587 
    K70 = 1.848748572104578e-11 
    K7I = 3.3061295871595854e-11 
    KR7 = 0.5591881755889987 
    FC7 = 0.8106127881714895 
    NC7 = 0.8658069174083853 
    F7 = 0.8240584965940714 
    KMT07 = 9.770962817450986e-12 
    K80 = 8.275074956372096e-11 
    K8I = 4.1e-11 
    KR8 = 2.018310964968804 
    FC8 = 0.4 
    NC8 = 1.2553838110134876 
    F8 = 0.4209573960687299 
    KMT08 = 1.1541070639631594e-11 
    K90 = 4.519264808559091e-12 
    K9I = 4.7e-12 
    KR9 = 0.9615457039487428 
    FC9 = 0.6 
    NC9 = 1.0317479120127726 
    F9 = 0.6000834874902236 
    KMT09 = 1.382550598618549e-12 
    K100 = 0.3096450863695991 
    K10I = 0.25743931345505155 
    KR10 = 1.202788658087618 
    FC10 = 0.6 
    NC10 = 1.0317479120127726 
    F10 = 0.6018431492208056 
    KMT10 = 0.08460084140946005 
    K1 = 1.1226940562514128e-13 
    K3 = 5.72157393619744e-32 
    K4 = 4.3101755489051554e-14 
    K2 = 4.18218325291928e-14 
    KMT11 = 1.5409123815433408e-13 
    K120 = 1.1347189485899005e-11 
    K12I = 1.3056412453398129e-12 
    KR12 = 8.690893862613637 
    FC12 = 0.525 
    NC12 = 1.1053976846744347 
    F12 = 0.687799880106919 
    KMT12 = 8.05353528642086e-13 
    K130 = 6.366695916021298e-11 
    K13I = 1.8e-11 
    KR13 = 3.537053286678499 
    FC13 = 0.36 
    NC13 = 1.3134958240255452 
    F13 = 0.41900056997297147 
    KMT13 = 5.879695584555179e-12 
    K140 = 17.009378371570726 
    K14I = 4.564267848402964 
    KR14 = 3.7266389564587676 
    FC14 = 0.4 
    NC14 = 1.2553838110134876 
    F14 = 0.46809787358991245 
    KMT14 = 1.6845064580062052 
    K150 = 2.157869078671962e-09 
    K15I = 9.047445679918368e-12 
    KR15 = 238.50588939832485 
    FC15 = 0.48 
    NC15 = 1.1548236285330042 
    F15 = 0.8692614914761878 
    KMT15 = 7.831759371132426e-12 
    K160 = 2.012292942193442e-07 
    K16I = 3.018614791212477e-11 
    KR16 = 6666.279341277496 
    FC16 = 0.5 
    NC16 = 1.1323080944932562 
    F16 = 0.9456542772542452 
    KMT16 = 2.8541378431417922e-11 
    K170 = 1.2422194910011785e-10 
    K17I = 1e-12 
    KR17 = 124.22194910011785 
    FC17 = 0.37515544518140886 
    NC17 = 1.2907517069311147 
    F17 = 0.7634489419941073 
    KMT17 = 7.573521758322604e-13 
    KMT18 = 2.2003659767410087e-12 
    KPPN0 = 1.551857482767404 
    KPPNI = 0.00041081313889657854 
    KRPPN = 3777.5264124599516 
    FCPPN = 0.36 
    NCPPN = 1.3134958240255452 
    FPPN = 0.8856983401448923 
    KBPPN = 0.00014785358968353823 
    J = [0.0, (5.0588119900733315e-05-5.28635068062321e-05j), (0.0002918675721471044+0.0003964587223093251j), (-8.049114614634952e-06+9.545098109617479e-06j), (0.011110839329396644+0.01069967217527265j), (0.023656023574982706+0.013790252809115113j), (0.17437464562394403+0.09232662230865064j), (0.0024430019681559384+0.0026179661243655814j), (-7.693506541422756e-07-6.782738407386116e-07j), 0.0, 0.0, (-4.442989459868073e-05+4.120018640748433e-05j), (6.653382894472865e-06+9.191958740977994e-05j), (-7.539916611133045e-06-5.550785067016451e-06j), (-3.439306665917984e-05-7.3481239527480636e-06j), (-2.8956690626312352e-05+2.0351103028337627e-05j), (-1.7371940114281233e-05+1.22092040015994e-05j), (-7.722106997208e-05+7.071050101069359e-05j), (4.85955915458031e-06+1.4340463862450509e-05j), (4.85955915458031e-06+1.4340463862450509e-05j), 0.0, (1.8778678096989683e-07-7.509378743323205e-07j), (-6.920475072441691e-06-2.057823540048321e-06j), (7.880822143030874e-06+2.301799787896688e-05j), (7.880822143030874e-06+2.301799787896688e-05j), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (7.859992498167215e-05+3.401322481731835e-05j), (1.1850273569187096e-05+5.128071294590582e-06j), (-2.1180416685342023e-05+4.35806242888813e-05j), (0.00016542040521635163+9.782935493278283e-05j), (0.0003770146199095273+0.00018912418156107668j), 0.0, 0.0, 0.0, 0.0, 0.0, (-5.013125312336557e-06+7.790787286991982e-06j), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (-1.598771240989052e-06-8.400857005678167e-07j), (-1.5624557098392968e-06-1.5046355534484377e-06j), (-2.313356871336077e-06-1.6367359293139654e-06j), (-4.410822563249446e-06-1.6036647091260393e-06j), (-1.3276499154118256e-05+1.086861359458105e-06j), (-8.932448533242091e-06-4.212435795923999e-07j), (-2.2220522279732937e-06-2.978941060992663e-06j), 0.0, 0.0, 0.0, (2.946605356105653e-06+0.0009379305298505687j)] 
    
    if lightm == 0:
    	J = [0]*len(J)
    # Environmental Variables: M, O2, N2
    M = 2.55e19 # 3rd body; number of molecules in per unit volume
    N2 = 0.79*M # Nitrogen mass mixing ratio : 79%
    O2 = 0.2096*M # Oxygen mass mixing ratio : 20.95%
    rate_values = numpy.zeros(1)
    # reac_coef has been formatted so that python can recognize it
    rate_values[0] = 1.2e-11*numpy.exp(440/TEMP)*0.055

    return rate_values
