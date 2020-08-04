'''module to solve the equations for fundamental processes in PyCHAM'''
# equations included: ODEs for gas-phase reactions, partitioning of gas-particle and 
# gas-wall; also solution of coagulation, deposition of particles to wall and nucleation 

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import numba
from numba import jit, f8
import matplotlib.pyplot as plt
import ipdb
from kimt_calc import kimt_calc
from recording import recording
from mov_cen_main import mov_cen_main as movcen # moving centre method for rebinning
from coag import coag
from wallloss import wallloss
from nuc import nuc
import scipy.constants as si
from rate_valu_calc import rate_valu_calc # function to update rate coefficients
import math
from pp_dursim import pp_dursim
from water_calc import water_calc
from volat_calc import volat_calc
import matplotlib.pyplot as plt
import sys
import time

def ode_gen(y, num_speci, num_eqn, rindx, pindx, rstoi, pstoi, H2Oi, 
			TEMP, RO2_indices, num_sb, 
			Psat, mfp, accom_coeff, surfT, y_dens, 
			N_perbin, DStar_org, y_mw, x, core_diss, Varr, Vbou, RH, rad0, 
			Vol0, end_sim_time, pconc, 
			save_step, rbou, therm_sp,
			Cw, light_time, light_stat, nreac, 
			nprod, prodn, reacn, new_partr, MV, nucv1, nucv2, nucv3, inflectDp, 
			pwl_xpre, pwl_xpro, inflectk, nuc_comp, ChamR, Rader, Pnow, testf, kgwt,
			dydt_vst, daytime, lat, lon, act_flux_path, DayOfYear, Ct, injectt, inj_indx,
			corei, const_compi, const_comp, const_infli, Cinfl, act_coeff, p_char, 
			e_field, const_infl_t, int_tol, photo_par_file, Jlen, dil_fac, pconct,
			lowersize, uppersize, mean_rad, std, update_step, Pybel_objects, tempt,
			Cfactor, coag_on, rindx_aq, pindx_aq, rstoi_aq, 
			pstoi_aq, nreac_aq, nprod_aq, prodn_aq, 
			reacn_aq):

	# inputs:---------------------------------------------------
	
	# y - initial concentrations of components (molecules/cc (air)) 
	# num_speci - number of components
	# num_eqn - number of equations related to gas-phase only [0] and related to 
	# 			particulates [1]
	# TEMP - temperature(s) (K) of chamber
	# Psat - saturation vapour pressures (molecules/cm3 (air))
	# y_dens - components' densities (kg/m3)
	# y_mw - components' molecular weights (g/mol)
	# x - radii of particle size bins (um) (excluding walls)
	# therm_sp - thermal speed of components (m/s)
	# DStar_org - gas-phase diffusion coefficient of components (m2/s)
	# Cw - effective absorptive concentration of wall (molecules/cm3 (air))
	# light_time - times (s) of when lights on and lights off (corresponding to light 
	# 				status in light_stat)
	# light_stat - order of lights on (1) and lights off (0)
	# chamA - chamber area (m2)
	# nreac - number of reactants per equation
	# nprod - number of products per equation
	# pindx - indices of equation products (cols) in each equation (rows)
	# prodn - pindx no. of columns
	# reacn - rindx no. of columns
	# rindx - index of reactants per equation
	# rstoi - stoichometry of reactants per equation
	# pstoi - stoichometry of products
	# pconc - concentration of seed particles (#/cc (air)) (1)
	# new_partr - radius of two ELVOC molecules together in a newly nucleating 
	# particle (cm)
	# rad0 - original radius at size bin centres (um)
	# MV - molar volume (cc/mol) (1D array)
	# nuc_comp - index of the nucleating component
	# ChamR - spherical equivalent radius of chamber (below eq. 2 Charan (2018)) (m)
	# Rader - flag of whether to use Rader and McMurry approach (1), or manual inputs
	# 		for wall loss (0) or to ignore particle wall loss (-1)
	# Pnow - pressure inside chamber at experiment start (Pa)
	# testf - flag to say whether in normal mode (0) or testing mode (1)
	# kgwt - mass transfer coefficient for vapour-wall partitioning (/s)
	# dydt_vst - dictionary containing record of tendency to change due to box model 
	#			mechanisms
	# daytime - time of day experiment starts (s)
	# DayOfYear - day of the year (1-365)
	# Ct - concentrations reached due to injections of components after experiment start 
	# (molecules/cc (air))
	# injectt - time of injections (s)
	# inj_indx - index of component being injected after experiment start
	# corei - index of core component
	# const_compi - index of components with constant gas-phase concentration
	# const_infli - index of components with constant influx to chamber
	# Cinfl - concentration of constant influx(es) (molecules/cc.s)
	# act_coeff - activity coefficient of components (dimensionless)
	# p_char - average number of charges per particle (/particle)
	# e_field - average electric field inside chamber (g.m/A.s3)
	# const_infl_t - times of constant influx (s)
	# int_tol - absolute (0 index) and relative (1 index) tolerances for integration
	# photo_par_file - name of file with with estimates for photolysis absorption
	# 					cross-sections and quantum yields
	# Jlen - number of photolysis reactions
	# dil_fac - dilution factor rate (/s)
	# pconct - times (s) at which seep particles injected into chamber
	# lowersize - smallest radius bound (um)
	# uppersize - greatest radius bound (um)
	# mean_rad - mean radius of particles (relevant if only one size bin or number size
	# distribution being calculated (um)
	# std - standard deviation for lognormal size distribution (dimensionless)
	# update_step - time step for updating initial values/constants (s)
	# Pybel_objects - list of Pybel objects representing the species in spec_list
	# (omitting water and core, if present)
	# tempt - times (s) at which chamber temperatures given in TEMP reached
	# Cfactor - one billionth the number of molecules in a unit volume of chamber
	#			at experiment start (molecules/cc)
	# coag_on - flag to say whether coagulation to be modelled or not
	# rindx_aq - reactant indices per aqueous reaction
	# pindx_aq - product indices per aqueous reaction
	# rstoi_aq - stoichiometries of reactants per aqueous reaction
	# pstoi_aq - stoichiometries of products per aqueous reaction
	# nreac_aq - total number of reactants per aqueous reaction
	# nprod_aq - total number of products per aqueous reaction
	# prodn_aq - maximum number of products per aqueous reaction
	# reacn_aq - maximum number of reactants per aqueous reaction
			
	# ------------------------------------------------------------------------------------
	
	# testing mode
	if testf==1:
		return(0, 0, 0, 0) # return dummies
		
	if testf==2:
		# called from test_kimt_calc.py
		# recreate the solute effect used in dydt function below
		sol_eff = np.zeros((num_sb-1))
		for ibin in range(num_sb-1): # size bin loop
			Csit = y[num_speci*(ibin+1):num_speci*(ibin+2)]
			
			# sum of molecular concentrations per bin (molecules/cc (air))
			conc_sum = np.zeros((1))
			conc_sum[0] = ((Csit.sum()-Csit[corei])+Csit[corei]*core_diss)

			sol_eff[ibin] = Csit[H2Oi]/conc_sum # mole fraction of water in particle
		return(sol_eff)
		
	# ------------------------------------------------------------------------------------

	R_gas = si.R # ideal gas constant (kg.m2.s-2.K-1.mol-1)
	NA = si.Avogadro # Avogadro's number (molecules/mol)
	
	step = 0 # ode time interval step number
	t0 = update_step # remember original initial value/constant time step (s)
	y0 = np.zeros((num_speci+num_sb*num_speci))	
	y0[:] = y[:] # initial concentrations (molecules/cc (air))
	y00 = np.zeros((num_speci+num_sb*num_speci))	
	y00[:] = y[:] # initial concentrations (molecules/cc (air))
	
	
	# initial volumes of particles in size bins at start of time steps
	if num_sb>1:
		Vstart = np.zeros((num_sb-1))
		Vstart[:] = Vol0[:]*N_perbin
	else:
		Vstart = 0.0
	sumt = 0.0 # total time integrated over (s)
	
	# record initial values
	if num_sb>0:
		# particle-phase concentrations (molecules/cc (air))
		yp = np.transpose(y[num_speci:-(num_speci)].reshape(num_sb-1, num_speci)) 
	else:
		yp = 0.0
	
	if len(light_time)>0: # check status of lights at simulation start (on of off)
		if (light_time[0] == 0.0): # if a status given for simulation start
			lightm = light_stat[0]
		else: # if no status given for simulation start, default to no lights at this time
			lightm = 0
	else: # if no input provided default to lights off
		lightm = 0
		
	# count on injection times of seed particles
	seedt_count = 0
	
	if num_sb>1:
		# update partitioning coefficients
		[kimt, kelv_fac] = kimt_calc(y, mfp, num_sb, num_speci, accom_coeff, y_mw,   
							surfT, R_gas, TEMP[0], NA, y_dens, N_perbin, DStar_org, 
							x.reshape(1, -1)*1.0e-6, Psat, therm_sp, H2Oi, act_coeff)
	else:
		kimt = 0.0
		kelv_fac = 0.0
	
	save_count = int(1) # count on number of times saving code called
	
	# reaction rate coefficients at experiment time = 0s
	reac_coef = rate_valu_calc(RO2_indices, y[H2Oi], TEMP[0], lightm, y, 
								daytime+sumt, 
								lat, lon, act_flux_path, DayOfYear, Pnow, 
								photo_par_file, Jlen)

	# setup recording matrices and record initial conditions
	[t_out, y_mat, Nresult_dry, Nresult_wet, x2, dydt_vst, 
				Cfactor_vst] = recording(y, N_perbin, x, 
				save_count-1, sumt,
    			0, 0,  0, 0, 0, math.ceil(end_sim_time/save_step), 
				num_speci, num_sb, y_mw[:, 0], y_dens[:, 0]*1.0e-3, yp, Vbou, rindx, 
				rstoi, pindx, nprod, dydt_vst, RO2_indices, H2Oi, TEMP[0], lightm, nreac,
				pconc[:, seedt_count], core_diss, Psat, kelv_fac, kimt, kgwt, Cw, 
				daytime+sumt, lat, lon, 
				act_flux_path, DayOfYear, act_coeff, Pnow, photo_par_file, Jlen, 
				reac_coef, Cfactor, 0)
	
	
	tnew = t0 # initial maximum integration time step (s)
	# number concentration of nucleated particles formed (# particles/cc (air))
	new_part_sum1 = 0.0
	
	# count in number of time steps since time interval was last reduced due to 
	# moving-centre
	tinc_count = 10
	# flag for whether maximum integration time step has been reduced due to boundary 
	# conditions
	bc_red = 0 
	light_time_count = 0 # count on light setting changes
	gasinj_count = 0 # count on injection times of components
	temp_count = 0 # count on changes to temperature
	influx_count = 0 # count on constant influx of gas-phase components
	# count on time since update to initial values/constants last called (s)
	update_count = 0.0 
	# temperature at start (K)
	temp_now = TEMP[0]
	
	# number of components with constant gas-phase concentration and with constant influx
	num_const_compi = len(const_compi)
	const_infli_len = len(const_infli)

	# needs to be a numpy array to be used in integrator
	const_compi = np.array(const_compi)
	
	
	if len(const_infli)>0:
		const_infli = const_infli.astype(int) # ensure int type for use in indexing
	
	print('starting ode solver')
	
	while sumt < end_sim_time: # step through time intervals to do ode
		# start of update for changed initial values/constants ---------------------------
		
		# ---------------------
		# change of light setting check
		
		if len(light_time)>0:
		
			# check whether changes occur at start of this time step
			if (sumt == light_time[light_time_count] and light_time_count>-1):
				# whether lights on (1) or off (0) during this step
				lightm = light_stat[light_time_count]
				
				if (light_time_count<(len(light_stat)-1)):
					light_time_count += 1 # keep count of light setting index
				else:
					light_time_count = -1 # reached end
				# reset flag for time step reduction due to boundary conditions
				bc_red = 0
					
			
			# check whether light on/off changes during proposed integration time step
			if (sumt+tnew > light_time[light_time_count] and light_time_count!=-1):
				# if yes, then reset integration time step so that next step coincides 
				# with change
				tnew = light_time[light_time_count]-sumt
				bc_red = 1 # flag for time step reduction due to boundary conditions
				
		if (len(light_time) == 0): # if no input provided default to lights off
			lightm = 0 
		
		# -------------------------------------------------------------------------------
		# check on updates to temperature (K)
		
		if len(TEMP)>1: # because a temperature must be given for experiment start
		
			# check whether changes occur at start of this time step
			if (sumt == tempt[temp_count]):
				print('updating temperature inside chamber to ' +str(TEMP[temp_count]) + ' K')
				# new temperature (K)
				temp_now = TEMP[temp_count]
				
				# update vapour pressure of water (log10(atm)), but don't change 
				# gas-phase concentration because we assume RH allowed to change with
				# varying temperature
				[_, Psat_water, _] = water_calc(temp_now, RH, 6.02214129e+23)
				
				# update vapour pressures of all components (molecules/cc and Pa), 
				# ignore density output
				[Psat, _, Psat_Pa] = volat_calc(0, Pybel_objects, temp_now, H2Oi,   
								num_speci, Psat_water, [], [], 0, corei, 
								pconc, 0, 0.0, [], 1)

				# note, assume that air pressure inside chamber stays constant despite
				# varying temperature, therefore total molecular concentration must vary
				# (molecules/cc (air))
				ntot = Pnow*(NA/(8.3144598e6*temp_now))
				# remember number of molecules in one billionth of total number prior
				# to update
				Cfactor0 = Cfactor
				# update number of molecules in one billionth of this
				Cfactor = ntot*1.0e-9 # ppb-to-molecules/cc
								
				# update mean free path and thermal speed
				# mean thermal speed of each molecule (m/s) (11.151 Jacobson 2005)
				# note that we need the weight of one molecule, which is why y_mw is divided by
				# Avogadro's constant, and we need it in kg, which is why we multiply by 1e-3
				therm_sp = (np.power((8.0E0*si.k*temp_now)/(np.pi*(y_mw/si.N_A)*1.0E-3), 0.5E0))
				
				# mean free path (m) for each species (16.23 of Jacobson 2005)
				# molecular weight of air (28.966 g/mol taken from table 16.1 Jacobson 2005)
				mfp = (((64.0*DStar_org)/(5*np.pi*therm_sp))*(28.966/(28.966+y_mw))).reshape(-1, 1)
				
				# alter constant concentration (molecules/cc) of any components
				# with constant gas-phase concentration (ppb)
				if num_const_compi>0:
					
					y[const_compi[:]] = y[const_compi[:]]*(Cfactor/Cfactor0)
	
				if (temp_count<(len(tempt)-1)):
					temp_count += 1 # keep count of temperature setting index
				else:
					temp_count = -1 # reached end
				bc_red = 0 # reset flag for time step reduction due to boundary conditions
				
			# check whether light on/off changes during proposed integration time step
			if (sumt+tnew > tempt[temp_count] and temp_count!=-1):
				# if yes, then reset integration time step so that next step coincides 
				# with change
				tnew = tempt[temp_count]-sumt
				bc_red = 1 # flag for time step reduction due to boundary conditions
				
		# ---------------------
		# check on instantaneous injection of components
		
		if len(injectt)>0 and gasinj_count>-1: # if any injections occur
		
			# check whether changes occur at start of this time step
			if (sumt == injectt[gasinj_count]):
				# account for change in gas-phase concentration,
				# convert from ppb/s to molecules/cc.s (air)
				y[inj_indx] += Ct[:, gasinj_count]*Cfactor
				
				if (gasinj_count<(Ct.shape[1]-1)):
					gasinj_count += 1 # update count on injections
				else:
					gasinj_count = -1 # reached end
				bc_red = 0 # reset flag for time step reduction due to boundary conditions
					
			# check whether changes occur during proposed integration time step
			if (sumt+tnew > injectt[gasinj_count] and gasinj_count!=-1):
				# if yes, then reset integration time step so that next step coincides 
				# with change
				tnew = injectt[gasinj_count]-sumt
				bc_red = 1 # flag for time step reduction due to boundary conditions
		
		# ----------------------
		# instantaneous seed particle influx check
		
		if (sum(pconct[0, :])>0) and seedt_count>-1: # if constant influx occurs
		
			# check whether changes occur at start of this time step
			if (sumt == pconct[0, seedt_count]):
				
				# account for change in seed particles
				[y[num_speci:-num_speci], N_perbin, x, 
						Varr] = pp_dursim(y[num_speci:-num_speci], N_perbin, 
									mean_rad[0, seedt_count],
									pconc[:, seedt_count], corei, lowersize, 
									uppersize, num_speci, num_sb, MV, rad0, 
									std[0, seedt_count], y_dens, H2Oi, rbou)
				if (seedt_count<(pconct.shape[1]-1)):
					seedt_count += 1
				else:
					seedt_count = -1 # reached end
				bc_red = 0 # reset flag for time step reduction due to boundary conditions
				
			# check whether changes occur during proposed integration time step
			if (sumt+tnew > pconct[0, seedt_count] and seedt_count!=-1): 
				# if yes, then reset integration time step so that next step coincides 
				# with change
				tnew = pconct[0, seedt_count]-sumt
				bc_red = 1 # flag for time step reduction due to boundary conditions

		
		# ----------------------
		# check on constant influxes of components, note this causes a change in the ode
		# as influx occurs over a period and is not instantaneous
		
		if len(const_infl_t)>0 and influx_count>-1: # if constant influx occurs
		
			# in case influxes begin after simulation start
			if (sumt == 0.0 and const_infl_t[influx_count] != 0.0):
				Cinfl_now = np.zeros((Cinfl.shape[0], 1))
		
			# check whether changes occur at start of this time step
			if (sumt == const_infl_t[influx_count]):
				
				# influx of components now, convert from ppb/s to molecules/cc.s (air)
				Cinfl_now = (Cinfl[:, influx_count]*Cfactor).reshape(-1, 1)
				
				
				# update index counter for constant influxes - used in integrator below
				if (influx_count<(Cinfl.shape[1]-1)):
					influx_count += 1
				else:
					influx_count = -1 # reached end
				bc_red = 0 # reset flag for time step reduction due to boundary conditions
			
			# check whether changes occur during proposed integration time step
			if (sumt+tnew > const_infl_t[influx_count] and influx_count!=-1):
				# if yes, then reset integration time step so that next step coincides 
				# with change
				tnew = const_infl_t[influx_count]-sumt
				bc_red = 1 # flag for time step reduction due to boundary conditions
		
		# end of update for changed boundary conditions ----------------------------------
		
		# check whether time step needs reducing to ensure operator-split time step not
		# overrun
		if update_count+tnew>update_step:
			print('temporarily reducing time step to enable update to particle number concentration at requested interval')
			tnew = update_step-update_count
			bc_red = 1
		
		# --------------------------------------------------------------------------------


		# update integration time step
		if (sumt+tnew)>end_sim_time: # ensure we finish at correct time
			tnew = end_sim_time-sumt # integration time step (s)
		t = tnew # reset maximum integration time (s)
		
		print('cumulative time (s) through simulation: ', sumt)
		a = (np.where(N_perbin>1.0e-10))[0]
		if len(a)>0:
			a = int(a[0])
			
		# update reaction rate coefficients
		[reac_coef, reac_coef_aq] = rate_valu_calc(RO2_indices, y[H2Oi], temp_now, lightm, y, 
									daytime+sumt, 
									lat, lon, act_flux_path, DayOfYear, Pnow, 
									photo_par_file, Jlen)
		
		y0[:] = y[:] # update initial concentrations (molecules/cc (air))
		# update particle volumes at start of time step (um3)
		Vstart = Varr*N_perbin
		redt = 1 # reset time reduction flag
		
		
		if num_sb>1:
			# update partitioning coefficients
			[kimt, kelv_fac] = kimt_calc(y, mfp, num_sb, num_speci, accom_coeff, y_mw,   
							surfT, R_gas, temp_now, NA, y_dens, N_perbin, DStar_org, 
							x.reshape(1, -1)*1.0e-6, Psat, therm_sp, H2Oi, act_coeff)
							
		# ensure no confusion that components are present due to low value fillers for  
		# concentrations (molecules/cc (air))
		y0[y0==1.0e-40] = 0.0
		
		# enter a while loop that continues to decrease the time step until particle
		# size bins don't change by more than one size bin (given by moving centre)
		# note, need to have rstoi and pstoi multiplication in the gas-phase reaction part
		while redt == 1:
			
			# numba compiler to convert to machine code
			@jit(f8[:](f8, f8[:]), nopython=True, cache=False)
			# ode solver -------------------------------------------------------------
			def dydt(t, y):
				
				# empty array to hold rate of change (molecules/cc(air).s)
				dydt = np.zeros((len(y)))
				# gas-phase rate of change ------------------------------------
				for i in range(num_eqn[0]): # equation loop
					
					# gas-phase rate of change (molecules/cc (air).s)
					if (y[rindx[i, 0:nreac[i]]]==0.0).sum()>0: 
						continue # if any reactants not present skip this reaction
					else:
						gprate = ((y[rindx[i, 0:nreac[i]]]**
									rstoi[i, 0:nreac[i]]).prod())*reac_coef[i]
						# loss of reactants
						dydt[rindx[i, 0:nreac[i]]] -= gprate*rstoi[i, 0:nreac[i]]
						# gain of products
						dydt[pindx[i, 0:nprod[i]]] += gprate*pstoi[i, 0:nprod[i]]
				
				# the constant gas-phase influx of components with this property
				if const_infli_len>0:
					for i in range(const_infli_len):
						dydt[const_infli[i]] = dydt[const_infli[i]]+Cinfl_now[i, 0]

				if num_sb>1: # as num_sb includes 1 for wall
					# gas-particle partitioning, based on eqs. 3 and 4 of Zaveri et al.
					# (2008): doi:10.1029/2007JD008782
					# and eq. 3 of Riipinen et al.
					# (2010): doi:10.1016/j.atmosenv.2009.11.022
					# -----------------------------------------------------------
					for ibin in range(num_sb-1): # size bin loop
							
						Csit = y[num_speci*(ibin+1):num_speci*(ibin+2)]
						# sum of molecular concentrations per bin (molecules/cc (air))
						conc_sum = np.zeros((1))
						
						conc_sum[0] = ((Csit.sum()-Csit[corei])+Csit[corei]*core_diss)
						
						# prevent numerical error due to division by zero
						ish = conc_sum==0.0
						conc_sum[ish] = 1.0e-40
							
						# particle surface gas-phase concentration (molecules/cc (air))
						Csit = (Csit/conc_sum)*Psat[:, 0]*kelv_fac[ibin]*act_coeff[:, 0]
							
						# partitioning rate (molecules/cc.s)
						dydt_all = kimt[:, ibin]*(y[0:num_speci]-Csit)
							
						# gas-phase change
						dydt[0:num_speci] -= dydt_all
						# particle-phase change
						dydt[num_speci*(ibin+1):num_speci*(ibin+2)] += dydt_all
						
						# rate of change to particulate components due to
						# reactions in particulates (molecules/cc (air).s)
						for i in range(num_eqn[1]): # particulate equation loop
							if (y[rindx_aq[i, 0:nreac_aq[i]]]==0.0).sum()>0: 
								continue # if any reactants not present skip this reaction
							else:
								gprate = ((y[rindx_aq[i, 0:nreac_aq[i]]]**
											rstoi_aq[i, 0:nreac_aq[i]]).prod())*reac_coef_aq[i]
								# loss of reactants
								dydt[rindx_aq[i, 0:nreac_aq[i]]+num_speci*(ibin+1)] -= gprate*rstoi_aq[i, 0:nreac_aq[i]]
								# gain of products
								dydt[pindx_aq[i, 0:nprod_aq[i]]+num_speci*(ibin+1)] += gprate*pstoi_aq[i, 0:nprod_aq[i]]
				
				if (kgwt*Cw)>1.0e-10:
					# -----------------------------------------------------------
					# gas-wall partitioning (dydt is in molecules/cc.s (air))
					
					# concentration at wall (molecules/cc (air))
					Csit = y[num_speci*num_sb:num_speci*(num_sb+1)]
					Csit = (Psat[:,0]*(Csit/Cw)*act_coeff[:, 0]) # with Raoult term
	
					dydt_all = (kgwt)*(y[0:num_speci]-Csit)
							
					# gas-phase change
					dydt[0:num_speci] -= dydt_all
					# wall concentration change
					dydt[num_speci*num_sb:num_speci*(num_sb+1)] += dydt_all
				
				# constant gas-phase concentration of components with this property
				if num_const_compi>0:
					dydt[const_compi[:]] = 0.0

				return(dydt)
				
			
			mod = Explicit_Problem(dydt, y0)
			mod_sim = CVode(mod) # define a solver instance
			# absolute tolerance, going higher than can 1.0e-3 cause issues with water 
			# vapour
			mod_sim.atol = int_tol[0]
			# relative tolerance, going higher than 1.0e-4 can cause issues with water 
			# vapour
			mod_sim.rtol = int_tol[1]
			# check if total integration time exceeds the next required recording time
			# step, if it does, then limit the integration sub-time step to that of the
			# recording interval (s)
			if sumt+t>save_step*save_count:
				mod_sim.maxh = (save_step*save_count)-sumt
			mod_sim.discr = 'BDF' # the integration approach, default is 'Adams'

			t_array, res = mod_sim.simulate(t)		
			y = res[-1, :] # new concentrations (molecule/cc (air))

			# low value filler for concentrations (molecules/cc (air)) to prevent 
			# numerical errors
			y0[y0==0.0] = 1.0e-40
			y[y==0.0] = 1.0e-40
			
			
			if num_sb>1 and (N_perbin>1.0e-10).sum()>0: # if particles present
				# call on the moving centre method for rebinning particles
				(N_perbin, Varr, y, x, redt, t, bc_red) = movcen(N_perbin, 
				Vbou, num_sb, num_speci, y_mw, x, Vol0, t, 
				t0, tinc_count, y0, MV, Psat[:, 0], bc_red, res, t_array)
				
			else: # if no moving-centre needed, then allow model to continue
				redt = 0
			
			if (redt == 0):
				if t<t0 and tinc_count<=0:
					tnew = t*2.0
				if tnew>t0: # in case tnew exceeds user-defined maximum for time step
					tnew = t0

			# if time step needs reducing then reset concentrations to their
			# values preceding the ode
			if (redt == 1):
				y[:] = y0[:]
			# check whether maximum integration time step can return to original
			if (redt == 0) and (bc_red == 1):
				# return maximum integration time step (s) to original
				tnew = t0
				bc_red = 0
			# start counter to determine number of integrations before next trying to 
			# increasing time interval
			if redt == 1:
				tinc_count = 10 # reset to maximum
			if redt == 0 and tinc_count>-1:
				tinc_count -= 1 # decrease number of steps remaining
			if tinc_count==-1:
				tinc_count = 10 # reset to maximum
		
		sumt += t # total time covered (s)
		step += 1 # ode time interval step number
		update_count += t # count on time since operator-split processes last called (s)

		# start of update to particle number concentration section -----------------------
		# the following particle-phase processes are evaluated to update the particle 
		# number concentration constant: coagulation, particle loss to wall and nucleation
		if update_count >= update_step:
			
			if num_sb>1: 
				if (N_perbin>1.0e-10).sum()>0:
					
					# coagulation
					# y indices due to final element in y being number of ELVOC molecules
					# contributing to newly nucleated particles
					[N_perbin, y[num_speci:-(num_speci)], x, Gi, eta_ai, Varr] = coag(RH, 
							temp_now, x*1.0e-6, (Varr*1.0e-18).reshape(1, -1), 
							y_mw.reshape(-1, 1), x*1.0e-6, 
							np.transpose(y[num_speci::].reshape(num_sb, num_speci)), 
							(N_perbin).reshape(1, -1), update_count, 
							(Vbou*1.0e-18).reshape(1, -1), 
							num_speci, 0, (np.squeeze(y_dens*1.0e-3)), rad0, Pnow, 0,
							np.transpose(y[num_speci::].reshape(num_sb, num_speci)),
							(N_perbin).reshape(1, -1), (Varr*1.0e-18).reshape(1, -1),
							coag_on)
					
					if Rader > -1:
						# particle loss to walls
						[N_perbin, 
						y[num_speci:-(num_speci)]] = wallloss(N_perbin.reshape(-1, 1), 
														y[num_speci:-(num_speci)], Gi, 
														eta_ai, x*2.0e-6, y_mw, 
														Varr*1.0e-18, num_sb, num_speci, 
														temp_now, update_count, 
														inflectDp, pwl_xpre,
														pwl_xpro, inflectk, ChamR, Rader, 
														0, p_char, e_field)
						
				# particle nucleation
				if len(nuc_comp)>0 and (nucv1 != 0.0):
					
					[N_perbin, y, x[0], Varr[0], new_part_sum1] = nuc(sumt, new_part_sum1, 
								N_perbin, y, y_mw.reshape(-1, 1), 
								np.squeeze(y_dens*1.0e-3),  
								num_speci, x[0], new_partr, MV, nucv1, nucv2, 
								nucv3, nuc_comp[0])
			
			# reset count on time since initial value/constant update last called (s)
			update_count = 0
				
		# end of operator-split section --------------------------------------------------
							
		# dilution of aerosol (gases and particles), most likely due to extraction
		# from chamber
		y -= y*(dil_fac*t) # dilution of gases (molecules/cc (air))
		N_perbin -= N_perbin*(dil_fac*t) # dilution of particle phase (#/cc (air))

		
		# save at every time step given by save_step (s) and at end of experiment
		if (save_step*save_count-sumt)<1.0e-10 or (sumt-save_step*save_count)>=0.0 or sumt == end_sim_time:
			
			if num_sb>0: # if particles present
				# particle-phase concentrations (molecules/cc (air))
				yp = np.transpose(y[num_speci:-(num_speci)].reshape(num_sb-1, 
										num_speci))
			else:
				yp = 0.0
			
			# if particles present, the recording time step is at least the same as the
			# step for updating particle concentration and therefore the integration step,
			# so the maximum number of recordings is one per integration step
			# but if no particle phase, then may need to record concentrations in gas 
			# and wall at several times covered by integration
			while ((save_step*save_count-sumt)<1.0e-10):
				
				# convert list to numpy array
				t_array = np.array(t_array)
				# select solution at closest time to recording interval
				indx = np.where(np.abs(((sumt-t)+t_array)-save_step*save_count)==
								np.min(np.abs(((sumt-t)+t_array)-save_step*save_count)))[0][0]
				ythen = res[indx, :] # get concentrations (molecules/cm3) at this time
				# time at this time
				sumt_rec = (sumt-t)+t_array[indx]
				
				# record values
				[t_out, y_mat, Nresult_dry, Nresult_wet, x2, 
					dydt_vst, Cfactor_vst] = recording(ythen, 
					N_perbin, x, save_count, 
					sumt_rec, y_mat, Nresult_dry, Nresult_wet, x2, t_out, int(end_sim_time/save_step), 
					num_speci, num_sb, y_mw[:, 0], y_dens[:, 0]*1.0e-3, yp, Vbou, rindx, 
					rstoi, pindx, nprod, dydt_vst, RO2_indices, H2Oi, temp_now, lightm, nreac,
					pconc, core_diss, Psat, kelv_fac, kimt, kgwt, Cw, daytime+sumt, lat, lon, 
					act_flux_path, DayOfYear, act_coeff, Pnow, photo_par_file, Jlen, 
					reac_coef, Cfactor, Cfactor_vst)
					
				save_count += int(1) # track number of times saved at
		
	return(t_out, y_mat, Nresult_dry, Nresult_wet, x2, dydt_vst, Cfactor_vst)