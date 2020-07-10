'''module called on by PyCHAM to initiate the box model'''
# this module takes PyCHAM inputs from the user_input module and prepares for calling
# the solver, including parsing chemical equations and setting the initial gas- and
# particle-phase conditions

import numpy as np
from ode_gen import ode_gen
import matplotlib.pyplot as plt
import os
import eqn_parser # to parse the .scm file
from init_conc_func import init_conc_func
from pp_intro import pp_intro
from kimt_prep import kimt_prep
from saving import saving
import time # timing how long operations take
import user_input as ui
import pickle # for storing inputs
from volat_calc import volat_calc

def run(testf):
	
	# inputs:
	# testf - test flag, 1 for test mode (called by test_front.py), 2 for test mode 
	# (called by test_PyCHAM.py), 0 for normal mode
	
	if testf==2:
		print('"Run Model" button works fine')
		return()
	if testf==1:
		print('calling user_input.py')
	# module to ask, receive and return required inputs
	[fname, num_sb, lowersize, uppersize, end_sim_time, resfname,
	TEMP, PInit, RH, lat, lon, start_sim_time, act_flux_path, save_step, Cw, 
	ChamR, nucv1, nucv2, nucv3, nuc_comp, new_partr, inflectDp, pwl_xpre,  
	pwl_xpro, inflectk, xmlname, init_conc, Comp0, Rader, vol_Comp, volP, 
	pconc, std, mean_rad, core_diss, light_stat, light_time, kgwt, testm, 
	dydt_trak, DayOfYear, space_mode, Ct, Compt, injectt, seed_name, 
	const_comp, const_infl, Cinfl, act_comp, act_user, seed_mw, 
	umansysprop_update, core_dens, p_char, e_field, const_infl_t, 
	chem_scheme_markers, int_tol, photo_par_file, dil_fac, pconct, accom_coeff_ind, 
	accom_coeff_user, update_step, tempt, coag_on] = ui.run(0, testf)
	
	if testm == 1:
		print('PyCHAM calls front fine, now returning to PyCHAM.py')
		print('Please select the "Plot Results" button to ensure this works fine')
		return()
	
	if testf==1:
		print('user_input.py called and returned fine')
		print('calling eqn_parser.extract_mechanism')
	
	# obtain gas-phase reaction info
	[rindx, pindx, rstoi, pstoi, reac_coef, spec_list, Pybel_objects, num_eqn, num_speci, 
		RO2_indices, nreac, nprod, prodn, 
		reacn, spec_namelist, Jlen] = eqn_parser.extract_mechanism(fname, xmlname, 
							PInit, testf, RH, start_sim_time, lat, 
							lon, act_flux_path, DayOfYear, chem_scheme_markers, 
							photo_par_file)

	if testf==1:
		print('eqn_parser.extract_mechanism called and returned fine')
		print('calling init_conc_func')
	# set up initial gas-phase concentration array
	[y, H2Oi, y_mw, num_speci, 
	Cfactor, y_indx_plot, corei, dydt_vst, spec_namelist, 
							inj_indx, const_compi, 
							const_infli, core_diss, 
							Psat_water, nuci] = init_conc_func(num_speci, 
							Comp0, init_conc, TEMP[0], RH, 
							reac_coef, fname, 
							PInit, start_sim_time, lat, lon, Pybel_objects, testf, pconc,
							act_flux_path, dydt_trak, end_sim_time, save_step, rindx, 
							pindx, num_eqn, nreac, nprod, DayOfYear, 
							spec_namelist, Compt, seed_name, const_comp, const_infl, 
							seed_mw, core_diss, nuc_comp)

	if testf==1:
		print('init_conc_func called and returned fine')
		print('calling kimt_prep')
	# set up partitioning variables
	[DStar_org, mfp, accom_coeff, therm_sp, surfT, Cw, act_coeff] = kimt_prep(y_mw, 
											TEMP[0], 
											num_speci, testf, Cw, act_comp, act_user, 
											accom_coeff_ind, accom_coeff_user, 
											spec_namelist, num_sb)

	# volatility (molecules/cc (air)) and density (rho, kg/m3) of components
	if testf==1:
		print('kimt_prep called and returned fine')
		print('calling volat_calc.py')
		
	[Psat, y_dens, Psat_Pa] = volat_calc(spec_list, Pybel_objects, TEMP[0], H2Oi, 
								num_speci,  
								Psat_water, vol_Comp, volP, testf, corei, pconc,
								umansysprop_update, core_dens, spec_namelist, 0, nuci,
								nuc_comp)

	if testf==1:
		print('volat_calc called and returned fine')
		print('calling wall_prep')
	
	if testf==1:
		print('wall_prep called and returned fine')
		print('calling pp_intro')
	
	# set up particle phase part
	# check whether there are particles at the start of simulation
	if pconct[0, 0] == 0.0:
		pconc_now = pconc[:, 0]
	else:
		pconc_now = np.zeros((1))

	[y, N_perbin, x, Varr, Vbou, rad0, Vol0, rbou, 
							MV, num_sb, nuc_comp, rbou00, upper_bin_rad_amp] = pp_intro(y, 
							num_speci, Pybel_objects, TEMP[0], H2Oi, 
							mfp, accom_coeff, y_mw, surfT, DStar_org, 
							RH, num_sb, lowersize, uppersize, pconc_now, nuc_comp, 
							testf, std[0, 0], mean_rad[0, 0], 
							therm_sp, Cw, y_dens, Psat, core_diss, kgwt, space_mode, 
							corei, spec_namelist, act_coeff)
	
	t1 = time.clock() # get wall clock time before call to solver
	if testf==1:
		print('pp_intro called and returned fine')
		print('calling ode_gen')
	
	# call on ode function
	[t_out, y_mat, Nresult_dry, Nresult_wet, x2, 
				dydt_vst, Cfactor_vst] = ode_gen(y, 
				num_speci, num_eqn, 
				rindx, pindx, 
				rstoi, pstoi, H2Oi, TEMP, RO2_indices, 
				num_sb, Psat, mfp, accom_coeff, surfT, y_dens, N_perbin,
				DStar_org, y_mw, x, core_diss, Varr, Vbou, RH, rad0, Vol0,
				end_sim_time, pconc, save_step, 
				rbou, therm_sp, Cw, light_time, light_stat,
				nreac, nprod, prodn,
				reacn, new_partr, MV, nucv1, nucv2, nucv3, inflectDp, pwl_xpre, 
				pwl_xpro, inflectk, nuc_comp, ChamR, Rader, PInit, testf, kgwt, dydt_vst,
				start_sim_time, lat, lon, act_flux_path, DayOfYear, Ct, injectt, inj_indx,
				corei, const_compi, const_comp, const_infli, Cinfl, act_coeff, p_char, 
				e_field, const_infl_t, int_tol, photo_par_file, Jlen, dil_fac, pconct,
				lowersize, uppersize, mean_rad, std, update_step, Pybel_objects, tempt,
				Cfactor, coag_on)
				
	
	t2 = time.clock() # get wall clock time after call to solver
	time_taken = t2-t1 # computer time taken for entire simulation (s)
	if testf==0: # in normal mode
		print(str('time taken= '+str(time_taken)))
	
		# make new pickle dump to store the indices and names of interesting gas-phase 
		# components along with initial pickle variables
		list_vars = [fname, resfname, y_indx_plot, Comp0]
	if testf==1:
		print('ode_gen called and returned fine')
		print('dumping variables in pickle file')
		# dummy list of variables to dump
		list_vars = ['fnametest','resfnametest', 0, 0]
		with open('test_var_store.pkl','wb') as f:
			pickle.dump(list_vars,f)
	
	if testf==00:
		with open('PyCHAM/var_store.pkl','wb') as f:
			pickle.dump(list_vars,f) 
	
	if testf==1:
		print('dumped successfully')
	# save data
	output_by_sim = saving(fname, y_mat, t_out, Nresult_dry, Nresult_wet, x2, num_sb, 
							y_mw, num_speci, 
							resfname, rbou, Cfactor, MV, testf, dydt_vst, dydt_trak,
							spec_namelist, rbou00, upper_bin_rad_amp, Cfactor_vst, 
							time_taken)
	if testf==1:
		print('saving called and returned successfully')
	return()