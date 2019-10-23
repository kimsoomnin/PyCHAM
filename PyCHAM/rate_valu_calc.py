# function to calculate gas-phase reaction coefficients

import numpy as np
from Rate_coefficients import evaluate_rates # calculate the rate coef. for each equation


def rate_valu_calc(RO2_indices, H2O, TEMP, lightm, y):

	# ---------------------------------------------
	# inputs:
	
	# lightm - whether lights off (0) or on (1)
	# ---------------------------------------------
	
	# calculate total RO2 concentration
	if (RO2_indices.size == 0):
		RO2 = 0
	else:
		RO2 = np.sum(y[RO2_indices[:,1]])
	
	# Calculate the new rate coef. array (/s) 
	reac_coef = evaluate_rates(RO2, H2O, TEMP, lightm)
	
	return reac_coef
