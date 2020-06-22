import numpy as np

def dydt(t, y): # define function:
	dydt = np.zeros((len(y)))
	num_eqn = 10.0
	if num_eqn == 1:
		dydt[0] = y[0]*0.1
	else:
		dydt[0] = y[0]*0.2
	return(dydt)