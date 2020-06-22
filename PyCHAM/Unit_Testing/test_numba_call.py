from numba import jit, f8, int32
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import matplotlib.pyplot as plt
import numpy as np
import importlib # for reloading changed test_numba_func
import test_numba_func


t_end = 10.0
y_rec = np.zeros((int(t_end+1), 2)) # dependent variable record
t = 1.0 # integration time step
t_tot = 0.0 # cumulative time
count = int(0)
y_rec[count, 1] = 2.0 # starting value of dependent variable
y_rec[count, 0] = t_tot # starting value of independent variable
a = 2.0
num_eqn = 1

# write dydt code
f = open('test_numba_func.py', mode='w')
f.write('import numpy as np\n')
f.write('\n')
f.write('def dydt(t, y): # define function:\n')
f.write('	dydt = np.zeros((len(y)))\n')
f.write('	num_eqn = ' + str(num_eqn) + '\n')
f.write('	if num_eqn == 1:\n')
f.write('		dydt[0] = y[0]*0.1\n')
f.write('	else:\n')
f.write('		dydt[0] = y[0]*0.2\n')
f.write('	return(dydt)')
f.close()
test_numba_func = importlib.reload(test_numba_func) # imports updated version
dydt = test_numba_func.dydt
dydt = jit(f8[:](f8, f8[:]), nopython=True)(dydt)

while t_tot<t_end:

	# pretend that boundary conditions change at t_tot=5.0
	if t_tot == 5.0:
		num_eqn = 10.0
		# write dydt code
		f = open('test_numba_func.py', mode='w')
		f.write('import numpy as np\n')
		f.write('\n')
		f.write('def dydt(t, y): # define function:\n')
		f.write('	dydt = np.zeros((len(y)))\n')
		f.write('	num_eqn = ' + str(num_eqn) + '\n')
		f.write('	if num_eqn == 1:\n')
		f.write('		dydt[0] = y[0]*0.1\n')
		f.write('	else:\n')
		f.write('		dydt[0] = y[0]*0.2\n')
		f.write('	return(dydt)')
		f.close()
		test_numba_func = importlib.reload(test_numba_func) # imports updated version
		dydt = test_numba_func.dydt
		dydt = jit(f8[:](f8, f8[:]), nopython=True)(dydt)
		
	mod = Explicit_Problem(dydt, y_rec[count, 1])
	mod_sim = CVode(mod) # define a solver instance
	t_array, res = mod_sim.simulate(t)
	
	t_tot += 1.0
	count += 1
	y_rec[count, 0] = t_tot+t
	y_rec[count, 1] = res[-1]

plt.plot(y_rec[:, 0], y_rec[:, 1])
plt.show()