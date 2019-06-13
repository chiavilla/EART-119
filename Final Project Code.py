#!/usr/bin/env python
# coding: utf-8

# # Final Project Code
# -----

# In[200]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

#=============================1=======================================
#                    Function Definitions
#=====================================================================
def runge_kutta_vec(tn, Yn, fct_RHS, params):
    """
    fourth order runge kutta stepper, for single or system of ODEs
    :input:
                 tn           - current time step
                 Yn           - function value at time tn
                 fct_RHS      - vectorised function that defines slope (derivative) at point (t, y)
                                = all RHS of the system of ODEs
                                
                 params       - py dictionary that includes step size 'h'
                                and all parameters needed in function:
                                fct_RHS
    :return:
                 a_fn1        - values of derivatives at current tn for all ODEs
    """
    h = params['h']
    # runge-kutta
    Kn1 = fct_RHS( tn, Yn,   params)
    Kn2 = fct_RHS( tn + .5*h, Yn + .5*h*Kn1, params)
    Kn3 = fct_RHS( tn + .5*h, Yn + .5*h*Kn2, params)
    Kn4 = fct_RHS( tn +    h, Yn +    h*Kn3, params)
    return (Kn1 + 2*Kn2 + 2*Kn3 + Kn4)/6

def lorenz(t, Yn, par):
    """
    describe the fourth order ODE for the lorenz system
    :ODE:
             dx/dt = sigma*(y - x)
             dy/dt = x*(rho - z) - y
             dz/dt = x*y - beta*z
    :input:
             t  - time vector
             Yn - displacement
    :return:
             dx_dt, dy_dt, dz_dt = lorenz equations
    """
    x, y, z = Yn[0], Yn[1], Yn[2]
    # lorenz system
    dx_dt = par['sigma']* (-x + y)
    dy_dt = (par['rho']*x) - y- (x*z)
    dz_dt = (-par['beta']*z) + (x*y)
    return np.array([dx_dt, dy_dt, dz_dt])

def num_sol(at, y0, par):
    """
    solve the numerical solution of fourth order ODE for the lorenz system
    :ODE:
             dx/dt = sigma*(y - x)
             dy/dt = x*(rho - z) - y
             dz/dt = x*y - beta*z
    :param y0:
             - IC
    :param at:
             - time vector
    :param par:
             - dictionary with fct parameters
    :return:
             a_x, a_y, a_z = 4th order runge-kutta
    """
    nSteps    = at.shape[0]
    # create vectors
    a_x = np.zeros( nSteps)
    a_y = np.zeros( nSteps)
    a_z = np.zeros( nSteps)
    # set initial conditions
    a_x[0] = y0[0]
    a_y[0] = y0[1]
    a_z[0] = y0[2]
    for i in range( nSteps-1):
        # slope at previous time step, i
        fn1, fn2, fn3 = runge_kutta_vec( at[i], np.array([a_x[i], a_y[i], a_z[i]]), lorenz, dPar)
        # integration step: runge-kutta
        a_x[i+1] = a_x[i] + fn1*dPar['h']
        a_y[i+1] = a_y[i] + fn2*dPar['h']
        a_z[i+1] = a_z[i] + fn3*dPar['h']
    return a_x, a_y, a_z


#=============================2=======================================
#                         Parameters
#=====================================================================
dPar = { # parameters
        'sigma' : 10,
        'rho'   : 28,
        'beta'  : 8./3.,
        # initial conditions
        'y01' : 1, 'y02' : 1, 'y03' : 1,
        # time stepping
        'h'      : 1e-2,
        'tStart' : 0,
        'tStop'  : 20*np.pi }

#=============================3=======================================
#                      Analytical Solution
#=====================================================================
a_t = np.arange(dPar['tStart'], dPar['tStop']+dPar['h'], dPar['h'])

#=============================4=======================================
#                       Numerical Solutions
#=====================================================================
ax_num, ay_num, az_num = num_sol( a_t, [dPar['y01'], dPar['y02'], dPar['y03']], dPar)

#=============================5=======================================
#                          Plotting
#=====================================================================
#---------- Lorenz Attractor ----------
plt.figure(1, figsize = (8, 6))
ax = plt.axes(projection='3d')
ax.set_xlabel('Rate of Convection')
ax.set_ylabel('Horizontal Temp Variation')
ax.set_zlabel('Vertical Temp Variation')
ax.plot3D(ax_num, ay_num, az_num, c='r')
plt.title('Lorenz Attractor')

#---------- dx/dt ----------
plt.figure(2)
ax1 = plt.subplot(111)
ax1.plot(a_t, ax_num,   'r-', lw = 1, alpha = .6, label = 'Numerical Displacement')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Displacement [mm]')
ax1.legend(loc = 'upper left')
plt.title('dx_dt')

#---------- dy/dt ----------
plt.figure(3)
ax2 = plt.subplot(111)
ax2.plot(a_t, ay_num,   'g-', lw = 1, alpha = .6, label = 'Numerical Displacement')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Displacement [mm]')
ax2.legend(loc = 'upper left')
plt.title('dy_dt')

#---------- dz/dt ----------
plt.figure(4)
ax3 = plt.subplot(111)
ax3.plot(a_t, az_num,   'b-', lw = 1, alpha = .6, label = 'Numerical Displacement')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Displacement [mm]')
ax3.legend(loc = 'upper left')
plt.title('dz_dt')

plt.show()


# In[ ]:




