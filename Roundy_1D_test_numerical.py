#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Physics Thesis Code for Calculating fusion rate
# Author: Jacob van de Lindt
# July 17th, 2019

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[163]:


# Initialize Physical Dimensions of the Machine

R_cathode = 1
R_anode = 2
R_mag = 1.25

R_coil = .04
R_alloy = .01
R_steel = .01
R_stalk = .06

# Specify Physical constants

mu_o = 1
epsi_o = 1
k = 1
e = 1
m_dueterium  = 2.014
m_neutron = 1.008664
m_proton = 1.007276
m_electron = 0.00054858

#Specify Tunable metrics:

I = 20 # [A]
V_grid = 10000 # [V]





# In[134]:


# Specify Descretization for Simulation

res = .01

dx = res
dy = res
dz = res
dt = .1
num_time_steps = 100 * dt

ds = np.sqrt(dx**2 + dy**2 + dz**2)

# Define Needed Field Functions

def E_field_x(x, y, z):
    Q_enc = 100
    if ((x < -R_cathode) or (x > R_cathode)):
        return Q_enc / (4 * np.pi * epsi_o * x**2)
    else:
        return 0
#     x_low = np.where(x <= -R_cathode)
#     out_low = Q_enc / (4 * np.pi * epsi_o * x[x_low]**2)
#     out_mid = np.zeros(x[np.where(x > -R_cathode and x < R_cathode)].shape[0])
#     x_high = np.where(x >= R_cathode)
#     out_high = Q_enc / (4 * np.pi * epsi_o * x_high**2)
#     return [out_low, out_mid, out_high]

v_E_field_x = np.vectorize(E_field_x)

def E_field_y(x, y, z):
    return 0
def E_field_z(x, y, z):
    return 0

def B_field_x(x, y, z):
    return 0
def B_field_y(x, y, z):
    return 0
def B_field_z(x, y, z):
    return 0

def P_fus_scalar_field(x, y, z):
    return 0




# In[257]:


# Define randome initialization function
def random_ion_initializer():
    return x, y, z

# Define the ion cloud density initial guess. Assume that it is a half sine wave initially.
def density_distrobution_initializer(x, period, Amplitude):
    return Amplitude*np.cos((np.pi/2*x) / period) * e

plt.plot(rx_vec, density_distrobution_initializer(rx_vec, np.max(rx_vec), 10))


# In[258]:


# Initialize the dimension vector of the Machine:
r_res = .005
dr = r_res
rx_vec = np.arange(0, R_anode, r_res)

dt = dr*2
time_steps = 100

#initialize the time vector
t_vec = np.arange(0, time_steps*dt, dt)

#Initialize the particle position vector with zeros up to the end of the time steps
prx_vec = np.zeros(t_vec.shape[0])

#Initialize the particle velocity vector with zeros up to the end of the time steps
pveloc_vec = np.zeros(t_vec.shape[0])

# Initialize the potential vector with zeros
Psi_vec_beta = np.zeros(rx_vec.shape[0])

# Initialize the E-field vector with zeros

E_vec = np.zeros(rx_vec.shape[0])
# Begin calculating the potential and Electric field
Amplitude = 10000
density_vec = density_distrobution_initializer(rx_vec, rx_vec.shape[0], Amplitude)

#initial condition
Psi_anode = 0
#tunable second initial condition
Psi_anode_second = 0
#target second initial condition
Psi_cathode = 100 #V



# In[247]:


# forward differencing function, had difficulties du to random choice of initial conditions

def make_Psi_forward(Psi_init, rho_vec):
    Psi = Psi_init
    Psi[0] = Psi_anode
    Psi[1] = Psi_anode_second

    for m in range(2, rx_vec.shape[0]):
        r_mm = np.abs(rx_vec[m-1])
        top_first = (2*dr + 2*r_mm) * Psi[m-1]
        top_second = r_mm * Psi[m-2]
        top_third = ( r_mm * rho_vec[m-1] * dr**2 ) / epsi_o
        bottom = 2 * dr + r_mm
        Psi[m] = ( top_first - top_second - top_third ) / ( bottom )

    return Psi
Psi_forward = make_Psi_forward(Psi_vec_beta, density_vec)
plt.plot(rx_vec, Psi_forward)


# In[261]:


# finite difference matrix implimentation

def make_Psi_global(position_vec, rho_vec, V_grid, cathod_pos_from_left):
    n = position_vec.shape[0]
    node_matrix = np.zeros((n, n))

    m_cathode = round(cathod_pos_from_left / dr)
    # Add Dirchele boundery conditions
    node_matrix[0, n-1] = 1
    node_matrix[1, m_cathode] = 1

    for ii in range(2, n):
        m = ii - 1
        r_m = np.abs(position_vec[m])
        node_matrix[ii, ii - 2] = r_m
        node_matrix[ii, ii - 1] = -( 2*dr + 2*r_m  )
        node_matrix[ii, ii] = 2*dr + r_m

    # now create the b vector in the expression Ax = b

    b_vec = np.zeros((n, 1))

    for jj in range(2, n):
        m = ii - 1
        r_m = np.abs(position_vec[m])
        b_vec[jj] = - (r_m*rho_vec[m]*dr**2) / epsi_o
    b_vec[0] = 0
    b_vec[1] = V_grid

    mat_inverse = np.linalg.inv(node_matrix)
    return np.matmul(mat_inverse, b_vec)




Psi_no_density = make_Psi_global(rx_vec, density_vec, V_grid, 1)


plt.plot(rx_vec, Psi_no_density)
#plt.plot(rx_vec, density_vec)
plt.xlabel('r [m]')
plt.ylabel('Potential [V]')
plt.show()

def check_behavior(x):
    return 15500 / x

plt.plot(rx_vec, check_behavior(rx_vec), 'r')

#Psi_small_density = make_Psi_global(rx_vec, density_vec, V_grid, 1)

#plt.plot(rx_vec, Psi_small_density, 'g')


# In[ ]:


#create the E-field from the potential
def make_E_field(Psi):
    n = Psi.shape[0]
    # Warning not yet ready
