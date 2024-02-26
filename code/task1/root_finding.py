import numpy as np
from scipy.optimize import fsolve

# Dynamics parameters
mm = 1480 #Kg
Iz = 1950 #Kgm^2
aa = 1.421 #m
bb = 1.029 #m
mu = 1
gg = 9.81 #m/s^2

'''
ASSOCIAZIONI:
xx[0]=x; xx[1]=y; xx[2]=psi; xx[3]=V; xx[4]=beta; xx[5]=psipunto;
uu[0]=delta; uu[1]=Fx
'''

def equation(xy,xx):
    uu = np.zeros(2)
    uu = xy

    #Fz
    Fz = np.zeros(2)
    Fz[0]=(mm*gg*bb)/(aa+bb) #F_zf
    Fz[1]=(mm*gg*aa)/(aa+bb) #F_zr

    #beta
    beta = np.zeros(2)
    beta[0] = uu[0] - (xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])) #beta_f
    beta[1] = - (xx[3]*np.sin(xx[4]) - bb*xx[5])/(xx[3]*np.cos(xx[4])) #beta_r

    #Fy
    Fy = np.zeros(2)
    Fy[0] = mu*Fz[0]*beta[0] #F_yf
    Fy[1] = mu*Fz[1]*beta[1] #F_yr

    #functions
    f1 = (1/mm) * (Fy[1]*np.sin(xx[4]) + uu[1]*np.cos(xx[4]-uu[0]) + Fy[0]*np.sin(xx[4]-uu[0]))
    f2 = ((1/(mm*xx[3])) * (Fy[1]*np.cos(xx[4]) - uu[1]*np.sin(xx[4]-uu[0]) + Fy[0]*np.cos(xx[4]-uu[0]))) - xx[5]
    f3 = (1/Iz) * (((uu[1]*np.sin(uu[0]) + Fy[0]*np.cos(uu[0])) * aa) - Fy[1]*bb)

    return f1,f2

def root_finding(xx):
    result_u0 = []
    result_u1 = []
    for i in range (0, 1):
        for j in range(0, 10):
            temp_u0, temp_u1 = fsolve(equation, [i,j], (xx, ))
            result_u0.append(temp_u0)
            result_u1.append(temp_u1)

    return result_u0, result_u1









