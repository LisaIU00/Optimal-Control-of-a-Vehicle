import numpy as np
import matplotlib.pyplot as plt


import dynamics as dyn
import task_plot as tp
import solver as slv

import scipy.linalg as scp

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 12})

#######################################
# Parameters
#######################################

tf = 20 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns   #number of states
ni = dyn.ni   #number of inputs

T_pred = 50 #MPC Prediction horizon
TT = int(tf/dt)-T_pred # discrete-time samples

#######################################
# Dynamics
#######################################

xx_opt = np.load('x_opt_task2.npy') 
uu_opt = np.load('u_opt_task2.npy') 
print(xx_opt.shape)

Qcost = np.diag([100, 1000, 10, 100, 100, 10])
Rcost = np.diag([10000, 0.0001]) 

AA = np.zeros((ns,ns,TT+T_pred)) 
BB = np.zeros((ns,ni,TT+T_pred)) 
QQ = np.zeros((ns,ns,TT+T_pred)) 
RR = np.zeros((ni,ni,TT+T_pred)) 
QQf = np.zeros((ns,ns))

for tt in range(TT+T_pred):
    fx,fu = dyn.dynamics(xx_opt[:,tt], uu_opt[:,tt])[1:]
    AA[:,:,tt]=fx.T
    BB[:,:,tt]=fu.T
    QQ[:,:,tt] = Qcost
    RR[:,:,tt] = Rcost

QQf = scp.solve_discrete_are(AA[:,:,-1],BB[:,:,-1],Qcost,Rcost)

#############################
# Model Predictive Control
#############################
err_x = np.zeros((ns,TT))
err_u = np.zeros((ni,TT))

xx_real_mpc = np.zeros((ns,TT))
uu_real_mpc = np.zeros((ni,TT))

xx0 = xx_opt[:,0]

xx_real_mpc[:,0] = xx0
xx_real_mpc[0,0]=-30 
xx_real_mpc[3,0]=8 

for tt in range(TT-1):    

    xx_t_mpc = xx_real_mpc[:,tt] 
    uu_real_mpc[:,tt] = slv.linear_mpc(ns,ni,AA, BB, QQ[:,:,tt], RR[:,:,tt], QQf, xx_t_mpc, tt, T_pred, xx_opt, uu_opt)[0]

    xx_real_mpc[:,tt+1] = dyn.dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt])[0]

    err_x[:,tt+1] = xx_real_mpc[:,tt+1]-xx_opt[:,tt+1]
    err_u[:,tt] = uu_real_mpc[:,tt]-uu_opt[:,tt]
    
    perc = (tt*100)/TT
    if perc%5==0:
        print(perc, "% :")
        print("consider range for this equation: [",tt," ; ",T_pred+tt,"]")
        print("UU_real_mpc of tt: ",uu_real_mpc[:,tt])
        print("XX_real_mpc of tt+1: ",xx_real_mpc[:,tt+1])

#plot result
tp.plotResult(ns, ni, TT, tf,  xx_real_mpc, xx_opt[:,0:TT],  uu_real_mpc, uu_opt[:,0:TT])

#plot error
tp.plotError(ns, ni, tf, TT, err_x, err_u)
