import numpy as np
import matplotlib.pyplot as plt

import dynamics as dyn
import task_plot as tp

import solver_t3 as lqr
import scipy.linalg as scp

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 17})

#######################################
# Parameters
#######################################

tf = 20 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns   #number of states
ni = dyn.ni   #number of inputs

TT = int(tf/dt) # discrete-time samples

#######################################
# Dynamics
#######################################

xx_opt = np.load('x_opt_task2.npy') 
uu_opt = np.load('u_opt_task2.npy') 

Qcost = np.diag([100, 1000, 10, 100, 100, 10])
Rcost = np.diag([10000, 0.0001]) 


AA = np.zeros((ns,ns,TT)) 
BB = np.zeros((ns,ni,TT)) 
QQ = np.zeros((ns,ns,TT)) 
RR = np.zeros((ni,ni,TT)) 
QQf = np.zeros((ns,ns))


for tt in range(TT):
    fx,fu = dyn.dynamics(xx_opt[:,tt], uu_opt[:,tt])[1:]
    AA[:,:,tt]=fx.T
    BB[:,:,tt]=fu.T
    QQ[:,:,tt] = Qcost
    RR[:,:,tt] = Rcost
QQf = scp.solve_discrete_are(AA[:,:,-1],BB[:,:,-1],Qcost,Rcost)

KK,PP = lqr.lti_LQR(AA,BB,QQ,RR,QQf,TT)


xx_result = np.zeros((ns,TT))
x0 = xx_opt[:,0] + [0.3, 0, 0, 8, 0, 0]
#x0 = [0.3, 0, 0, 8, 0, 0]
xx_result[:,0] = x0
uu_result = np.zeros((ni,TT))
err_x = np.zeros((ns,TT))
err_u = np.zeros((ni,TT))

for tt in range(TT-1):
    uu_result[:,tt] = uu_opt[:,tt] + KK[:,:,tt]@(xx_result[:,tt]-xx_opt[:,tt])
    xx_result[:,tt+1] = dyn.dynamics(xx_result[:,tt], uu_result[:,tt])[0]
    #xx_result[:,tt+1] = AA[:,:,tt]@xx_result[:,tt] + BB[:,:,tt]@uu_result[:,tt]
    
    err_x[:,tt+1] = xx_result[:,tt+1]-xx_opt[:,tt+1]
    err_u[:,tt] = uu_result[:,tt]-uu_opt[:,tt]

print("***TASK2***")
print("Y -> ", xx_opt[1,:])
print("V -> ", xx_opt[3,:])
print("----TASK3----")
print("Y -> ", xx_result[1,:])
print("V -> ", xx_result[3,:])

np.save('x_opt_task3',xx_result)
np.save('u_opt_task3',xx_result)


#######################################
# Plots
#######################################

tt_hor = np.linspace(0,tf,TT)

#plot system trajectory and desired trajectory
tp.plotResult(ns, ni, tt_hor, xx_opt, xx_result, uu_opt, uu_result)

#plot error for different initial condition
tp.plotError(ns, ni, err_x, err_u, tt_hor)



