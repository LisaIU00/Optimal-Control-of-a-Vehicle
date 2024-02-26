import numpy as np
import matplotlib.pyplot as plt
import signal

import dynamics as dyn
import cost as cst
import ref_curve as rc
import newton_method as nm
import armijo as arm
import task_plot as tp


signal.signal(signal.SIGINT, signal.SIG_DFL)

#######################################
# Algorithm parameters
#######################################

max_iters = 10
stepsize_0 = 1
term_cond = 1e-7

# ARMIJO PARAMETERS
cc = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

visu_ref_curve = True
visu_armijo = True

#######################################
# Trajectory parameters
#######################################

tf = 10 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns   #number of states
ni = dyn.ni   #number of inputs

TT = int(tf/dt) # discrete-time samples

######################################
# Reference curve
######################################

ref_V_in = 10
ref_V_fin = 20

xx_ref, uu_ref = rc.ref_curve(ref_V_in, ref_V_fin, TT)
x0 = xx_ref[:,0] 

######################################
# Plot Reference curve
######################################

if visu_ref_curve:
    rc.plotRefCurve(xx_ref, tf, TT)

######################################
# Initial guess
######################################

uu_init = np.zeros((ni, TT))
xx_init = np.zeros((ns, TT))

xx_init[3] = np.ones((1, TT))*ref_V_in/2
for i in range(TT-1):
  xx_init[:,i+1] = dyn.dynamics(xx_init[:,i], uu_init[:,0])[0]

######################################
# Arrays to store data
######################################
  
xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

xx_pl = np.zeros((ns, TT, max_iters)) #to plot few intermediate state solution
uu_pl = np.zeros((ni, TT, max_iters)) #to plot few intermediate input solution

JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction
descent_arm = np.zeros(max_iters) # collect descent direction

######################################
# Main
######################################

print('-*-*-*-*-*-')

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init
print("V INIT : [ ", xx[3,:,0]," ]")


for kk in range(max_iters-1):

    AA = np.zeros((ns,ns,TT)) 
    BB = np.zeros((ns,ni,TT)) 
    
    for tt in range(TT):
        fx,fu = dyn.dynamics(xx[:,tt, kk], uu[:,tt,kk])[1:]
        AA[:,:,tt]=fx.T
        BB[:,:,tt]=fu.T

    JJ[kk] = 0

    # calculate cost
    for tt in range(TT-1):
        temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ[kk] += temp_cost
  
    temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1], AA[:,:,-1], BB[:,:,-1])[0]
    JJ[kk] += temp_cost

    ##################################
    # Descent direction calculation
    ##################################

    KK, sigma, deltau, dJ = nm.newton_method(ns, ni, TT, kk, xx[:,:,kk], uu[:,:,kk], xx_ref, uu_ref)

    
    for tt in reversed(range(TT)):
        descent[kk] += deltau[:,tt].T@deltau[:,tt]
        descent_arm[kk] += dJ[:,tt].T@deltau[:,tt]
    
    ##################################
    # Stepsize selection - ARMIJO
    ##################################

    stepsizes, costs_armijo, stepsize = arm.armijo(ns, ni, TT, kk, stepsize_0, armijo_maxiters, x0, uu[:,:,kk],  xx[:,:,kk], deltau[:,:], xx_ref, uu_ref, JJ[kk], descent_arm[kk], beta, cc, AA[:,:,-1], BB[:,:,-1], sigma, KK)

    ############################
    # Armijo plot
    ############################
        
    if visu_armijo:
        arm.plotArmijo(ns, ni, TT, kk, x0, stepsize_0, deltau[:,:], uu[:,:,kk], xx[:,:,kk],xx_ref, uu_ref, JJ[kk], descent_arm[kk], cc, stepsizes, costs_armijo, AA[:,:,-1], BB[:,:,-1], sigma, KK)

    ##############################
    # Update the current solution
    ##############################

    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))

    xx_temp[:,0] = x0

    for tt in range(TT-1):

        uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + stepsize*sigma[:,tt] #closed loop update #u in kk+1
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0] #xx in kk+1        

    xx[:,:,kk+1] = xx_temp
    uu[:,:,kk+1] = uu_temp
    
    xx_pl[:,:,kk] = xx_temp  
    uu_pl[:,:,kk] = uu_temp

    ############################
    # Termination condition
    ############################

    print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

    if descent[kk] <= term_cond:

        max_iters = kk+1

        break

############################
# Assign optimal solution
############################

xx_star = xx[:,:,max_iters-1]

uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

np.save('x_opt_task1',xx_star)
np.save('u_opt_task1',uu_star)

############################
# Plots
############################

# cost and descent
tp.plotDescent(descent, max_iters)

tp.plotCost(JJ, max_iters)

# plot optimal trajectory
tp.plotResult(ns, ni, TT, tf, max_iters, xx_star, xx_ref, xx_pl, uu_star, uu_ref, uu_pl)






    
    

    

        



        












