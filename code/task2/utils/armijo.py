import numpy as np
import matplotlib.pyplot as plt
import utils.dynamics as dyn
import cost as cst


def armijo(ns, ni, TT, kk, stepsize_0, armijo_maxiters, x0, uu, xx,  deltau, xx_ref, uu_ref, JJ, descent_arm, beta, cc, AAT, BBT, sigma, KK):
    stepsizes = []  # list of stepsizes
    costs_armijo = []

    stepsize = stepsize_0

    for ii in range(armijo_maxiters):

        # temp solution update
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = x0
        
        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt]) + stepsize*sigma[:,tt]
            xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]
            
        # temp cost calculation
        JJ_temp = 0

        for tt in range(TT-1):
            temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ_temp += temp_cost
                
        temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1], AAT, BBT)[0]
        
        JJ_temp += temp_cost
        stepsizes.append(stepsize)      # save the stepsize

        costs_armijo.append(np.min([JJ_temp, 100*JJ]))    # save the cost associated to the stepsize

        print("JJ_temp: ", JJ_temp, "\tJJ[", kk, "]: ", JJ)

        if JJ_temp > JJ  + cc*stepsize*descent_arm:
            # update the stepsize
            stepsize = beta*stepsize
        
        else:
            print('Armijo stepsize = {:.3e}'.format(stepsize))
            break
    
    
    return stepsizes, costs_armijo, stepsize


def plotArmijo(ns, ni, TT, kk, x0, stepsize_0, deltau, uu, xx, xx_ref, uu_ref, JJ, descent_arm, cc, stepsizes, costs_armijo, AAT, BBT, sigma, KK):
    plt.rcParams["figure.figsize"] = (20,16)
    plt.rcParams.update({'font.size': 17})
    
    
    steps = np.linspace(0,stepsize_0,int(2e1))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

        step = steps[ii]

        # temp solution update
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = x0

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt]) + step*sigma[:,tt]
            xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

        # temp cost calculation
        JJ_temp = 0

        for tt in range(TT-1):
            temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ_temp += temp_cost

        temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1], AAT, BBT)[0]
        JJ_temp += temp_cost

        costs[ii] = np.min([JJ_temp, 100*JJ])

    plt.figure(1)
    plt.clf()

    plt.plot(steps, costs, color='b', linewidth=3, label='Cost')
    plt.plot(steps, JJ + descent_arm*steps, color='r', linewidth=3, label='Tangent line')
    plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linewidth=3, linestyle='dashed', label='Armijo line')

    plt.scatter(stepsizes, costs_armijo, marker='*', label='Armijo Costs') 

    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.draw()

    nomeimg = "img/armijo_plot_"+str(kk)+".png"
    plt.savefig(nomeimg)
    plt.show()