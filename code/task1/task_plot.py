import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,16)
plt.rcParams.update({'font.size': 17})

def plotDescent(descent, max_iters):
    plt.figure('descent direction')
    plt.plot(np.arange(max_iters), descent[:max_iters],'g', linewidth=3)
    plt.xlabel('$k$')
    plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.yscale('log')
    plt.grid()
    nomeimg = "img/descent_direction.png"
    plt.savefig(nomeimg)
    plt.show(block=False)

def plotCost(JJ, max_iters):
    plt.figure('cost')
    plt.plot(np.arange(max_iters), JJ[:max_iters],'b', linewidth=3)
    plt.xlabel('$k$')
    plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.yscale('log')
    plt.grid()
    nomeimg = "img/cost.png"
    plt.savefig(nomeimg)
    plt.show(block=False)

def plotResult(ns, ni, TT, tf, max_iters, xx_star, xx_ref, xx_pl, uu_star, uu_ref, uu_pl):
    tt_hor = np.linspace(0,tf,TT)
    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    for s in range(ns):
        for j in range(0,max_iters):
            axs[s].plot(tt_hor, xx_pl[s,:,j], 'c--', linewidth=1)
            
        axs[s].plot(tt_hor, xx_ref[s,:], 'b--', linewidth=2)
        axs[s].plot(tt_hor, xx_star[s,:], 'm', linewidth=2)
        
        axs[s].grid()
        name = "$x_"+str(s)+"$"
        axs[s].set_ylabel(name)

    for i in range(ni):
        for j in range(0,max_iters):
            axs[i+ns].plot(tt_hor, uu_pl[i,:,j], 'c--', linewidth=1)

        if i==(ni-1):
            axs[i+ns].plot(tt_hor, uu_ref[i,:], 'b--', linewidth=2, label="Reference Trajectory")
            axs[i+ns].plot(tt_hor, uu_star[i,:],'m', linewidth=2, label="Optimal Trajectory")
        else:    
            axs[i+ns].plot(tt_hor, uu_ref[i,:], 'b--', linewidth=2)
            axs[i+ns].plot(tt_hor, uu_star[i,:],'m', linewidth=2)

        axs[i+ns].grid()
        name = "$u_"+str(i)+"$"
        axs[i+ns].set_ylabel(name)

    axs[ns+ni-1].set_xlabel('time')
    fig.legend(loc="upper left")

    nomeimg = "img/opt_trajectory.png"
    plt.savefig(nomeimg)
    plt.show()
