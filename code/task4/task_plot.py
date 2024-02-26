import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,16)
plt.rcParams.update({'font.size': 12})

def plotResult(ns, ni, TT, tf, xx_star, xx_ref, uu_star, uu_ref):
    tt_hor = np.linspace(0,tf,TT)

    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    for s in range(ns):
        
        axs[s].plot(tt_hor, xx_ref[s,:], 'g--', linewidth=2)
        axs[s].plot(tt_hor, xx_star[s,:], 'm', linewidth=2)
        
        axs[s].grid()
        name = "$x_"+str(s)+"$"
        axs[s].set_ylabel(name)

    for i in range(ni):
        if i==0:
            axs[i+ns].plot(tt_hor, uu_ref[i,:], 'g--', linewidth=2, label="Reference Trajectory")
            axs[i+ns].plot(tt_hor, uu_star[i,:],'m', linewidth=2, label="Optimal Trajectory")
        else:
            axs[i+ns].plot(tt_hor, uu_ref[i,:], 'g--', linewidth=2)
            axs[i+ns].plot(tt_hor, uu_star[i,:],'m', linewidth=2)

        axs[i+ns].grid()
        name = "$u_"+str(i)+"$"
        axs[i+ns].set_ylabel(name)
    
    fig.legend(loc="upper left")
    axs[ns+ni-1].set_xlabel('time')
    
    nomeimg = "img/opt_trajectory.png"
    plt.savefig(nomeimg)
    plt.show()

    plt.figure('Optimal Trajectory')
    plt.plot(xx_star[0,:], xx_star[1,:],'m', linewidth=2)
    plt.plot(xx_ref[0,:], xx_ref[1,:],'g--', linewidth=2)
    plt.xlabel('$x0$')
    plt.ylabel('$x1$')
    plt.grid()
    nomeimg = "img/XYvopt_trajectory.png"
    plt.savefig(nomeimg)
    plt.show(block=False)

def plotError(ns, ni, tf, TT, err_x, err_u):
    tt_hor = np.linspace(0,tf,TT)
    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    for s in range(ns):
        axs[s].plot(tt_hor, err_x[s,:], 'm', linewidth=2)
        axs[s].grid()
        name = "$x_"+str(s)+"$"
        axs[s].set_ylabel(name)

    for i in range(ni):
        axs[i+ns].plot(tt_hor, err_u[i,:],'m', linewidth=2)
        
        axs[i+ns].grid()
        name = "$u_"+str(i)+"$"
        axs[i+ns].set_ylabel(name)

    axs[ns+ni-1].set_xlabel('time')
    nomeimg = "img/error.png"
    plt.savefig(nomeimg)

