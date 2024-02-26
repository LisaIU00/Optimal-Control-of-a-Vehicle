import numpy as np
import matplotlib.pyplot as plt

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (20,16)
plt.rcParams.update({'font.size': 17})

def plotError(ns, ni, err_x, err_u, tt_hor):
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
    plt.show()


def plotResult(ns, ni, tt_hor, xx_opt, xx_result, uu_opt, uu_result):
    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    for s in range(ns):
        axs[s].plot(tt_hor, xx_opt[s,:], 'g--', linewidth=2)
        axs[s].plot(tt_hor, xx_result[s,:], 'm', linewidth=2)
        
        axs[s].grid()
        name = "$x_"+str(s)+"$"
        axs[s].set_ylabel(name)

    for i in range(ni):
        if i==0:
            axs[i+ns].plot(tt_hor, uu_opt[i,:], 'g--', linewidth=2, label="Reference Trajectory")
            axs[i+ns].plot(tt_hor, uu_result[i,:],'m', linewidth=2, label="Optimal Trajectory")
        else:
            axs[i+ns].plot(tt_hor, uu_opt[i,:], 'g--', linewidth=2)
            axs[i+ns].plot(tt_hor, uu_result[i,:],'m', linewidth=2)
        
        axs[i+ns].grid()
        name = "$u_"+str(i)+"$"
        axs[i+ns].set_ylabel(name)

    fig.legend(loc="upper left")
    axs[ns+ni-1].set_xlabel('time')
    nomeimg = "img/TASK3_opt_trajectory.png"
    plt.savefig(nomeimg)
    plt.show()