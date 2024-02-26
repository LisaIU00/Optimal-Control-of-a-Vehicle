import numpy as np
import matplotlib.pyplot as plt

import dynamics as dyn

ni = dyn.ni
ns = dyn.ns
dt = dyn.dt

def ref_curve(ref_V_in, ref_V_fin, TT):
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    xx_ref[3,:int(TT/2)] = np.ones((1,int(TT/2)))*ref_V_in
    for i in range(int(TT/2)):
        xx_ref[:,i+1] = dyn.dynamics(xx_ref[:,i], uu_ref[:,0])[0]

    xx_ref[3,int(TT/2):] = np.ones((1,int(TT/2)))*ref_V_fin
    for i in range(int(TT/2),TT-1,1):
        xx_ref[:,i+1] = dyn.dynamics(xx_ref[:,i], uu_ref[:,0])[0]

    return xx_ref, uu_ref


def plotRefCurve(xx_ref, tf, TT):
    plt.rcParams["figure.figsize"] = (20,16)
    plt.rcParams.update({'font.size': 17})

    tt_hor = np.linspace(0,tf,TT)
    fig, axs = plt.subplots(ns, 1, sharex='all')

    axs[0].plot(tt_hor, xx_ref[0,:], 'g', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x0$')

    axs[1].plot(tt_hor, xx_ref[1,:],'g', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$x1$')

    axs[2].plot(tt_hor, xx_ref[2,:], 'g', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$x2$')

    axs[3].plot(tt_hor, xx_ref[3,:],'g', linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel('$x3$')

    axs[4].plot(tt_hor, xx_ref[4,:], 'g', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$x4$')

    axs[5].plot(tt_hor, xx_ref[5,:],'g', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$x5$')

    axs[5].set_xlabel('time')
    
    plt.savefig("img/ref_curve.png")