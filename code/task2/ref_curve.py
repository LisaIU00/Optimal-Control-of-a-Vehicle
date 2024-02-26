import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import utils.dynamics as dyn


def ref_curve(ns, ni, TT, tf, dt, ref_V_in, ref_y_in, ref_y_fin):
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    a = 1
    b = tf/2 + dt
    
    # initial and final conditions
    x0 = ref_y_in*np.ones((1,))
    xf = ref_y_fin*np.ones((1,))

    # sampling time
    t_samp_temp = np.linspace(dt, 2*b - dt, TT)

    def bell_function(t, x, a): 
        tau = t - b
        exp_arg = -(2*b**2)*(tau**2 + b**2)/(tau**2 - b**2)**2
        beta = a * np.exp(exp_arg)
        return beta*np.ones((1,))

    sigma_temp = solve_ivp(bell_function, (dt, 2*b - dt), np.zeros((1,)), t_eval = t_samp_temp, args = (1,), method = 'Radau')
    norm_factor = sigma_temp.y[0][-1] # normalization factor
    
    sigma = solve_ivp(bell_function, (dt, 2*b - dt), x0, t_eval = t_samp_temp, args = ((xf-x0)/norm_factor,), method = 'Radau').y[0]

    t_samp = t_samp_temp - dt

    y = sigma

    xx_ref[1,:] = y
    xx_ref[3,:] = ref_V_in

    for tt in range(TT-1):
        xx_ref[0,tt+1] = dyn.dynamics(xx_ref[:,tt],uu_ref[:,tt])[0][0]
    
    return xx_ref, uu_ref


def plotRefCurve(xx_ref, tf, TT, ns):
    plt.rcParams["figure.figsize"] = (20,16)
    plt.rcParams.update({'font.size': 18})

    tt_hor = np.linspace(0,tf,TT)
    fig, axs = plt.subplots(ns, 1, sharex='all')

    axs[0].plot(tt_hor, xx_ref[0,:], 'g', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x$')

    axs[1].plot(tt_hor, xx_ref[1,:],'g', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$y$')

    axs[2].plot(tt_hor, xx_ref[2,:], 'g', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$psi$')

    axs[3].plot(tt_hor, xx_ref[3,:],'g', linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel('$V$')

    axs[4].plot(tt_hor, xx_ref[4,:], 'g', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$beta$')

    axs[5].plot(tt_hor, xx_ref[5,:],'g', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$ypsi.punto$')

    axs[5].set_xlabel('time')
    
    fig.suptitle("Reference curve")
    plt.savefig("img/ref_curve.png")