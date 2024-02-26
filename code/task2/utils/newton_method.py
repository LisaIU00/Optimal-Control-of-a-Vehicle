import numpy as np

import utils.dynamics as dyn
import cost as cst
import utils.solver as slv


def newton_method(ns, ni, TT, kk, xx, uu, xx_ref, uu_ref):
    AAt = np.zeros((ns,ns,TT)) 
    BBt = np.zeros((ns,ni,TT)) 
    aqt = np.zeros((ns, TT)) 
    brt = np.zeros((ni, TT)) 
    aqT = np.zeros((ns)) 
    QQt = np.zeros((ns,ns,TT)) 
    RRt = np.zeros((ni,ni,TT)) 
    SSt = np.zeros((ni,ns, TT)) 
    QQT = np.zeros((ns,ns)) 

    for tt in range(TT):
        fx, fu = dyn.dynamics(xx[:,tt], uu[:,tt])[1:] 
        AAt[:,:,tt] = fx.T 
        BBt[:,:,tt] = fu.T

    aqT, QQT = cst.termcost(xx[:,-1], xx_ref[:,-1], AAt[:,:,-1], BBt[:,:,-1])[1:] 

    dJ = np.zeros((ni,TT))     # DJ - gradient of J wrt u
    lmbd = np.zeros((ns, TT))  # lambdas - costate seq.

    lmbd[:,-1] = aqT

    for tt in reversed(range(TT-1)):  # integration backward in time

        ll, at, bt, lxx, luu, lxu = cst.stagecost(xx[:,tt], uu[:,tt], xx_ref[:,tt], uu_ref[:,tt])
        
        #Regularization
        QQt[:,:,tt] = lxx 
        RRt[:,:,tt] = luu
        SSt[:,:,tt] = lxu
        aqt[:, tt] = at.squeeze()
        brt[:, tt] = bt.squeeze()

        lmbd_temp = AAt[:,:,tt].T@lmbd[:,tt+1][:,None] + at      # costate equation
        lmbd[:,tt] = lmbd_temp.squeeze()

        dJ_temp = BBt[:,:,tt].T@lmbd[:,tt+1][:,None] + bt # gradient of J wrt u
        dJ[:,tt] = dJ_temp.squeeze()        
        
    KK,sigma, PP, deltau = slv.ltv_LQR(AAt,BBt,QQt,RRt,SSt,QQT, TT, xx, xx_ref, aqt, brt, aqT)

    return KK, sigma, deltau, dJ