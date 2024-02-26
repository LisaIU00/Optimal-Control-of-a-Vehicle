import numpy as np
import dynamics as dyn


def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT,xx, xx_ref, qqin = None, rrin = None, qqfin = None):

  """
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - xx (ns x TTxkk)
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
  
  ni = dyn.ni
  ns = dyn.ns

  KK = np.zeros((ni, ns, TT))
  sigma = np.zeros((ni, TT))
  PP = np.zeros((ns, ns, TT))
  pp = np.zeros((ns, TT))

  QQ = QQin
  RR = RRin
  SS = SSin
  QQf = QQfin
  
  qq = qqin
  rr = rrin

  qqf = qqfin

  AA = AAin
  BB = BBin

  uu = np.zeros((ni, TT))

  
  PP[:,:,-1] = QQf
  pp[:,-1] = qqf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:, tt+1][:,None]

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp
    
    PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
    ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze()


  # Evaluate KK
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1][:,None]

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp

    KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
    sigma_t = -MMt_inv@mmt

    sigma[:,tt] = sigma_t.squeeze()


  deltax = np.zeros((ns,TT))

  for tt in range(TT - 1):
    
    uu[:, tt] = KK[:,:,tt]@(deltax[:,tt]) + sigma[:,tt]  
    deltax[:,tt+1] = AA[:,:,tt]@deltax[:,tt] + BB[:,:,tt]@uu[:, tt] 
    

  uuout = uu

  return KK, sigma, PP, uuout
    