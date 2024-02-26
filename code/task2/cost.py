import numpy as np
import scipy.linalg as scp
import utils.dynamics as dyn
                     
ns = dyn.ns
ni = dyn.ni

QQt = np.diag([100, 1000, 10, 100, 100, 10])
RRt = np.diag([10000, 0.0001])


def stagecost(xx,uu, xx_ref, uu_ref):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - uu \in \R^1 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  #dl
  lx = QQt@(xx - xx_ref)
  lu = RRt@(uu - uu_ref)

  #ddl
  lxx = QQt
  luu = RRt
  lxu = np.zeros((ni,ns)) 

  return ll.squeeze(), lx, lu, lxx, luu, lxu

def termcost(xx,xx_ref, AA, BB):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """
  #calculate QQT solving Riccati equation 
  QQT = scp.solve_discrete_are(AA,BB,QQt,RRt)

  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)

  #dlT
  lTx = QQT@(xx - xx_ref)

  #ddlT
  lTxx = QQT

  return llT.squeeze(), lTx.squeeze(), lTxx
