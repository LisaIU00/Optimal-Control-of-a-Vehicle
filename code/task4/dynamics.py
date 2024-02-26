import numpy as np

ns = 6 #number of states
ni = 2 #number of inputs

dt = 1e-3 # discretization stepsize - Forward Euler

# Dynamics parameters

mm = 1480 #Kg
Iz = 1950 #Kgm^2
aa = 1.421 #m
bb = 1.029 #m
mu = 1
gg = 9.81 #m/s^2


def dynamics(xx,uu):
  
  """
  ASSOCIAZIONI:
  xx[0]=x; xx[1]=y; xx[2]=psi; xx[3]=V; xx[4]=beta; xx[5]=psipunto;
  uu[0]=delta; uu[1]=Fx
  
  """
  
  xxp = np.zeros((ns))

  #Fz
  Fz = np.zeros(2)
  Fz[0]=(mm*gg*bb)/(aa+bb) #F_zf
  Fz[1]=(mm*gg*aa)/(aa+bb) #F_zr

  #beta
  beta = np.zeros(2)
  beta[0] = uu[0] - (xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])) #beta_f
  beta[1] = - (xx[3]*np.sin(xx[4]) - bb*xx[5])/(xx[3]*np.cos(xx[4])) #beta_r

  #Fy
  Fy = np.zeros(2)
  Fy[0] = mu*Fz[0]*beta[0] #F_yf
  Fy[1] = mu*Fz[1]*beta[1] #F_yr

  xxp[0] = xx[0] + dt * (xx[3]*np.cos(xx[4])*np.cos(xx[2]) - xx[3]*np.sin(xx[4])*np.sin(xx[2]))
  xxp[1] = xx[1] + dt * (xx[3]*np.cos(xx[4])*np.sin(xx[2]) + xx[3]*np.sin(xx[4])*np.cos(xx[2]))
  xxp[2] = xx[5]
  xxp[3] = xx[3] + dt * (1/mm) * (Fy[1]*np.sin(xx[4]) + uu[1]*np.cos(xx[4]-uu[0]) + Fy[0]*np.sin(xx[4]-uu[0]))
  xxp[4] = xx[4] + dt * ((1/(mm*xx[3])) * (Fy[1]*np.cos(xx[4]) - uu[1]*np.sin(xx[4]-uu[0]) + Fy[0]*np.cos(xx[4]-uu[0])) - xx[5])
  xxp[5] = xx[5] + dt * (1/Iz) * (((uu[1]*np.sin(uu[0]) + Fy[0]*np.cos(uu[0])) * aa) - Fy[1]*bb)

  # Gradient
  fx = np.zeros((ns, ns))
  fu = np.zeros((ni, ns))

  #df1
  fx[0,0] = 1
  fx[1,0] = 0
  fx[2,0] = -dt*xx[3]*np.sin(xx[2])*np.cos(xx[4]) - dt*xx[3]*np.sin(xx[4])*np.cos(xx[2])
  fx[3,0] = -dt*np.sin(xx[2])*np.sin(xx[4]) + dt*np.cos(xx[2])*np.cos(xx[4])
  fx[4,0] = -dt*xx[3]*np.sin(xx[2])*np.cos(xx[4]) - dt*xx[3]*np.sin(xx[4])*np.cos(xx[2])
  fx[5,0] = 0

  fu[0,0] = 0
  fu[1,0] = 0

  #df2
  fx[0,1] = 0
  fx[1,1] = 1
  fx[2,1] = -dt*xx[3]*np.sin(xx[2])*np.sin(xx[4]) + dt*xx[3]*np.cos(xx[2])*np.cos(xx[4])
  fx[3,1] = dt*np.sin(xx[2])*np.cos(xx[4]) + dt*np.sin(xx[4])*np.cos(xx[2])
  fx[4,1] = -dt*xx[3]*np.sin(xx[2])*np.sin(xx[4]) + dt*xx[3]*np.cos(xx[2])*np.cos(xx[4])
  fx[5,1] = 0

  fu[0,1] = 0
  fu[1,1] = 0

  #df3
  fx[0,2] = 0.0
  fx[1,2] = 0.0
  fx[2,2] = 0.0
  fx[3,2] = 0.0
  fx[4,2] = 0.0
  fx[5,2] = 1.0

  fu[0,2] = 0
  fu[1,2] = 0

  #df4
  fx[0,3] = 0
  fx[1,3] = 0
  fx[2,3] = 0
  fx[3,3] = -(dt/mm)*(-mu*((mm*gg*bb)/(aa+bb))*np.sin(xx[4])/(xx[3]*np.cos(xx[4])) + mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]**2*np.cos(xx[4])))*np.sin(uu[0] - xx[4]) + 1 - ((dt/mm)*mu*(mm*gg*aa/(aa+bb)))*np.sin(xx[4])**2/(xx[3]*np.cos(xx[4])) - ((dt/mm)*mu*(mm*gg*aa/(aa+bb)))*(-xx[3]*np.sin(xx[4]) + bb*xx[5])*np.sin(xx[4])/(xx[3]**2*np.cos(xx[4]))
  fx[4,3] = (dt/mm)*uu[1]*np.sin(uu[0] - xx[4]) - (dt/mm)*(-mu*((mm*gg*bb)/(aa+bb)) - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])*np.sin(xx[4])/(xx[3]*np.cos(xx[4])**2))*np.sin(uu[0] - xx[4]) + (dt/mm)*(mu*((mm*gg*bb)/(aa+bb))*uu[0] - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])))*np.cos(uu[0] - xx[4]) - ((dt/mm)*mu*(mm*gg*aa/(aa+bb)))*np.sin(xx[4]) + ((dt/mm)*mu*(mm*gg*aa/(aa+bb)))*(-xx[3]*np.sin(xx[4]) + bb*xx[5])*np.sin(xx[4])**2/(xx[3]*np.cos(xx[4])**2) + ((dt/mm)*mu*(mm*gg*aa/(aa+bb)))*(-xx[3]*np.sin(xx[4]) + bb*xx[5])/xx[3]
  fx[5,3] = (dt*mu*(gg*aa*bb/(aa+bb)))*np.sin(xx[4])/(xx[3]*np.cos(xx[4])) + (dt*mu*(gg*aa*bb/(aa+bb)))*np.sin(uu[0] - xx[4])/(xx[3]*np.cos(xx[4]))

  fu[0,3] = -(dt/mm)*uu[1]*np.sin(uu[0] - xx[4]) - (dt/mm)*(mu*((mm*gg*bb)/(aa+bb))*uu[0] - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])))*np.cos(uu[0] - xx[4]) - (dt*mu*(gg*bb/(aa+bb)))*np.sin(uu[0] - xx[4])
  fu[1,3] = (dt/mm)*np.cos(uu[0] - xx[4])

  #df5
  fx[0,4] = 0
  fx[1,4] = 0
  fx[2,4] = 0
  fx[3,4] = (dt/mm)*((-mu*((mm*gg*bb)/(aa+bb))*np.sin(xx[4])/(xx[3]*np.cos(xx[4])) + mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]**2*np.cos(xx[4])))*np.cos(uu[0] - xx[4]) - (mm*gg*aa/(aa+bb))*np.sin(xx[4])/xx[3] - (mm*gg*aa/(aa+bb))*(-xx[3]*np.sin(xx[4]) + bb*xx[5])/xx[3]**2)/xx[3] - (dt/mm)*(uu[1]*np.sin(uu[0] - xx[4]) + (mu*((mm*gg*bb)/(aa+bb))*uu[0] - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])))*np.cos(uu[0] - xx[4]) + (mm*gg*aa/(aa+bb))*(-xx[3]*np.sin(xx[4]) + bb*xx[5])/xx[3])/xx[3]**2
  fx[4,4] = 1 + (dt/mm)*(-uu[1]*np.cos(uu[0] - xx[4]) + (-mu*((mm*gg*bb)/(aa+bb)) - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])*np.sin(xx[4])/(xx[3]*np.cos(xx[4])**2))*np.cos(uu[0] - xx[4]) + (mu*((mm*gg*bb)/(aa+bb))*uu[0] - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])))*np.sin(uu[0] - xx[4]) - (mm*gg*aa/(aa+bb))*np.cos(xx[4]))/xx[3]
  fx[5,4] = -dt + (dt/mm)*(((mm*gg*aa/(aa+bb))*bb)/xx[3] - ((mm*gg*aa/(aa+bb))*bb)*np.cos(uu[0] - xx[4])/(xx[3]*np.cos(xx[4])))/xx[3]

  fu[0,4] = (dt/mm)*(uu[1]*np.cos(uu[0] - xx[4]) - (mu*((mm*gg*bb)/(aa+bb))*uu[0] - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])))*np.sin(uu[0] - xx[4]) + mu*((mm*gg*bb)/(aa+bb))*np.cos(uu[0] - xx[4]))/xx[3]
  fu[1,4] = (dt/mm)*np.sin(uu[0] - xx[4])/xx[3]

  #df6
  fx[0,5] = 0
  fx[1,5] = 0
  fx[2,5] = 0
  fx[3,5] = ((dt/Iz)*aa)*(-mu*((mm*gg*bb)/(aa+bb))*np.sin(xx[4])/(xx[3]*np.cos(xx[4])) + mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]**2*np.cos(xx[4])))*np.cos(uu[0]) + ((dt/Iz)*((mm*gg*bb)/(aa+bb))*aa)*np.sin(xx[4])/(xx[3]*np.cos(xx[4])) + ((dt/Iz)*((mm*gg*bb)/(aa+bb))*aa)*(-xx[3]*np.sin(xx[4]) + bb*xx[5])/(xx[3]**2*np.cos(xx[4]))
  fx[4,5] = ((dt/Iz)*aa)*(-mu*((mm*gg*bb)/(aa+bb)) - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])*np.sin(xx[4])/(xx[3]*np.cos(xx[4])**2))*np.cos(uu[0]) + ((dt/Iz)*((mm*gg*bb)/(aa+bb))*aa) - ((dt/Iz)*((mm*gg*bb)/(aa+bb))*aa)*(-xx[3]*np.sin(xx[4]) + bb*xx[5])*np.sin(xx[4])/(xx[3]*np.cos(xx[4])**2)
  fx[5,5] = 1 -((dt/Iz)*((mm*gg*bb)/(aa+bb))*(aa**2))*np.cos(uu[0])/(xx[3]*np.cos(xx[4])) - ((dt/Iz)*((mm*gg*aa)/(aa+bb))*(bb**2))/(xx[3]*np.cos(xx[4]))

  fu[0,5] = ((dt/Iz)*aa)*uu[1]*np.cos(uu[0]) - ((dt/Iz)*aa)*(mu*((mm*gg*bb)/(aa+bb))*uu[0] - mu*((mm*gg*bb)/(aa+bb))*(xx[3]*np.sin(xx[4]) + aa*xx[5])/(xx[3]*np.cos(xx[4])))*np.sin(uu[0]) + ((dt/Iz)*((mm*gg*bb)/(aa+bb))*aa)*np.cos(uu[0])
  fu[1,5] = ((dt/Iz)*aa)*np.sin(uu[0])

  xxp = xxp.squeeze()

  return xxp, fx, fu
