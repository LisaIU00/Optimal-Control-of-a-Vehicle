import cvxpy as cp

def linear_mpc(ns,ni,AA, BB, QQ, RR, QQf, xxt, Tmin, T_pred, x_opt, u_opt):
    """
        Linear MPC solver - Constrained LQR

        Given a measured state xxt measured at t
        gives back the optimal input to be applied at t

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xxt: initial condition (at time t)
          - T: time (prediction) horizon

        Returns
          - u_t: input to be applied at t
          - xx, uu predicted trajectory

    """
    xxt = xxt.squeeze()

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    cost = 0
    constr = []
    
    for tt in range(T_pred-1):
        cost += (1/2)*cp.quad_form((xx_mpc[:,tt]-x_opt[:,tt+Tmin-1]), QQ[:,:]) + (1/2)*cp.quad_form((uu_mpc[:,tt]-u_opt[:,tt+Tmin-1]), RR[:,:])
        constr += [xx_mpc[:,tt+1]-x_opt[:,tt+Tmin] == AA[:,:, Tmin+tt]@(xx_mpc[:,tt]-x_opt[:,tt+Tmin-1]) + BB[:,:,Tmin+tt]@(uu_mpc[:,tt]-u_opt[:,tt+Tmin-1])] # dynamics constraint
    
    cost += (1/2)*cp.quad_form((xx_mpc[:,T_pred-1]-x_opt[:,T_pred+Tmin-1]), QQf)
    constr += [xx_mpc[:,0] == xxt]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value