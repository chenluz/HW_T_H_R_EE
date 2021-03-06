"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
from numpy import linalg as LA
import numpy as np
import scipy.linalg
from numpy.linalg import inv
import time

itermediate_scaling = 1e-4
action_limit = 500;
sim_dt = 1e-3;
sing_mat_lambda = 1e-2;
minimum_alpha = 1e-8;
def simulate_dynamics(env, x, u, dt):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    next_x: np.array
    """
    # Set the state of the environment
    env.state = x;
    # Get resulting state
    ob_next, reward, is_terminal, info = env._step(u, dt = dt);
    
    return ob_next, reward;


def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    lx = np.zeros((1, x.shape[0]));
    lxx = np.zeros((x.shape[0], x.shape[0]));
    lu = u.copy().reshape((1, -1)) * 2.0 * itermediate_scaling # np.zeros((1, u.shape[0]));
    luu = np.zeros((u.shape[0], u.shape[0]));
    luu[0,0] = 2.0 * itermediate_scaling;
    luu[1,1] = 2.0 * itermediate_scaling;
    lux = np.zeros((u.shape[0], x.shape[0]));
    l = itermediate_scaling * LA.norm(u) ** 2.0;
    
    return (l, lx, lxx, lu, luu, lux);

def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
      Each column corresponds to the derivative of one feature in the
      state. 
    """
    A = np.zeros((x.shape[0], x.shape[0]));
    for i in range(x.shape[0]):
        x_copy_plus = x.copy();
        x_copy_plus[i] = x_copy_plus[i] + delta;
        x_copy_mins = x.copy();
        x_copy_mins[i] = x_copy_mins[i] - delta;
        d_i = (simulate_dynamics(env, x_copy_plus, u, dt)[0] -
              simulate_dynamics(env, x_copy_mins, u, dt)[0])/(delta * 2.0);
        A[:, i] = d_i; 
    return A;

def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]));
    for i in range(u.shape[0]):
        u_copy_plus = u.copy();
        u_copy_plus[i] = u_copy_plus[i] + delta;
        u_copy_mins = u.copy();
        u_copy_mins[i] = u_copy_mins[i] - delta;
        d_i = (simulate_dynamics(env, x, u_copy_plus, dt)[0] -
              simulate_dynamics(env, x, u_copy_mins, dt)[0])/(delta * 2.0);
        B[:, i] = d_i; 
    return B;

def cost_final(env, x, u):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """    
    scaling = 1e4;
    lx =  (x.copy().reshape((1, -1)) - env.goal.copy().reshape((1, -1))) * 2.0 * scaling # np.zeros((1, x.shape[0]));
    lxx = np.zeros((x.shape[0], x.shape[0]));
    lxx[0,0] = 2.0 * scaling;
    lxx[1,1] = 2.0 * scaling;
    lxx[2,2] = 2.0 * scaling;
    lxx[3,3] = 2.0 * scaling;
    lu = np.zeros((1, u.shape[0]));
    luu = np.zeros((u.shape[0], u.shape[0]));
    lux = np.zeros((u.shape[0], x.shape[0]));
    l = (LA.norm(x - env.goal) ** 2.0) * scaling;
    return (l, lx, lxx, lu, luu, lux);


def simulate(env, x0, U, step):
    reward_total = 0;
    cost_total = 0;
    step_total = 0;
    action_hist = np.zeros((1,2));
    action_clipped_hist = np.zeros((1,2));
    state_hist = np.zeros((1,4));
    ob_next = env.reset();
    is_sleeped = False;
    for i in range(U.shape[0]):
        env.render();
        if not is_sleeped:
            time.sleep(0.5)
            is_sleeped = True;
        action_this = U[i, :];
        cost_total += itermediate_scaling * LA.norm(action_this) ** 2.0;
        ob_next, reward_next, is_terminal, info = env._step(action_this, dt=sim_dt);
        reward_total += reward_next;
        step_total += 1;
        action_hist = np.append(action_hist, action_this.reshape(1,-1), axis = 0);
        state_hist = np.append(state_hist, ob_next.reshape(1,-1), axis = 0);
        action_clipped_hist = np.append(action_clipped_hist, env.u.reshape(1,-1), axis = 0);
    env.render(close=True);
    print ('final_cost', (LA.norm(ob_next - env.goal) ** 2.0) * 1e4)
    cost_total += (LA.norm(ob_next - env.goal) ** 2.0) * 1e4;
    #np.savetxt('reward_cost_%d.csv'%(step), np.array([reward_total, cost_total]), delimiter = ',');
    np.savetxt('action_hist_%f.csv'%(itermediate_scaling), action_hist, delimiter = ',');
    np.savetxt('action_clipped_hist_%f.csv'%(itermediate_scaling), action_clipped_hist, delimiter = ',');
    np.savetxt('state_hist_%f.csv'%(itermediate_scaling), state_hist, delimiter = ',');
    return (reward_total, cost_total);

def backward_recursion(x_trj, u_trj, tN, env, lamb):
    """
    x_trj has length tN + 1
    y_trj has length tN
    """
    K_array = np.zeros((2,4,tN)); # 2*4*tN
    k_array = np.zeros((2,1,tN)); # 2*1*tN
    # The last state
    x_last = x_trj[-1, :].flatten() # 4
    u_last = u_trj[-1, :].flatten() # 2
    l, l_x, l_xx, l_u, l_uu, l_ux = cost_final(env, x_last, u_last)
    l_xu = l_ux.copy().T; # 4*2
    K_T = np.zeros((2, 4)) # 2*4
    k_T = np.zeros((2, 1)) # 2*1
    V_T = l_xx + l_xu.dot(K_T) + K_T.T.dot(l_ux) + K_T.T.dot(l_uu).dot(K_T); # 4*4
    v_T = l_x.T + l_xu.dot(k_T) + K_T.T.dot(l_u.T) + K_T.T.dot(l_uu).dot(k_T) # 4*1
    V_tplus1 = V_T;
    v_tplus1 = v_T;
    # Backward recursion
    for i in range(tN):
        t = tN - i - 1;
        xt = x_trj[t, :].flatten(); # 4
        ut = u_trj[t, :].flatten(); # 2
        l, l_x, l_xx, l_u, l_uu, l_ux = cost_inter(env, xt, ut);
        l_xu = l_ux.copy().T; # 4*2
        A = approximate_A(env, xt, ut, delta=1e-5, dt=1e-5);
        B = approximate_B(env, xt, ut, delta=1e-5, dt=1e-5);
        Qt_uu = l_uu + B.copy().T.dot(V_tplus1).dot(B)# 2*2
        Qt_ux = l_ux + B.copy().T.dot(V_tplus1).dot(A)# 2*4
        Qt_xx = l_xx + A.copy().T.dot(V_tplus1).dot(A)# 4*4
        qt_u = l_u.copy().T + B.copy().T.dot(v_tplus1)# 2*1
        qt_x = l_x.copy().T + A.copy().T.dot(v_tplus1)# 4*1
        Kt = -inv(Qt_uu + sing_mat_lambda*np.identity(2)).dot(Qt_ux) # 2*4
        kt = -inv(Qt_uu + sing_mat_lambda*np.identity(2)).dot(qt_u) # 2*1
        K_array[:,:,t] = Kt;
        k_array[:,:,t] = kt;
        Vt = Qt_xx - Kt.copy().T.dot(Qt_uu).dot(Kt)# 4*4
        vt = qt_x - Kt.copy().T.dot(Qt_uu).dot(kt) # 4*1
        V_tplus1 = Vt;
        v_tplus1 = vt;
    return K_array, k_array;

def regularized_inv(Quu, lamb):
    """
    Copied from 
    https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
    """
    U, S, V = np.linalg.svd(Quu)
    S[S < 0.0] = 0.0 # no negative values
    S += lamb # add regularization term
    Quu_inv = np.dot(U, np.dot(np.diag(1.0/S), V.T));
    return Quu_inv

def forward_path(env, x_trj_cp, u_trj_cp, alpha, K_array, k_array, tN):
    # Forward pass
    cost_this = 0;
    xi = env.reset();
    x_trj = np.zeros((tN + 1, 4));
    u_trj = np.zeros((tN, 2));
    x_trj[0, :] = xi;
    x_T = env.goal;

    for i in range(tN):
        u_base = u_trj_cp[i, :] + K_array[:,:, i].dot(x_trj[i, :] - x_trj_cp[i, :]);
        u_change =  k_array[:, :, i].flatten()
        ui = u_base + alpha * u_change;
        ui = np.clip(ui, -action_limit, action_limit)
        u_trj[i, :] = ui;
        ob, reward, is_terminal, info = env._step(ui, dt=sim_dt);
        cost_this += itermediate_scaling * LA.norm(ui) ** 2.0;
        x_trj[i + 1, :] = ob;
    cost_this += (LA.norm(ob - env.goal) ** 2.0) * 1e4;
    return cost_this, u_trj, x_trj;

def epi_exhaust_search(env, x_trj_cp, u_trj_cp, K_array, k_array, cost_last, tN, minmum):
    alpha = 1.0;
    factor = 0.5;
    dontknow = 0.3;
    is_stop = False;
    while not is_stop:
        cost_this = forward_path(env, x_trj_cp, u_trj_cp, alpha, K_array, k_array, tN)[0];
        if cost_this <= cost_last or alpha < minmum:
            is_stop = True;
        else:
            alpha *= factor 
    
    return alpha

def backtrack_line_search(env, x_trj_cp, u_trj_cp, K_array, k_array, cost_last, tN):
    alpha = 1.0;
    factor = 0.5;
    dontknow = 0.3;
    is_stop = False;
    while not is_stop:
        cost_this = forward_path(env, x_trj_cp, u_trj_cp, alpha, K_array, k_array, tN)[0];
        if cost_this <= (cost_last - dontknow * alpha * LA.norm(k_array) ** 2.0):# or alpha < 1e-5:
            is_stop = True;
        else:
            alpha *= factor  
    return alpha

def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e5, lamb_factor = 1.001, is_warm_start = True):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    max_state_diff = 1000;
    iter_counter = 0;
    lamb = 100;
    lamb_factor = 1.1
    minimum = minimum_alpha;
    x_trj = np.zeros((tN + 1, 4));
    u_trj = np.zeros((tN, 2));
    # initial forward path
    x_trj[0, :] = env.reset();
    cost_last = 0;
    if is_warm_start:
        u_hist = np.loadtxt('action_hist_0.000100_500to5000limit_p4.csv', delimiter = ',')[1:101, :]
        for i in range(tN):
            ui = u_hist[i, :];
            cost_last += itermediate_scaling * LA.norm(ui) ** 2.0;
            u_trj[i, :] = ui;
            ob, reward, is_terminal, info = env._step(ui, dt=sim_dt);
            x_trj[i + 1, :] = ob;
    else:
        for i in range(tN):
            ui = np.array([0.0, 0.0]);#np.random.rand(2)
            cost_last += itermediate_scaling * LA.norm(ui) ** 2.0;
            u_trj[i, :] = ui;
            ob, reward, is_terminal, info = env._step(ui, dt=sim_dt);
            x_trj[i + 1, :] = ob;
    cost_last += (LA.norm(ob - env.goal) ** 2.0) * 1e4;    
    reward_cost_array = np.zeros((1, 2));
    while iter_counter < max_iter:
        print (iter_counter)
        if iter_counter % 100 == 0:
            print (iter_counter, ' Evaluating')
            reward_total, cost_total = simulate(sim_env, None, u_trj, iter_counter);
            reward_cost_array = np.append(reward_cost_array, np.array([[reward_total, cost_total]]),axis = 0);
            np.savetxt('reward_cost_%f.csv'%(itermediate_scaling), reward_cost_array, delimiter = ',');
        # Backward pass
        K_array, k_array = backward_recursion(x_trj, u_trj, tN, sim_env, lamb);
        # Forward pass
        x_trj_cp = x_trj.copy();
        u_trj_cp = u_trj.copy();
        alpha = epi_exhaust_search(env, x_trj_cp, u_trj_cp, K_array, k_array, cost_last, tN, minimum)
        cost_this, u_trj, x_trj = forward_path(env, x_trj_cp, u_trj_cp, alpha, K_array, k_array, tN);
        cost_last = cost_this;
        print (alpha)
        print (cost_this)
        iter_counter += 1;
        
    return u_trj;
