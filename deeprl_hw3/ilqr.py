"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
from numpy import linalg as LA
import numpy as np
import scipy.linalg
from numpy.linalg import inv
import time

itermediate_scaling = 1.0
action_limit = 5000;
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
    
    ## l_x
    #for i in range(x.shape[0]):
        #x_copy_plus = x.copy();
        #x_copy_plus[i] = x_copy_plus[i] + delta;
        #x_copy_mins = x.copy();
        #x_copy_mins[i] = x_copy_mins[i] - delta;
        #d_i = (simulate_dynamics(env, x_copy_plus, u, dt)[1] -
              #simulate_dynamics(env, x_copy_mins, u, dt)[1])/(delta * 2.0);
        #lx[:, i] = d_i;
    ## l_xx
    #for i in range(x.shape[0]):
        #for j in range(x.shape[0]):
            #if i==j:
                #x_copy_plus = x.copy();
                #x_copy_plus[i] = x_copy_plus[i] + delta;
                #x_copy_mins = x.copy();
                #x_copy_mins[i] = x_copy_mins[i] - delta;
                #d_i = (simulate_dynamics(env, x_copy_plus, u, dt)[1] -
                        #2 * simulate_dynamics(env, x.copy(), u, dt)[1] +
                        #simulate_dynamics(env, x_copy_mins, u, dt)[1])/(delta ** 2.0);
                #lxx[i, j] = d_i;
            #else:
                #x_copy_dt_0 = x.copy();
                #x_copy_dt_0[i] = x_copy_dt_0[i] + delta;
                #x_copy_dt_0[j] = x_copy_dt_0[j] + delta;
                #x_copy_dt_1 = x.copy();
                #x_copy_dt_1[i] = x_copy_dt_1[i] + delta;
                #x_copy_dt_1[j] = x_copy_dt_1[j] - delta;
                #x_copy_dt_2 = x.copy();
                #x_copy_dt_2[i] = x_copy_dt_2[i] - delta;
                #x_copy_dt_2[j] = x_copy_dt_2[j] + delta;
                #x_copy_dt_3 = x.copy();
                #x_copy_dt_3[i] = x_copy_dt_3[i] - delta;
                #x_copy_dt_3[j] = x_copy_dt_3[j] - delta;
                #d_i = (simulate_dynamics(env, x_copy_dt_0, u, dt)[1] -
                        #simulate_dynamics(env, x_copy_dt_1, u, dt)[1] -
                        #simulate_dynamics(env, x_copy_dt_2, u, dt)[1] +
                        #simulate_dynamics(env, x_copy_dt_3, u, dt)[1])/(4 * delta * delta);
                #lxx[i, j] = d_i; 
    ## l_u 
    #for i in range(u.shape[0]):
        #u_copy_plus = u.copy();
        #u_copy_plus[i] = u_copy_plus[i] + delta;
        #u_copy_mins = u.copy();
        #u_copy_mins[i] = u_copy_mins[i] - delta;
        #d_i = (simulate_dynamics(env, x, u_copy_plus, dt)[1] -
              #simulate_dynamics(env, x, u_copy_mins, dt)[1])/(delta * 2.0);
        #lu[:, i] = d_i;
    ## l_uu
    #for i in range(u.shape[0]):
        #for j in range(u.shape[0]):
            #if i==j:
                #u_copy_plus = u.copy();
                #u_copy_plus[i] = u_copy_plus[i] + delta;
                #u_copy_mins = u.copy();
                #u_copy_mins[i] = u_copy_mins[i] - delta;
                #d_i = (simulate_dynamics(env, x, u_copy_plus, dt)[1] -
                        #2 * simulate_dynamics(env, x, u, dt)[1] +
                        #simulate_dynamics(env, x, u_copy_mins, dt)[1])/(delta ** 2.0);
                #luu[i, j] = d_i;
            #else:
                #u_copy_dt_0 = u.copy();
                #u_copy_dt_0[i] = u_copy_dt_0[i] + delta;
                #u_copy_dt_0[j] = u_copy_dt_0[j] + delta;
                #u_copy_dt_1 = u.copy();
                #u_copy_dt_1[i] = u_copy_dt_1[i] + delta;
                #u_copy_dt_1[j] = u_copy_dt_1[j] - delta;
                #u_copy_dt_2 = u.copy();
                #u_copy_dt_2[i] = u_copy_dt_2[i] - delta;
                #u_copy_dt_2[j] = u_copy_dt_2[j] + delta;
                #u_copy_dt_3 = u.copy();
                #u_copy_dt_3[i] = u_copy_dt_3[i] - delta;
                #u_copy_dt_3[j] = u_copy_dt_3[j] - delta;
                #d_i = (simulate_dynamics(env, x, u_copy_dt_0, dt)[1] -
                        #simulate_dynamics(env, x, u_copy_dt_1, dt)[1] -
                        #simulate_dynamics(env, x, u_copy_dt_2, dt)[1] +
                        #simulate_dynamics(env, x, u_copy_dt_3, dt)[1])/(4 * delta * delta);
                #luu[i, j] = d_i; 
    ## l_ux
    #for i in range(u.shape[0]):
        #for j in range(x.shape[0]):
            #u_copy_dt_0 = u.copy();
            #u_copy_dt_0[i] = u_copy_dt_0[i] + delta;
            #x_copy_dt_0 = x.copy();
            #x_copy_dt_0[j] = x_copy_dt_0[j] + delta;
            #u_copy_dt_1 = u.copy();
            #u_copy_dt_1[i] = u_copy_dt_1[i] + delta;
            #x_copy_dt_1 = x.copy();
            #x_copy_dt_1[j] = x_copy_dt_1[j] - delta;
            #u_copy_dt_2 = u.copy();
            #u_copy_dt_2[i] = u_copy_dt_2[i] - delta;
            #x_copy_dt_2 = x.copy();
            #x_copy_dt_2[j] = x_copy_dt_2[j] + delta;
            #u_copy_dt_3 = u.copy();
            #u_copy_dt_3[i] = u_copy_dt_3[i] - delta;
            #x_copy_dt_3 = x.copy();
            #x_copy_dt_3[j] = x_copy_dt_3[j] - delta;
            #d_i = (simulate_dynamics(env, x_copy_dt_0, u_copy_dt_0, dt)[1] -
                    #simulate_dynamics(env, x_copy_dt_1, u_copy_dt_1, dt)[1] -
                    #simulate_dynamics(env, x_copy_dt_2, u_copy_dt_2, dt)[1] +
                    #simulate_dynamics(env, x_copy_dt_3, u_copy_dt_3, dt)[1])/(4 * delta * delta);
            #lux[i, j] = d_i; 
    ## l
    #l = simulate_dynamics(env, x, u, dt)[1];
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
        ob_next, reward_next, is_terminal, info = env.step(action_this);
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
    l, l_x, l_xx, l_u, l_uu, l_ux = cost_final(env, x_last, u_last);
    l_xu = l_ux.copy().T; # 4*2
    K_T = np.zeros((2, 4)) # -inv(l_uu).dot(l_ux); # 2*4
    k_T = np.zeros((2, 1)) # -inv(l_uu).dot(l_u.T) # 2*1
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
        #Ft = np.append(A, B, axis = 1); # 4*6
        #Ct = np.vstack((np.hstack((l_xx, l_xu)),np.hstack((l_ux, l_uu))));
        #ct = np.hstack((l_x, l_u)).T; # 6*1
        #Qt = Ct + Ft.T.dot(V_tplus1).dot(Ft); # 6*6
        #qt = ct + Ft.T.dot(V_tplus1).dot(np.zeros((4,1))) + Ft.T.dot(v_tplus1); # 6*1
        Qt_uu = l_uu + B.copy().T.dot(V_tplus1).dot(B) # Qt.copy()[4:,4:]; # 2*2
        Qt_ux = l_ux + B.copy().T.dot(V_tplus1).dot(A) # Qt.copy()[4:,0:4]; # 2*4
        Qt_xx = l_xx + A.copy().T.dot(V_tplus1).dot(A) # Qt.copy()[0:4, 0:4]; # 4*4
        #Qt_xu = Qt.copy()[0:4, 4:]; # 4*2
        qt_u = l_u.copy().T + B.copy().T.dot(v_tplus1)# qt.copy()[4:, :] # 2*1
        qt_x = l_x.copy().T + A.copy().T.dot(v_tplus1)# qt.copy()[0:4, :] # 4*1
        Kt = -inv(Qt_uu + 0.01*np.identity(2)).dot(Qt_ux) # 2*4
        kt = -inv(Qt_uu + 0.01*np.identity(2)).dot(qt_u) # 2*1
        K_array[:,:,t] = Kt;
        k_array[:,:,t] = kt;
        Vt = Qt_xx - Kt.copy().T.dot(Qt_uu).dot(Kt) # Qt_xx + Qt_xu.dot(Kt) + Kt.T.dot(Qt_ux) + Kt.T.dot(Qt_uu).dot(Kt); # 4*4
        vt = qt_x - Kt.copy().T.dot(Qt_uu).dot(kt) #qt_x + Qt_xu.dot(kt) + Kt.T.dot(qt_u) + Kt.T.dot(Qt_uu).dot(kt); # 4*1
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

def exhaust_search(sim_env, cur_state, u_base, u_change, is_final):
    steps = 100.0;
    alpha = 0.0;
    delta = 0.1/steps;
    cost_max = -1e20;
    alpha_opt = alpha;
    for _ in range(int(steps)):
        u_new = u_base + alpha * u_change;
        sim_env.state = cur_state;
        ob, cost_this, is_terminal, info = sim_env.step(u_new);
        #if is_final:
            #sim_env.state = cur_state;
            #ob, reward, is_terminal, info = sim_env.step(u_new);
            #cost_this = (LA.norm(ob - sim_env.goal) ** 2.0) * 1e4;
        #else:
            #cost_this = itermediate_scaling * LA.norm(u_new) ** 2.0;
        if cost_this > cost_max:
            cost_max = cost_this;
            alpha_opt = alpha;
        alpha += delta;
    alpha_opt = max(alpha_opt, 1e-5);
    return alpha_opt;

def forward_path(env, x_trj_cp, u_trj_cp, alpha, K_array, k_array, tN):
    # Forward pass
    cost_this = 0;
    xi = env.reset();
    x_trj = np.zeros((tN + 1, 4));
    u_trj = np.zeros((tN, 2));
    x_trj[0, :] = xi;
        #max_state_diff = LA.norm(x_trj[0, :] - x_trj_cp[0, :]);
    for i in range(tN):
        u_base = u_trj_cp[i, :] + K_array[:,:, i].dot(x_trj[i, :] - x_trj_cp[i, :]);#xT); #x_trj_cp[i, :])
        u_change =  k_array[:, :, i].flatten()
        ui = u_base + alpha * u_change;
        ui = np.clip(ui, -action_limit, action_limit)
        cost_this += itermediate_scaling * LA.norm(ui) ** 2.0;
        u_trj[i, :] = ui;
        ob, reward, is_terminal, info = env.step(ui);
        x_trj[i + 1, :] = ob;
        #max_state_diff = max(LA.norm(x_trj[i + 1, :] - x_trj_cp[i + 1, :]), max_state_diff);
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
    minimum = 1e-8;
    x_trj = np.zeros((tN + 1, 4));
    u_trj = np.zeros((tN, 2));
    # initial forward path
    x_trj[0, :] = env.reset();
    cost_last = 0;
    if is_warm_start:
        u_hist = np.loadtxt('action_hist_0.000000_p1.csv', delimiter = ',')[1:101, :]
        for i in range(tN):
            ui = u_hist[i, :];
            cost_last += itermediate_scaling * LA.norm(ui) ** 2.0;
            u_trj[i, :] = ui;
            ob, reward, is_terminal, info = env.step(ui);
            x_trj[i + 1, :] = ob;
    else:
        for i in range(tN):
            ui = np.array([0.0, 0.0]);#np.random.rand(2)
            cost_last += itermediate_scaling * LA.norm(ui) ** 2.0;
            u_trj[i, :] = ui;
            ob, reward, is_terminal, info = env.step(ui);
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
        if iter_counter % 80 == 0:
            minimum = max(1e-8, minimum * 0.9);
        # Backward pass
        K_array, k_array = backward_recursion(x_trj, u_trj, tN, sim_env, lamb);
        #print ('K', K_array, 'k', k_array)
        # Forward pass
        x_trj_cp = x_trj.copy();
        u_trj_cp = u_trj.copy();
        alpha = epi_exhaust_search(env, x_trj_cp, u_trj_cp, K_array, k_array, cost_last, tN, minimum)
        cost_this, u_trj, x_trj = forward_path(env, x_trj_cp, u_trj_cp, alpha, K_array, k_array, tN);
        #if cost_this <= cost_last:
            #lamb /= lamb_factor;
            ##alpha = min(alpha + 0.000001, 0.1);
        #else:
            #lamb *= lamb_factor;
            ##alpha = max(alpha - 0.00001, 0.0);
        cost_last = cost_this;
        print (alpha)
        print (cost_this)
        iter_counter += 1;
        
    return u_trj;
