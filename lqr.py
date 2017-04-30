import gym
import copy
import deeprl_hw3
import numpy as np
from deeprl_hw3.controllers import calc_lqr_input
import matplotlib.pyplot as plt
env_name = 'TwoLinkArm-v1'; # Change for different env
env = gym.make(env_name);
env_copy = copy.deepcopy(env);
is_terminal = False;
ob_next = env.reset();
F = calc_lqr_input(env, env_copy);
goal = env.goal;
reward_total = 0;
step_total = 0;
action_hist = np.zeros((1,2));
action_clipped_hist = np.zeros((1,2));
state_hist = np.zeros((1,4));
while not is_terminal:
    env.render();
    action_this = -F.dot(ob_next - goal);
    ob_next, reward_next, is_terminal, info = env.step(action_this);
    reward_total += reward_next;
    step_total += 1;
    action_hist = np.append(action_hist, action_this.reshape(1,-1), axis = 0);
    state_hist = np.append(state_hist, ob_next.reshape(1,-1), axis = 0);
    action_clipped_hist = np.append(action_clipped_hist, env.u.reshape(1,-1), axis = 0);
    #print ('Observation', ob_next);
print("Terminal?", is_terminal)

# The following codes are for plotting
plot_x = np.array(range(action_hist.shape[0]));
plt.plot(plot_x, action_hist[:,0], label = 'action_0');
plt.plot(plot_x, action_hist[:,1], label = 'action_1');
plt.xlabel('Steps')
plt.title('%s: unclipped actions'%(env_name));
plt.legend()
#x1,x2,y1,y2 = plt.axis()
#plt.axis((x1,x2,-1000,3500))
plt.show()

plot_x = np.array(range(action_hist.shape[0]));
plt.plot(plot_x, action_clipped_hist[:,0], label = 'action_0');
plt.plot(plot_x, action_clipped_hist[:,1], label = 'action_1');
plt.xlabel('Steps')
plt.title('%s: clipped actions'%(env_name));
plt.legend()
#x1,x2,y1,y2 = plt.axis()
#plt.axis((x1,x2,-1000,3500))
plt.show()

plt.plot(plot_x, state_hist[:,0], label = 'q_0');
plt.plot(plot_x, state_hist[:,1], label = 'q_1');
plt.plot(plot_x, state_hist[:,2], label = 'dq_0');
plt.plot(plot_x, state_hist[:,3], label = 'dq_1');
plt.xlabel('Steps')
plt.title('%s: state trajectory'%(env_name));
plt.legend()
#x1,x2,y1,y2 = plt.axis()
#plt.axis((x1,x2,-200,200))
plt.show()
print ('reward total: ', reward_total);
print ('step total: ', step_total);

