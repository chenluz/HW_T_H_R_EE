import gym
import numpy as np

from keras.models import model_from_yaml
import tensorflow as tf
import numpy as np
import time
from deeprl_hw3.imitation import load_model

def get_total_reward(env, model):
    """compute total reward for testing

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    return 0.0



def choose_action(model, observation): #change!!
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    """
    #Use softmax policy to sample
    weights = model.weights #correct??
    policy = tf.nn.softmax(tf.matmul(observation, weights))
    observation = np.asarray(observation)
    #Since CartPole dimensionality is 4
    observation = observation.reshape(1,4) 
    x = tf.placeholder("float", [None, 4]) 
    # sess = tf.Session(tf.Graph()) 
    # softmax_out = sess.run(policy, feed_dict={x:observation}) 
    # p = softmax_out[0] 
    p = policy[0]
    action = np.random.choice([0,1], 1, replace = True, p = p)[0] #Sample action from prob density
    return p, action
    #return .5, 0

def reinforce(env):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """
    return 0.0

def discount_rewards(r):
    """ take 1D float array of rewards and compute reward without discount"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add + r[t]
        discounted_r[t] = running_add
    return discounted_r

def evaluate():
    pass

def main():

    env = make('CartPole-v0')
    batch_size = 5
    tf.reset_default_graph()

    observations = tf.placeholder(tf.float32, [None,4] , name="input_x")
    model = load_model("./CartPole-v0_config.yaml","./CartPole-v0_weights.h5f")
    weights = model.weights

    #Use softmax policy to sample
    policy = tf.nn.softmax(tf.matmul(observations, weights)) #check softmax
    probability = policy[0] #correct??

    #From here we define the parts of the network needed for learning a good policy.
    #tvars = tf.trainable_variables()
    input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
    advantages = tf.placeholder(tf.float32,name="reward_signal")
    # The loss function. This sends the weights in the direction of making actions 
    # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
    loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability)) 
    loss = -tf.reduce_mean(loglik * advantages) 
    #newGrads = tf.gradients(loss,tvars)
    newGrads = tf.gradients(loss,weights) #correct??

    # Once we have collected a series of gradients from multiple episodes, we apply them.
    # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
    adam = tf.train.AdamOptimizer(learning_rate=0.01) # Our optimizer
    #correct??
    W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
    W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
    batchGrad = [W1Grad,W2Grad]
    #updateGrads = adam.apply_gradients(zip(batchGrad,tvars))
    updateGrads = adam.apply_gradients(zip(batchGrad,weights)) #correct?

    xs,hs,dlogps,drs,ys = [],[],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 1
    total_episodes = 10000
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        rendering = False
        sess.run(init)
        observation = env.reset() # Obtain an initial observation of thse environment

        # Reset the gradient placeholder. We will collect gradients in 
        # gradBuffer until we are ready to update our policy network. 
        gradBuffer = sess.run(weights) #correct??
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while episode_number <= total_episodes:
            #p,action = choose_action(model,observation)
            # Make sure the observation is in a shape the network can handle.
            x = np.reshape(observation,[1,4])
            tfprob = sess.run(probability,feed_dict={observations: x})
            action = 1 if np.random.uniform() < tfprob else 0
            xs.append(x) # observation

            y = 1 if action == 0 else 0 #a "fake label"
            ys.append(y)

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward

            drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

            if done: 
                episode_number += 1
                # stack together all inputs, hidden states, action gradients, and rewards for this episode #change!!
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                xs,hs,dlogps,drs,ys = [],[],[],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                #gradBuffer[ix] += grad[0] #correct??
                gradBuffer[ix] += grad
                
            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0: 
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print 'Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size)
                
                if reward_sum/batch_size > 200: 
                    print "Task solved in",episode_number,'episodes!'
                    break

                reward_sum = 0




