import gym
import numpy as np

from keras.models import model_from_yaml
import tensorflow as tf
import numpy as np
#from deeprl_hw3.imitation import load_model
#from deeprl_hw3.imitation import wrap_cartpole

def load_model(model_config_path, model_weights_path=None):
    """Load a saved model.

    Parameters
    ----------
    model_config_path: str
      The path to the model configuration yaml file. We have provided
      you this file for problems 2 and 3.
    model_weights_path: str, optional
      If specified, will load keras weights from hdf5 file.

    Returns
    -------
    keras.models.Model
    """
    with open(model_config_path, 'r') as f:
        model = model_from_yaml(f.read())

    if model_weights_path is not None:
        model.load_weights(model_weights_path)

    model.summary()

    return model

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
    # cumulative_reward = 0
    # episode_length = 0
    # f1=open('reinforce.txt','w')
    # while episode_length < 100:
    #       observation = env.reset()
    #       episode_reward = 0

    #       #for step in range(max_episode_length):
    #         # Make sure the observation is in a shape the network can handle.
    #         x = np.reshape(observation,[1,4])
        
    #         # Run the policy network and get an action to take. 
    #         tfprob = sess.run(probability,feed_dict={observations: x})
    #         action = 1 if np.random.uniform() < tfprob else 0

    #         observation, reward, done, info = env.step(action)

    #         cumulative_reward += reward
    #         episode_reward += reward
    #         if done:
    #             episode_length += 1
    #             break
    #             f1.write('{}\n'.format(episode_reward))

    #     cumulative_reward /= num_episodes
    #     episode_length /= num_episodes
    #     f1.close()


    #     logging.info('average episode length: {}'.format(episode_length))
    #     logging.info('average accumulative reward: {}'.format(cumulative_reward))

    #     return cumulative_reward
    return 0.0



def choose_action(model, observation): 
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
    # weights = model.weights #correct??
    # policy = tf.nn.softmax(tf.matmul(observation, weights)) #correct??
    # observation = np.asarray(observation)
    # #Since CartPole dimensionality is 4
    # observation = observation.reshape(1,4) 
    # x = tf.placeholder("float", [None, 4]) 
    # # sess = tf.Session(tf.Graph()) 
    # # softmax_out = sess.run(policy, feed_dict={x:observation}) 
    # # p = softmax_out[0] 
    # p = policy[0]
    # action = np.random.choice([0,1], 1, replace = True, p = p)[0] #Sample action from prob density
    # return p, action
    return .5, 0

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
    for t in reversed(range(0, r.size)):
        running_add = running_add + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():

    env = gym.make('CartPole-v0')
    
    # hyperparameters
    H = 10 # number of hidden layer neurons
    batch_size = 1 # every how many episodes to do a param update?
    learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
    gamma = 1 # discount factor for reward

    D = 4 # input dimensionality
    tf.reset_default_graph()

    #This defines the network as it goes from taking an observation of the environment to 
    #giving a probability of chosing to the action of moving left or right.
    observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
    # W1 = tf.get_variable("W1", shape=[D, H],
    #        initializer=tf.contrib.layers.xavier_initializer())
    # layer1 = tf.nn.relu(tf.matmul(observations,W1))
    # W2 = tf.get_variable("W2", shape=[H, 1],
    #        initializer=tf.contrib.layers.xavier_initializer())
    # score = tf.matmul(layer1,W2)
    # probability = tf.nn.sigmoid(score)
    model = load_model('CartPole-v0_config.yaml','CartPole-v0_weights.h5f')
    weights = model.weights
    probability = tf.matmul(observations,weights)
    # assign_ops = []
    # for w, new_w in zip(model.weights, weights):
    #     assign_ops.append(w.assign(new_w))

    #From here we define the parts of the network needed for learning a good policy.
    #tvars = tf.trainable_variables()
    tvars = weights
    input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
    advantages = tf.placeholder(tf.float32,name="reward_signal")

    # The loss function. This sends the weights in the direction of making actions 
    # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
    loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
    loss = -tf.reduce_mean(loglik * advantages) 
    newGrads = tf.gradients(loss,tvars)

    # Once we have collected a series of gradients from multiple episodes, we apply them.
    # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
    #W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
    #W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
    #batchGrad = [W1Grad,W2Grad]
    batchGrad = tf.placeholder(tf.floag32,name="batch_grad")
    updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

    xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 1
    total_episodes = 10000
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        rendering = False
        sess.run(init)
        observation = env.reset() # Obtain an initial observation of the environment

        # Reset the gradient placeholder. We will collect gradients in 
        # gradBuffer until we are ready to update our policy network. 
        gradBuffer = sess.run(tvars)
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while episode_number <= total_episodes:

            # Rendering the environment slows things down, 
            # so let's only look at it once our agent is doing a good job.
            if reward_sum/batch_size > 100 or rendering == True : 
               # env.render()
                rendering = True
            
            # Make sure the observation is in a shape the network can handle.
            x = np.reshape(observation,[1,D])
        
            # Run the policy network and get an action to take. 
            tfprob = sess.run(probability,feed_dict={observations: x})
            action = 1 if np.random.uniform() < tfprob else 0
        
            xs.append(x) # observation
            y = 1 if action == 0 else 0 # a "fake label"
            ys.append(y)

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward

            drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

            if done: 
                episode_number += 1
                print('episode_number:{}\n'.format(episode_number))
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                tfp = tfps
                xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

                # compute the discounted reward backwards through time
                discounted_epr = discount_rewards(epr)
                # size the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
            
                # Get the gradient for this episode, and save it in the gradBuffer
                tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
                
                # If we have completed enough episodes, then update the policy network with our gradients.
                if episode_number % batch_size == 0: 
                    #sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                    sess.run(updateGrads,feed_dict={batchGrad: gradBuffer})
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                    # Give a summary of how well our network is doing for each batch of episodes.
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print ('Average reward for episode {}'.format(reward_sum/batch_size))
                    print ('Total average reward {}'.format(running_reward/batch_size))

                    if reward_sum/batch_size > 300: 
                        print('Task solved in'+ str(episode_number) + 'episodes!')
                        break
                    reward_sum = 0

                if rendering == True and episode_number % 1000 == 0:
                    f1 = open('test_{}.txt'.format(episode_number),'w')
                    episode_length = 0
                    observation_eval = env.reset()
                    episode_reward = 0
                    
                    while episode_length < 100:
                        x_eval = np.reshape(observation_eval,[1,D])

                        # Run the policy network and get an action to take. 
                        tfprob = sess.run(probability,feed_dict={observations: x_eval})
                        action = 1 if np.random.uniform() < tfprob else 0

                        # step the environment and get new measurements
                        observation, reward_eval, done, info = env.step(action)
                        episode_reward += reward_eval
 
                        if done:
                            episode_length += 1
                            observation_eval = env.reset()
                            print('episode_length:{}\n'.format(episode_length))
                            f1.write('{}\n'.format(episode_reward))
                            episode_reward = 0

                    f1.close()

                observation = env.reset()

    print(str(episode_number) + 'Episodes completed.')


if __name__ == '__main__':
    main()



