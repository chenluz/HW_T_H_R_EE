#!/usr/bin/env python
"""Run CartPole Environment with Imitation Learning."""
import argparse
import os
import random
import gym
import logging

import numpy as np
import tensorflow as tf
from keras.models import Sequential

from keras import optimizers

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)

from deeprl_hw3 import imitation
from gym import wrappers

        
def parser():
    parser = argparse.ArgumentParser(description='Run imitation Learning on CartPole-v0')
    parser.add_argument('--env', default='CartPole-v0', help='Environment name')
    parser.add_argument(
        '-o', '--output', default='imitation-res', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--data_episode', default=1, type=int);
    parser.add_argument('--batch_size', default=32, type=int);
    parser.add_argument('--learning_rate', default=0.0001);
    parser.add_argument('--epochs', default=100, type=float);
    parser.add_argument('--is_wrapped', default=False, type=bool);
    parser.add_argument('--is_test_expert', default=False, type=bool);
    args = parser.parse_args()
    return args
    

def print_result(args, expert, env, env_test, env_test_wrapped, episode):
    # generate training with 1 episode
    print("the reuslt for data with " + str(episode) +" episodes........")
    # generate training
    X, Y = imitation.generate_expert_training_data(expert, env, episode, False)


    # create new model by loading the configuration of expert 
    clone_model = imitation.load_model('CartPole-v0_config.yaml')

    # Compile model
    clone_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    hist = clone_model.fit(X, Y, args.epochs, args.batch_size, verbose=0)
    print(hist.history['acc'][-1])
    print(hist.history['loss'][-1])

    # evalute the model with original environemnt  
    print("the reuslt of original environment:")  
    imitation.test_cloned_policy(env_test, clone_model)

    # evalute the model  with wrapped environment 
    print("the reuslt of wrapped environment:") 
    imitation.test_cloned_policy(env_test_wrapped, clone_model)

   

def main(): 
    args = parser()

    #load the expert model 
    expert = imitation.load_model('CartPole-v0_config.yaml','CartPole-v0_weights.h5f')
    
    #create the env
    env = gym.make(args.env)

    # create the test  model
    env_test = gym.make(args.env);
    env_test.seed(0)

    # create the test model with wrapped environment
    env_for_wrapped = gym.make(args.env);
    env_for_wrapped.seed(0)
    env_test_wrapped = imitation.wrap_cartpole(env_for_wrapped)


    print_result(args, expert, env, env_test, env_test_wrapped, 1)
    print_result(args, expert, env, env_test, env_test_wrapped, 10)
    print_result(args, expert, env, env_test, env_test_wrapped, 50)
    print_result(args, expert, env, env_test, env_test_wrapped, 100)

  

    print("the reuslt for expert........")
    imitation.test_cloned_policy(env_test_wrapped, expert)
    
          
        

if __name__ == '__main__':
    main()
