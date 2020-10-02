import gym
from collections import deque
import tensorflow as tf
from dqn_agent import DQNAgent
import numpy as np
import random

# setting seeds for result reproducibility. This is not super important
random.seed(2212)
np.random.seed(2212)
tf.set_random_seed(2212)

# Hyperparameters / Constants
EPISODES = 10_000
REPLAY_MEMORY_SIZE = 1_00_000
MINIMUM_REPLAY_MEMORY = 1_000
MINIBATCH_SIZE = 32
EPSILON = 1
EPSILON_DECAY = 0.99
MINIMUM_EPSILON = 0.001
DISCOUNT = 0.99
VISUALIZATION = False
ENV_NAME = 'MountainCar-v0'

# Environment details
env = gym.make(ENV_NAME)
action_dim = env.action_space.n
observation_dim = env.observation_space.shape

# creating own session to use across all the Keras/Tensorflow models we are using
sess = tf.Session()

# Replay memory to store experiances of the model with the environment
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# Our models to solve the mountaincar problem.
agent = DQNAgent(sess, action_dim, observation_dim)


def train_dqn_agent():
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
    X_cur_states = []
    X_next_states = []
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        X_cur_states.append(cur_state)
        X_next_states.append(next_state)
    
    X_cur_states = np.array(X_cur_states)
    X_next_states = np.array(X_next_states)
    
    # action values for the current_states
    cur_action_values = agent.model.predict(X_cur_states)
    # action values for the next_states taken from our agent (Q network)
    next_action_values = agent.model.predict(X_next_states)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        if not done:
            # Q(st, at) = rt + DISCOUNT * max(Q(s(t+1), a(t+1)))
            cur_action_values[index][action] = reward + DISCOUNT * np.amax(next_action_values[index])
        else:
            # Q(st, at) = rt
            cur_action_values[index][action] = reward
    # train the agent with new Q values for the states and the actions
    agent.model.fit(X_cur_states, cur_action_values, verbose=0)

max_reward = -999999
for episode in range(EPISODES):
    cur_state = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:
        episode_length += 1
        # set VISUALIZATION = True if want to see agent while training. But makes training a bit slower.
        if VISUALIZATION:
            env.render()

        if(np.random.uniform(0, 1) < EPSILON):
            # Take random action
            action = np.random.randint(0, action_dim)
        else:
            # Take action that maximizes the total reward
            action = np.argmax(agent.model.predict(np.expand_dims(cur_state, axis=0))[0])

        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if done and episode_length < 200:
            # If episode is ended the we have won the game. So, give some large positive reward
            reward = 250 + episode_reward
            # save the model if we are getting maximum score this time
            if(episode_reward > max_reward):
                agent.model.save_weights(str(episode_reward)+"_agent_.h5")
        else:
            # In oher cases reward will be proportional to the distance that car has travelled 
            # from it's previous location + velocity of the car
            reward = 5*abs(next_state[0] - cur_state[0]) + 3*abs(cur_state[1])
            
        # Add experience to replay memory buffer
        replay_memory.append((cur_state, action, reward, next_state, done))
        cur_state = next_state
        
        if(len(replay_memory) < MINIMUM_REPLAY_MEMORY):
            continue
        
        train_dqn_agent()


    if(EPSILON > MINIMUM_EPSILON and len(replay_memory) > MINIMUM_REPLAY_MEMORY):
        EPSILON *= EPSILON_DECAY

    # some bookkeeping.
    max_reward = max(episode_reward, max_reward)
    print('Episode', episode, 'Episodic Reward', episode_reward, 'Maximum Reward', max_reward, 'EPSILON', EPSILON)
