import gym
from dqn_agent import DQNAgent
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sys

model_weight_file = sys.argv[1]

sess = tf.Session()
K.set_session(sess)
env = gym.make('MountainCar-v0')

action_dim = env.action_space.n
observation_dim = env.observation_space.shape

# create and load weights of the model
agent = DQNAgent(sess, action_dim, observation_dim)
agent.model.load_weights(model_weight_file)
# Number of episodes in which agent manages to won the game before time is over
episodes_won = 0
# Number of episodes for which we want to test the agnet
TOTAL_EPISODES = 10 

for _ in range(TOTAL_EPISODES):
    cur_state = env.reset()
    done = False
    episode_len = 0
    while not done:
        env.render()
        episode_len += 1
        next_state, reward, done, _ = env.step(np.argmax(agent.model.predict(np.expand_dims(cur_state, axis=0))))
        if done and episode_len < 200:
            episodes_won += 1
        cur_state = next_state
     
print(episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')
