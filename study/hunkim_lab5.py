import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector): #https://gist.github.com/stober/1943451
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make("FrozenLake-v0")

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
#Learning rate
learning_rate = .85
#Discount factor
dis = 0.99
#Set learning parameters
num_episodes = 2000

rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    e = 1. / ((i // 100) + 1)

    #The Q-Table learning algorithm
    while not done:
        #Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state,:] + np.random.rand(1, env.action_space.n) / (i+1))

        #Get new state and reward from environment
        new_state, reward, done,_ = env.step(action)

        #Update Q-Table with new knowledge using learning rate
        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state,:]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: ", str(sum(rList)/num_episodes))
print("Final Q-Talbe Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()