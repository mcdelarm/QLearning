
import numpy as np
import gym
import random


def argmax(arr):
    arr_max = np.max(arr)
    return np.random.choice(np.where(arr == arr_max)[0])


max_steps = 99                # Max steps per episode


def train():
    # Create env
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

    # Create q-table
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))

    # Parameters
    total_episodes = 15000        # Total episodes
    learning_rate = 0.8           # Learning rate to increase learning in earlier episodes
    gamma = 0.95                  # Discounting rate to prioritize immediate rewares

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate to balance exploration and exploitation
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability
    decay_rate = 0.005             # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    # Training
    for episode in range(total_episodes):
        # Reset the environment
        state, _ = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            rand = random.uniform(0, 1)

            # Exploitation
            if rand > epsilon:
                action = argmax(qtable[state, :])

            # Exploration
            else:
                action = env.action_space.sample()

            s_, r, done, _, _ = env.step(action)

            # Use temporal difference equation to update qtable
            qtable[state, action] = qtable[state, action] + learning_rate * (
                        r + gamma * np.max(qtable[s_, :]) - qtable[state, action])

            total_rewards += r

            # Our new state is s_
            state = s_

            # Finish episode if we're dead or reached the goal
            if done == True:
                break

        # Reduce epsilon for less exploration and more exploitation
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)

    print("Score over time: " + str(sum(rewards) / total_episodes))
    print(qtable)
    env.close()
    return qtable


qtable = train()

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode='human')
env.reset()
for episode in range(5):
    state, _ = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future reward given that state
        action = argmax(qtable[state, :])

        new_state, reward, done, _, _ = env.step(action)

        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()

            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()

