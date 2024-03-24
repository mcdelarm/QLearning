# Q-Learning for FrozenLake-v1 Environment

This repository contains a Python implementation of the Q-learning algorithm applied to the FrozenLake-v1 environment from the OpenAI Gym.

## Introduction

Q-learning is a reinforcement learning technique used to train agents in an environment to make decisions. In this specific implementation, Q-learning is utilized to train an agent to navigate the FrozenLake environment, a grid-world where the agent must find a path from the starting point to the goal without falling into holes.

## Installation

To run the code, make sure you have Python installed on your system along with the following dependencies:

gym
numpy
You can install these dependencies using pip or with your respective IDE

## Usage

To train the agent, execute the train() function defined in the q_learning.py file. This function initializes the Q-table, trains the agent through a specified number of episodes, and returns the learned Q-table.

To visualize the performance of the trained agent, you can run the provided code after training. This code demonstrates the agent's behavior in the FrozenLake environment over a few episodes utilizing the learned Q-table values.

## Code Structure

q_learning.py: Contains the Q-learning algorithm implementation along with the training function.
README.md: This file providing information about the repository.

## Acknowledgements

This implementation is inspired by the materials provided by OpenAI Gym.
The FrozenLake environment is part of the Gym library provided by OpenAI.
