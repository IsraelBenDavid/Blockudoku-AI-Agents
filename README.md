# Blockudoku AI Agents
## Introduction

This project implements various AI agents to play Blockudoku, a puzzle game that combines elements of Sudoku and block puzzles. The objective of the game is to place blocks on a 9x9 grid, filling up rows, columns, or 3x3 squares to clear them from the board. By developing intelligent agents using different AI techniques, this project aims to explore and compare various algorithms in reinforcement learning and search strategies, providing insights into their effectiveness in solving complex problems.

## Table of Contents
+ [Project Overview](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#project-overview)
  + [Agents](https://github.com/)
+ [File Structure](https://github.com/)
+ [Installation](https://github.com/)
  + [Prerequisites](https://github.com/)
  + [Dependencies](https://github.com/)
  + [Steps](https://github.com/)
+ [Usage](https://github.com/)
  + [Command-Line Arguments](https://github.com/)
  + [Examples](https://github.com/)
  + [Additional Notes](https://github.com/)
+ [Contributing](https://github.com/)
  + [Reporting Bugs](https://github.com/)
  + [Requesting Features](https://github.com/)
+ [License](https://github.com/)

## Project Overview
### Agents

The project includes several AI agents implementing different algorithms:
+ BaselineAgent: Utilizes Deep Q-Networks (DQN) with PyTorch to learn optimal policies through experience replay and neural network approximations of Q-values.

+ PolicyGradientAgent: Implements a Policy Gradient method using PyTorch to directly learn a policy that maps states to action probabilities.

+ MinMaxAgent: Contains Minimax-based agents, including Alpha-Beta Pruning and Expectimax, to perform lookahead searches for optimal decision-making in adversarial and stochastic environments.

+ KerasBaselineAgent: An alternative DQN implementation using Keras (TensorFlow), serving as a comparison point for the PyTorch-based BaselineAgent.

These agents interact with the game environment to make decisions, learn from experiences, and improve their gameplay over time. By comparing these different approaches, the project highlights the strengths and weaknesses of each algorithm in the context of the Blockudoku game.
