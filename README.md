# Blockudoku AI Agents
## Introduction

This project implements various AI agents to play Blockudoku, a puzzle game that combines elements of Sudoku and block puzzles. The objective of the game is to place blocks on a 9x9 grid, filling up rows, columns, or 3x3 squares to clear them from the board. By developing intelligent agents using different AI techniques, this project aims to explore and compare various algorithms in reinforcement learning and search strategies, providing insights into their effectiveness in solving complex problems.

## Table of Contents
+ [Project Overview](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#project-overview)
  + [Agents](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#agents)
+ [File Structure](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#file-structure)
+ [Installation](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#installation)
  + [Prerequisites](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#prerequisites)
  + [Dependencies](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#dependencies)
  + [Steps](https://github.cs.huji.ac.il/israelbd/Blockudoku-ai/edit/master/README.md#steps)
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

## File Structure

+ [`main.py`](main.py): Entry point of the application. Parses command-line arguments and initializes agents and game settings.

+ [`Engine.py`](Engine.py): Defines the game environment, including logic, state representation, and rendering.

+ [`BaselineEngine.py`](BaselineEngine.py): Variant of the game engine used by the BaselineAgent.

+ [`BaselineAgent.py`](BaselineAgent.py): Implements the DQN agent using PyTorch.

+ [`PolicyGradientAgent.py`](PolicyGradientAgent.py): Implements the Policy Gradient agent using PyTorch.

+ [`PolicyNetwork.py`](PolicyNetwork.py): Defines the neural network architecture for the PolicyGradientAgent.

+ [`MinMaxAgent.py`](MinMaxAgent.py): Contains Minimax-based agents, including AlphaBetaAgent and ExpectimaxAgent.

+ [`PlayAgents.py`](PlayAgents.py): Provides functions to run games with different agents and handle user interaction.

+ [`GridCell.py`](GridCell.py): Defines properties and behaviors of individual cells in the game grid.

+ [`Shape.py`](Shape.py): Defines the shapes that appear in the game and their interactions with the grid.

+ [`ShapesStructure.py`](ShapesStructure.py): Contains structures and possible orientations of all shapes used in the game.

+ [`Constants.py`](Constants.py): Contains global constants and configurations used throughout the project.

+ [`KerasBaselineAgent.py`](KerasBaselineAgent.py): Earlier DQN implementation using Keras (TensorFlow).

+ [`README.md`](README.md): Documentation and instructions for the project.

## Installation
### Prerequisites

+ **Python 3.6** or higher
+ **pip** package installer

### Dependencies

+ `numpy`
+ `pygame`
+ `torch` (PyTorch)
+ `tensorflow` and `keras` (for `KerasBaselineAgent.py`)
+ `matplotlib` (optional, for plotting and visualization)

### Steps

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/blockudoku-ai.git
cd blockudoku-ai
```

2. **Create a Virtual Environment** (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install Required Packages**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install packages manually:

```bash
pip install numpy pygame torch tensorflow keras matplotlib
```

4. **Prepare Model Weights**
+ For `BaselineAgent` and `PolicyGradientAgent`, ensure that the model weight files are placed in the appropriate `checkpoints/` directory as specified in `Constants.py`.
+ Download pre-trained models or train new models using the provided training scripts.

## Usage
### Command-Line Arguments

+ `game`: Type of play to run. Options:
  + `Manual`: Play the game manually.
  + `AI_agent`: Play the game using AI agents.
  + `Random`: Play the game with random actions.
  + `Baseline`: Run the baseline models.
+ `-ba`, `--basic_agent`: Basic agent to use. Options:
  + `MiniMax`
  + `AlphaBeta`
  + `Expectimax`
  + `PolicyGradient`
+ `-bd`, `--basic_depth`: Depth for the basic agent's search algorithm. Default: `1`.
+ `-sa`, `--smart_agent`: Smart agent to use when the number of valid actions is below a threshold. Options are the same as `--basic_agent`. Default: `None`.
+ `-sd`, `--smart_depth`: Depth for the smart agent's search algorithm. Default: `2`.
+ `-n`, `--num_of_games`: Number of games to run. Default: `1`.
+ `-d`, `--display`: Display the game GUI. Use this flag to enable rendering.
+ `-t`, `--threshold`: Threshold for switching from the basic agent to the smart agent. Default: `8`.

### Examples

+ **Run the Game Manually**
```bash
python main.py Manual
```

+ **Run the Game with the AlphaBeta Agent at Depth 2 and Display the Game**
```bash
python main.py AI_agent -ba AlphaBeta -bd 2 -d
```

+ **Run 10 Games with the Policy Gradient Agent Without Rendering**
```bash
python main.py AI_agent -ba PolicyGradient -n 10
```

+ **Run the Baseline Model**
```bash
python main.py Baseline
```

### Additional Notes

+ **Rendering:** Enabling rendering with the -d flag will slow down the game due to graphical processing. For faster execution, especially during training, run the game without rendering.

+ **Model Weights:** Ensure that the model weights are correctly loaded. Paths are defined in `Constants.py`. If you don't have pre-trained models, you may need to train the agents first.

+ **Threshold Parameter:** The `-t` or --threshold parameter determines when the game switches from using the basic agent to the smart agent based on the number of valid actions.
