README template:
section 1: brief explanation about the repo - 1. general explanation. 2. brief explanation about the agents.
section 2: files explanation.
section 3: instalation and usage explanation IN DETAIL


# Blockudoku-ai

# Blockudoku AI Game Script - Running Guide
Basic Usage:
python your_script.py [game] [options]

# Game Modes (game)
Manual: Play manually.
AI_agent: Run with AI agents.
Random: Play with random moves.
Base_line: Replay recorded games.

# Options
-ba, --basic_agent: Choose basic agent (MiniMax, AlphaBeta, Expectimax, PolicyGradient). Default: AlphaBeta.
-bd, --basic_depth: Set basic agent's depth. Default: 1.
-sa, --smart_agent: Choose smart agent. Default: AlphaBeta.
-sd, --smart_depth: Set smart agent's depth. Default: 2.
-n, --num_of_games: Number of games to run. Default: 1.
-t, --threshold: Threshold to switch agents. Default: 8.
-d, --display: Show game (True/False). Default: False.
