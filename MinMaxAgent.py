import abc
from Constants import *


class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def get_action(self, game_state):
        return

    def stop_running(self):
        pass


class MultiAgentSearchAgent(Agent):
    def __init__(self, depth=2):
        self.depth = depth

    def evaluation_function(self, game_state):
        """
        This evaluation function assesses the quality of a Blockudoku game state by considering the current score,
        the number of empty cells (with emphasis on the central area), and the potential to clear rows, columns,
        or 3x3 grids that are nearly full. It also accounts for placement flexibility by evaluating open spaces that
        could accommodate future pieces and the number of legal moves available to the agent.
        The function combines these factors with weighted importance to generate a score that aids in making
        optimal decisions during gameplay.
        """
        score = game_state.score
        empty_cells = sum(1 for row in game_state.grid for cell in row if cell.empty)
        central_area = sum(1 for r in range(3, 6) for c in range(3, 6) if game_state.grid[r][c].empty)

        # Clearing Potential
        near_clears = 0
        for i in range(9):
            if sum(1 for cell in game_state.grid[i] if not cell.empty) == 8:
                near_clears += 1
            if sum(1 for j in range(9) if not game_state.grid[j][i].empty) == 8:
                near_clears += 1
        for square_row in range(0, 9, 3):
            for square_col in range(0, 9, 3):
                if sum(1 for r in range(3) for c in range(3) if
                       not game_state.grid[square_row + r][square_col + c].empty) == 8:
                    near_clears += 1

        # Placement Flexibility
        open_spaces = sum(1 for row in range(9) for col in range(9)
                          if game_state.grid[row][col].empty and
                          any(game_state.grid[row + i][col + j].empty for i in range(3) for j in range(3)))

        # Placement Opportunities
        possible_placements = len(game_state.get_agent_legal_actions())

        reward = (score * 1
                  + empty_cells * 2
                  + central_area * 3
                  + near_clears * 5
                  + open_spaces * 2
                  + possible_placements * 3)

        return reward

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """

        return self.player_move(game_state, self.depth)['action']

    def player_move(self, game_state, depth):
        actions = game_state.get_legal_actions(agent_index=PLAYER)
        if depth == 0:
            return {"action": STOP_ACTION, "score": self.evaluation_function(game_state)}
        if len(actions) == 0:
            return {"action": actions[0], "score": self.opponent_move(game_state, depth)}
        max_score_action = {"action": STOP_ACTION, "score": float('-inf')}
        for action in actions:
            curr_state = game_state.generate_successor(agent_index=PLAYER, action=action)
            curr_score = self.opponent_move(curr_state, depth)
            if curr_score > max_score_action["score"]:
                max_score_action["action"] = action
                max_score_action["score"] = curr_score
        return max_score_action

    def opponent_move(self, game_state, depth):
        actions = game_state.get_legal_actions(agent_index=OPPONENT)
        min_score_action = float('inf')
        for action in actions:
            curr_state = game_state.generate_successor(agent_index=OPPONENT, action=action)
            curr_score = self.player_move(curr_state, depth - 1)['score']
            if curr_score < min_score_action:
                min_score_action = curr_score
        return min_score_action


class AlphaBetaAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth
        """
        return self.player_move(game_state, self.depth)['action']

    def player_move(self, game_state, depth, alpha=float('-inf'), beta=float('inf')):
        actions = game_state.get_legal_actions(agent_index=PLAYER)
        if depth == 0:
            return {"action": STOP_ACTION, "score": self.evaluation_function(game_state)}
        if len(actions) == 0:
            return {"action": STOP_ACTION, "score": self.opponent_move(game_state, depth, alpha, beta)}
        max_score_action = {"action": STOP_ACTION, "score": float('-inf')}
        for action in actions:
            curr_state = game_state.generate_successor(agent_index=PLAYER, action=action)
            curr_score = self.opponent_move(curr_state, depth, alpha, beta)
            alpha = max(alpha, curr_score)
            if curr_score > max_score_action["score"]:
                max_score_action["action"] = action
                max_score_action["score"] = curr_score
            if beta <= alpha: break
        return max_score_action

    def opponent_move(self, game_state, depth, alpha=float('-inf'), beta=float('inf')):
        actions = game_state.get_legal_actions(agent_index=OPPONENT)
        min_score_action = float('inf')
        for action in actions:
            curr_state = game_state.generate_successor(agent_index=OPPONENT, action=action)
            curr_score = self.player_move(curr_state, depth - 1, alpha, beta)['score']
            beta = min(beta, curr_score)
            if curr_score < min_score_action:
                min_score_action = curr_score
            if beta <= alpha: break
        return min_score_action


class ExpectimaxAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.player_move(game_state, self.depth)['action']

    def player_move(self, game_state, depth):
        actions = game_state.get_legal_actions(agent_index=PLAYER)
        if depth == 0:
            return {"action": STOP_ACTION, "score": self.evaluation_function(game_state)}
        if len(actions) == 0:
            return {"action": STOP_ACTION, "score": self.opponent_expected_move(game_state, depth)}
        max_score_action = {"action": STOP_ACTION, "score": float('-inf')}
        for action in actions:
            curr_state = game_state.generate_successor(agent_index=PLAYER, action=action)
            curr_score = self.opponent_expected_move(curr_state, depth)
            if curr_score > max_score_action["score"]:
                max_score_action["action"] = action
                max_score_action["score"] = curr_score
        return max_score_action

    def opponent_expected_move(self, game_state, depth):
        actions = game_state.get_legal_actions(agent_index=OPPONENT)
        expected_score = 0
        for action in actions:
            curr_state = game_state.generate_successor(agent_index=OPPONENT, action=action)
            curr_score = self.player_move(curr_state, depth - 1)['score']
            expected_score += curr_score / len(actions)
        return expected_score
