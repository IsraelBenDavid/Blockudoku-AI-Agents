import abc
from constants import *

class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def get_action(self, game_state):
        return

    def stop_running(self):
        pass
class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        # self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    def evaluation_function(self, game_state):
        return game_state.score

    @abc.abstractmethod
    def get_action(self, game_state):
        return





class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""

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
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
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
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
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