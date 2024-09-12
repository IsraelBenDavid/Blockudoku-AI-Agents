import argparse
import numpy as np
from PlayAgents import human_play, run_previous_work, run_random_game, run_games
from MinMaxAgent import MinmaxAgent, AlphaBetaAgent, ExpectimaxAgent
from PolicyGradientAgent import PolicyGradientAgent
from Constants import POLICY_GRADIENT_WEIGHTS_PATH


def main():
    agent_mapping = {
        'MiniMax': MinmaxAgent,
        'AlphaBeta': AlphaBetaAgent,
        'Expectimax': ExpectimaxAgent,
        'PolicyGradient': PolicyGradientAgent,
        'None': None
    }

    parser = argparse.ArgumentParser(description='Blockudoku')

    games = ['Manual', 'AI_agent', 'Random', 'Baseline']
    agents = ['MiniMax', 'AlphaBeta', 'Expectimax', 'PolicyGradient']

    parser.add_argument('game', help='Type of play to run.', default='Manual', type=str, choices=games)
    parser.add_argument('-ba', '--basic_agent', choices=agents, help='Basic agent.', default='AlphaBeta', type=str)
    parser.add_argument('-bd', '--basic_depth', help='Basic agent max searching depth.', default=1, type=int)
    parser.add_argument('-sa', '--smart_agent', choices=agents, help='Smart agent.', default="None", type=str)
    parser.add_argument('-sd', '--smart_depth', help='Smart agent max searching depth.', default=2, type=int)
    parser.add_argument('-n', '--num_of_games', help='The number of games to run.', default=1, type=int)
    parser.add_argument('-d', '--display', help='Display the game or not.', action='store_true')
    parser.add_argument('-t', '--threshold', help='Threshold (switch from basic to smart).', default=8, type=int)

    args = parser.parse_args()

    if args.game == 'Manual':
        human_play()
        return
    elif args.game == 'Baseline':
        run_previous_work()
        return
    elif args.game == 'Random':
        run_random_game()
        return
    elif args.game == 'AI_agent':
        basic_agent_class = agent_mapping[args.basic_agent]
        if args.basic_agent == 'PolicyGradient':
            basic_agent = basic_agent_class()
            basic_agent.load_model(POLICY_GRADIENT_WEIGHTS_PATH)
        else:
            basic_agent = basic_agent_class(args.basic_depth)

        smart_agent_class = agent_mapping[args.smart_agent]
        if args.smart_agent == 'PolicyGradient':
            smart_agent = smart_agent_class()
            smart_agent.load_model(POLICY_GRADIENT_WEIGHTS_PATH)
        elif args.smart_agent == "None":
            smart_agent = None
        else:
            smart_agent = smart_agent_class(args.smart_depth)
    else:
        return
    if args.display:
        print("NOTE: Playing time is longer when displaying so that you could see the placements.")
    run_games(args.num_of_games, basic_agent, smart_agent, args.threshold, args.display, render_time=0.5)


if __name__ == '__main__':
    main()
    input("Press Enter to continue...")
