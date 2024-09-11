import argparse
import numpy as np
from play_agent import human_play, run_previous_work
from MinMax_agent import MinmaxAgent, AlphaBetaAgent, ExpectimaxAgent
from policy_gradient_agent import PolicyGradientAgent
from MinMaxEngine import run_random_game, run_games

def main():
    agent_mapping = {
        'MiniMax': MinmaxAgent,
        'AlphaBeta': AlphaBetaAgent,
        'Expectimax': ExpectimaxAgent,
        'PolicyGradient': PolicyGradientAgent
    }

    parser = argparse.ArgumentParser(description='Blockudoku')

    games = ['Manual', 'AI_agent', 'Random', 'Base_line']
    agents = ['MiniMax', 'AlphaBeta', 'Expectimax', 'PolicyGradient']

    parser.add_argument('game', help='Type of play to run.', default='Manual', type=str, choices=games)
    parser.add_argument('-ba', '--basic_agent', choices=agents, help='Basic agent.', default='AlphaBeta', type=str)
    parser.add_argument('-bd', '--basic_depth', help='Basic agent max searching depth.', default=1, type=int)
    parser.add_argument('-sa', '--smart_agent', choices=agents, help='Smart agent.', default='AlphaBeta', type=str)
    parser.add_argument('-sd', '--smart_depth', help='Smart agent max searching depth.', default=2, type=int)
    parser.add_argument('-n', '--num_of_games', help='The number of games to run.', default=1, type=int)
    parser.add_argument('-d', '--display', help='Display the game or not.', action='store_true')
    parser.add_argument('-t', '--threshold', help='Threshold (switch from basic to smart).', default=8, type=int)

    args = parser.parse_args()

    if args.game == 'Manual':
        human_play()
        return
    elif args.game == 'Base_line':
        run_previous_work()
        return
    elif args.game == 'Random':
        run_random_game()
        return
    elif args.game == 'AI_agent':
        basic_agent_class = agent_mapping[args.basic_agent]
        if args.basic_agent == 'PolicyGradient':
            basic_agent = basic_agent_class(None)
            basic_agent.load_model("checkpoints/pg_agent/pg_agent.pth")
        else:
            basic_agent = basic_agent_class(args.basic_depth)

        smart_agent_class = agent_mapping[args.smart_agent]
        if args.smart_agent == 'PolicyGradient':
            smart_agent = smart_agent_class(None)
            smart_agent.load_model("checkpoints/pg_agent/pg_agent.pth")
        else:
            smart_agent = smart_agent_class(args.smart_depth)
    else:
        return

    run_games(args.num_of_games, basic_agent, smart_agent, args.threshold, args.display)


if __name__ == '__main__':
    main()
    input("Press Enter to continue...")