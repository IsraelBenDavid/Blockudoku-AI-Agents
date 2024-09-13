import random
import time
import pygame as pg
import BaselineEngine
from BaselineAgent import BaselineAgent

import Engine
from PolicyGradientAgent import PolicyGradientAgent

from Constants import *


def run_previous_work(render=True):
    game = BaselineEngine.Blockudoku()
    if render:
        pg.init()
        screen = pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)])
        game.setScreen(screen)
    agent = BaselineAgent(game, 1, 0.001, 0.995)
    agent.load_model(BASELINE_WEIGHTS_PATH)

    for i in range(100): # steps limit
        if render:
            time.sleep(0.2)
        game.step(agent.test_action(game.state))
        if render:
            game.drawGameHeadless()

    if render:
        pg.quit()

    return game.score


def human_play():
    game = BaselineEngine.Blockudoku()
    pg.init()
    screen = pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)])
    game.setScreen(screen)
    running = True
    while running:
        running = game.play()
    print("Game Over")
    print(f"Your Score: {game.score}")
    time.sleep(2)
    pg.quit()


def run_games(num_episodes, basic_agent, smart_agent=None, threshold=8, render=False, render_time=0.0):
    scores = []
    game = Engine.Blockudoku()
    # game.seed(541)
    if render:
        pg.init()
        game.setScreen(pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)]))
        game.drawGameHeadless()

    for i in range(1, num_episodes + 1):
        game.reset()
        steps = 0
        while True:
            valid_actions = game.get_agent_legal_actions()
            if len(valid_actions) == 0: break
            if 1 < len(valid_actions) <= threshold and smart_agent:
                action = smart_agent.get_action(game)
            else:
                action = basic_agent.get_action(game)

            game.apply_action(action, render=render, render_time=render_time)
            game.apply_opponent_action(random.choice(game.get_opponent_legal_actions()))
            steps += 1
            if render:
                running = game.drawGameHeadless()
                if not running: break

        print(f"episode: {i} \t| score: {game.score} \t| steps: {steps}")
        scores.append(game.score)
    if render:
        pg.quit()
    # Print the results
    print("Minimum score:", min(scores))
    print("Maximum score:", max(scores))
    print("Mean score:", sum(scores) / len(scores))


def run_random_game():
    game = Engine.Blockudoku()
    pg.init()
    game.setScreen(pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)]))
    game.drawGameHeadless()
    running = True
    while running:

        valid_actions = game.get_agent_legal_actions()
        if len(valid_actions) == 0: break

        game.apply_action((random.randint(0, 8), random.randint(0, 8)), render=True)

        random_op_action = random.choice(game.get_opponent_legal_actions())
        game.apply_opponent_action(random_op_action)
        running = game.drawGameHeadless()
    print(f"score: {game.score}")
    pg.quit()


