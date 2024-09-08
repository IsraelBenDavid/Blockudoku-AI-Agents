import random
import time
import pygame as pg
import Engine
from baseline_agent import BaselineAgent

import MinMaxEngine
from policy_gradient_agent import PolicyGradientAgent


def run_previous_work(render=True):
    game = Engine.Blockudoku()
    if render:
        pg.init()
        screen = pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)])
        game.setScreen(screen)
    agent = BaselineAgent(game, 1, 0.001, 0.995)
    agent.load_model("checkpoints/baseline/model.pth")

    running = True
    i = 0
    while running:
        i += 1
        if i % 100 == 0: break
        if render:
            time.sleep(0.2)
        game.step(agent.test_action(game.state))
        if render:
            game.drawGameHeadless()

    if render:
        pg.quit()

    return game.score


def run_pg_agent(render=True):
    game = MinMaxEngine.Blockudoku()
    agent = PolicyGradientAgent(None)
    agent.load_model(f"checkpoints/pg_agent/pg_agent.pth")
    if render:
        pg.init()
        game.setScreen(pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)]))
        game.drawGameHeadless()
    running = True
    while running and not game.lost:
        if render:
            time.sleep(0.2)
        action = agent.get_action(game)
        game.apply_action(action, render=render)
        random_op_action = random.choice(game.get_opponent_legal_actions())
        game.apply_opponent_action(random_op_action)
        if render:
            running = game.drawGameHeadless()
    if render:
        pg.quit()
    return game.score


def human_play():
    game = Engine.Blockudoku()
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


if __name__ == "__main__":
    scores = []
    for i in range(10):
        scores.append(run_pg_agent(render=False))

    print(sum(scores) / len(scores))

