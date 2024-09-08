# TODO: optional: add more state

import pygame as pg
from GridCell import *
from Shape import *
import random
import numpy as np
import time
from collections import deque

MOVEMENT_PUNISHMENT = -1 #* 0.5
INVALID_MOVEMENT_PUNISHMENT = -10 * 100
INVALID_PLACEMENT_PUNISHMENT = -10 * 100
LOSE_PUNISHMENT = 0
REWARD_PLACEMENT = 10 * 50
REPEAT_MOVEMENT_PUNISHMENT = -10 * 10


class Space:

    def __init__(self, n, sample):
        self.n = n
        self.sampleV = sample

    def sample(self):
        return self.sampleV


class Blockudoku:

    def __init__(self):
        self.screen = None
        self.window_size = pg.Vector2(450, 700)
        self.board_loc = pg.Vector2(1, 90)
        self.board_size = pg.Vector2(self.window_size.x - 2, self.window_size.x)
        self.cell_size = self.window_size.x // 9
        self.grid = []
        self.score = 0
        self.cleared_recently = False
        self.lost = False
        self.state = np.zeros((9, 9, 2))

        for r in range(9):
            self.grid.append([])
            for c in range(9):
                self.grid[r].append(GridCell(r, c))

        self.current_shape = Shape()
        self._calculateState()

        self.action_space = Space(5, 0)
        self.observation_space = Space(2 ** len(self.state), self.state)


        self.action_history = deque(maxlen=2)  # Store the last two actions
        self.opposite_actions = {1: 3, 2: 4, 3: 1, 4: 2}  # Map opposite actions (Right-Left, Down-Up)

        # possible states:
        # * invalid cells
        # * blocks that will be cleared if shape is placed
        # * cells' boarders

    def seed(self, seed):
        random.seed(seed)

    def setScreen(self, screen):
        self.screen = screen

    def reset(self):
        self.score = 0
        self.cleared_recently = False
        self.lost = False
        self.current_shape = Shape()
        for row in range(9):
            for col in range(9):
                self.grid[row][col].empty = True
        self.state = np.zeros(self.state.shape)
        self._calculateState()
        return self.state

    def play(self):
        running = True

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    self.reset()
                if event.key == pg.K_SPACE:
                    self.step(0)
                if event.key == pg.K_RIGHT:
                    self.step(1)
                if event.key == pg.K_DOWN:
                    self.step(2)
                if event.key == pg.K_LEFT:
                    self.step(3)
                if event.key == pg.K_UP:
                    self.step(4)

        self.screen.fill((255, 255, 255))

        self._drawCells(self.screen, self.grid, self.cell_size, self.board_loc)
        self.current_shape.draw(self.screen, self.board_loc, self.cell_size, self.grid)
        self._drawBorders(self.screen, self.cell_size, self.board_loc, self.board_size)
        self._displayScore(self.screen)

        pg.display.flip()
        return running and not self.lost

    def render(self, mode='human'):
        board = np.zeros((9, 9))

        for row in range(9):
            for col in range(9):
                if not self.grid[row][col].empty:
                    board[row][col] = 1

        for block in self.current_shape.blocks:
            b_row = self.current_shape.row + block[0]
            b_col = self.current_shape.col + block[1]
            if board[b_row][b_col] == 0:
                board[b_row][b_col] = 2
            else:
                board[b_row][b_col] = 3

        for row in range(9):
            if row % 3 == 0:
                print("+-----+-----+-----+")
            for col in range(9):
                if col % 3 == 0:
                    print("|", end="")
                else:
                    print(":", end="")
                if board[row][col] == 0:
                    print(" ", end="")
                elif board[row][col] == 1:
                    print("\033[0;30;44m ", end="\033[0;0m")
                elif board[row][col] == 2:
                    print("\033[0;30;42m ", end="\033[0;0m")
                else:
                    print("\033[0;30;41m ", end="\033[0;0m")
            print("|")

        print("+-----+-----+-----+")

        if self.screen is not None:
            self.drawGameHeadless()
        return True

    def drawGameHeadless(self):
        running = True

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        self.screen.fill((255, 255, 255))

        self._drawCells(self.screen, self.grid, self.cell_size, self.board_loc)
        self.current_shape.draw(self.screen, self.board_loc, self.cell_size, self.grid)
        self._drawBorders(self.screen, self.cell_size, self.board_loc, self.board_size)
        self._displayScore(self.screen)

        pg.display.flip()

        return running

    def get_agent_legal_actions(self):  # legal actions are the valid places to put
        legal_actions = []
        if self.current_shape.isPlaceable(self.grid):
            legal_actions.append(0)
        if self.current_shape.col + self.current_shape.width < 9:
            legal_actions.append(1)
        if self.current_shape.row + self.current_shape.height < 9:
            legal_actions.append(2)
        if self.current_shape.col > 0:
            legal_actions.append(3)
        if self.current_shape.row > 0:
            legal_actions.append(4)
        return legal_actions

    def step(self, action):
        if action == 0:  # place
            valid = self.current_shape.place(self.grid)
            if valid:
                reward = self._blockPlaced() + REWARD_PLACEMENT
                self.action_history.clear()

            else:
                reward = INVALID_PLACEMENT_PUNISHMENT

        else:
            if action == 1:
                valid = self.current_shape.moveRight()
            elif action == 2:
                valid = self.current_shape.moveDown()
            elif action == 3:
                valid = self.current_shape.moveLeft()
            else:
                valid = self.current_shape.moveUp()

            if valid:
                reward = MOVEMENT_PUNISHMENT

                # Check if the current action cancels out the previous action
                if len(self.action_history) > 0 and self.opposite_actions.get(action) == self.action_history[-1]:
                    reward = REPEAT_MOVEMENT_PUNISHMENT  # Penalize for repeating actions that cancel each other
            else:
                reward = INVALID_MOVEMENT_PUNISHMENT

                # Update action history
            self.action_history.append(action)

        self._calculateState()
        return self.state, reward, self.lost

    def _calculateState(self):
        # layer 1: filled cells
        for row in range(9):
            for col in range(9):
                if self.grid[row][col].empty:
                    self.state[row][col][0] = 0
                else:
                    self.state[row][col][0] = 1

        # layer 2: current shape
        self.state[:, :, 1] = np.zeros(self.state[:, :, 1].shape)
        for block in self.current_shape.blocks:
            self.state[self.current_shape.row + block[0]][self.current_shape.col + block[1]][1] = 1

    def _scoreBoard(self):
        cleared_blocks = []

        # check for vertical lines
        for row in range(9):
            cleared = True
            for col in range(9):
                if self.grid[row][col].empty:
                    cleared = False
                    break

            if cleared:
                cleared_blocks += self.grid[row]

        # check for horizontal lines
        for col in range(9):
            cleared = True
            for row in range(9):
                if self.grid[row][col].empty:
                    cleared = False
                    break

            if cleared:
                cleared_blocks += [grid_row[col] for grid_row in self.grid]

        # check for cleared squares
        for square_row in range(0, 9, 3):
            for square_col in range(0, 9, 3):
                cleared = True
                for row in range(3):
                    for col in range(3):
                        if self.grid[square_row + row][square_col + col].empty:
                            cleared = False
                            break

                if cleared:

                    for row in range(3):
                        for col in range(3):
                            cleared_blocks.append(self.grid[square_row + row][square_col + col])

        # give score
        reward = 0
        if len(cleared_blocks) > 0:
            if len(cleared_blocks) <= 18:
                reward += len(cleared_blocks) * 2
            else:
                reward += len(cleared_blocks) * 4

            for cleared_block in cleared_blocks:
                cleared_block.empty = True

        return reward

    def _blockPlaced(self):
        reward = 0
        reward += self._scoreBoard()
        if reward > 0:
            if self.cleared_recently:
                reward += 9
            self.cleared_recently = True
        else:
            self.cleared_recently = False

        reward += self.current_shape.remainingBlocks(self.grid)
        self.score += reward

        self.current_shape = Shape()
        if not self.current_shape.validSpaceExists(self.grid):
            self.lost = True
            reward -= LOSE_PUNISHMENT

        return reward

    def _displayScore(self, screen):
        font = pg.font.SysFont(None, 44)
        if self.lost:
            color = (255, 0, 0)
        else:
            color = (0, 0, 0)
        img = font.render('Score: ' + str(self.score), True, color)
        screen.blit(img, (self.window_size.x / 2 - 60, 37))

    def _drawCells(self, screen, grid, cell_size, board_loc):
        for r in range(9):
            for c in range(9):
                grid[r][c].draw(screen, board_loc, cell_size)

    def _drawBorders(self, screen, cell_size, board_loc, board_size):
        color = (0, 0, 0)

        rect = pg.Rect(board_loc.x + cell_size * 3, board_loc.y, cell_size * 3, board_size.y)
        pg.draw.rect(screen, color, rect, 2)
        rect = pg.Rect(board_loc.x, board_loc.y + cell_size * 3, board_size.x, cell_size * 3)
        pg.draw.rect(screen, color, rect, 2)

        rect = pg.Rect(board_loc.x, board_loc.y, board_size.x, board_size.y)
        pg.draw.rect(screen, color, rect, 3)


def run_previous_work():
    from baseline_agent import BaselineAgent
    game = Blockudoku()

    pg.init()

    screen = pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)])

    game.setScreen(screen)
    agent = BaselineAgent(game, 1, 0.001, 0.995)
    agent.load_model("checkpoints/baseline/model.pth")

    # game.render()
    running = True
    i = 0
    while running:
        i += 1
        if i % 100 == 0: return
        time.sleep(0.02)
        game.step(agent.test_action(game.state))
        # running = game.play()
        game.drawGameHeadless()
    pg.quit()

def play():
    game = Blockudoku()
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
    play()
    run_previous_work()