import pygame as pg
import random
from ShapesStructure import *


class Shape:
    def __init__(self, shape_ID=None, row=0, col=0):

        if shape_ID is None:
            shape_ID = random.randrange(0, TOTAL_SHAPES)
        self.row = row
        self.col = col
        shape = ShapesStructure().shapes[shape_ID]
        self.blocks = shape["blocks"]
        self.width = 0
        self.height = 0
        for block in self.blocks:
            if self.height < block[0] + 1:
                self.height = block[0] + 1
            if self.width < block[1] + 1:
                self.width = block[1] + 1

        self.orientations = shape["orientations"]
        self.placed = False

        self.rotate()

    def draw(self, screen, board_loc, cell_size, grid):
        if self.placed:
            return
        border_color = (0, 150, 0)

        for block in self.blocks:
            block_row = self.row + block[0]
            block_col = self.col + block[1]
            color = (0, 255, 0)
            if not grid[block_row][block_col].empty:
                color = (255, 0, 0)

            x = block_col * cell_size
            y = block_row * cell_size
            rect = pg.Rect(board_loc.x + x + 1, board_loc.y + y + 1, cell_size - 1, cell_size - 1)
            pg.draw.rect(screen, color, rect, 0)
            pg.draw.rect(screen, border_color, rect, 1)

    def rotate(self, n_times=None):

        if n_times is None:
            n_times = random.randrange(0, self.orientations)

        for _ in range(n_times):
            temp = self.width
            self.width = self.height
            self.height = temp
            for block_i in range(len(self.blocks)):
                self.blocks[block_i] = (self.blocks[block_i][1], self.blocks[block_i][0])

        minRow = self.blocks[0][0]
        minCol = self.blocks[0][1]
        for block in self.blocks:
            if block[0] < minRow:
                minRow = block[0]
            if block[1] < minCol:
                minCol = block[1]

        for block_i in range(len(self.blocks)):
            self.blocks[block_i] = (self.blocks[block_i][0]-minRow, self.blocks[block_i][1]-minCol)

    def moveRight(self):
        if self.col + self.width < 9:
            self.col += 1
            return True
        else:
            return False

    def moveDown(self):
        if self.row + self.height < 9:
            self.row += 1
            return True
        else:
            return False

    def moveLeft(self):
        if self.col > 0:
            self.col -= 1
            return True
        else:
            return False

    def moveUp(self):
        if self.row > 0:
            self.row -= 1
            return True
        else:
            return False

    def isPlaceable(self, grid):
        # check if the shape is placeble
        for block in self.blocks:
            block_row = self.row + block[0]
            block_col = self.col + block[1]
            if not grid[block_row][block_col].empty:
                return False
        return True

    def place(self, grid):

        if not self.isPlaceable(grid):
            return False

        # place the shape
        self.placed = True
        for block in self.blocks:
            block_row = self.row + block[0]
            block_col = self.col + block[1]
            grid[block_row][block_col].empty = False

        return True

    def validSpaceExists(self, grid):

        # check if the shape is placeble
        for row in range(9-self.height+1):
            for col in range(9-self.width+1):
                placeble = True
                for block in self.blocks:
                    block_row = row + block[0]
                    block_col = col + block[1]
                    if not grid[block_row][block_col].empty:
                        placeble = False
                        break

                if placeble:
                    return True

    def remainingBlocks(self, grid):
        count = 0
        for block in self.blocks:
            block_row = self.row + block[0]
            block_col = self.col + block[1]
            if not grid[block_row][block_col].empty:
                count += 1

        return count
