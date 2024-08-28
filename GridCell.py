import pygame as pg


class GridCell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.empty = True

    def draw(self, screen, board_loc, cell_size):
        border_color = (150, 150, 150)

        color = (255, 255, 255)
        if not self.empty:
            color = (150, 150, 255)
        x = self.col*cell_size
        y = self.row*cell_size
        rect = pg.Rect(board_loc.x + x + 1, board_loc.y + y + 1, cell_size - 1, cell_size - 1)
        pg.draw.rect(screen, color, rect, 0)
        pg.draw.rect(screen, border_color, rect, 1)

