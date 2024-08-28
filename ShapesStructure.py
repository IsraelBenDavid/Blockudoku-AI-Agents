BLOCK1 = 0
LINE2 = 1
DIAGONAL2 = 2
LINE3 = 3
STAIRS3 = 4
DIAGONAL3 = 5
BLOCK4 = 6
LINE4 = 7
L4 = 8
J4 = 9
T4 = 10
S4 = 11
Z4 = 12
LINE5 = 13
L5 = 14
T5 = 15
U5 = 16
PLUS5 = 17

TOTAL_SHAPES = 18


class ShapesStructure:
    def __init__(self):
        self.setShapesIDs()

        self.shapes = [self.block1, self.line2, self.diagonal2, self.line3, self.stairs3, self.diagonal3, self.block4,
                       self.line4, self.l4, self.j4, self.t4, self.s4, self.z4, self.line5, self.l5, self.t5, self.u5,
                       self.plus5]

    def setShapesIDs(self):
        self.block1 = {
            "id": BLOCK1,
            "blocks": [(0, 0)],
            "orientations": 1,
            "name": "Block1"
        }

        self.line2 = {
            "id": LINE2,
            "blocks": [(0, 0), (0, 1)],
            "orientations": 2,
            "name": "Line2"
        }
        self.diagonal2 = {
            "id": DIAGONAL2,
            "blocks": [(0, 0), (1, 1)],
            "orientations": 2,
            "name": "Diagonal2"
        }
        self.line3 = {
            "id": LINE3,
            "blocks": [(0, 0), (0, 1), (0, 2)],
            "orientations": 2,
            "name": "Line3"
        }
        self.stairs3 = {
            "id": STAIRS3,
            "blocks": [(0, 0), (0, 1), (1, 1)],
            "orientations": 4,
            "name": "Stairs3"
        }
        self.diagonal3 = {
            "id": DIAGONAL3,
            "blocks": [(0, 0), (1, 1), (2, 2)],
            "orientations": 2,
            "name": "Diagonal3"
        }
        self.block4 = {
            "id": BLOCK4,
            "blocks": [(0, 0), (0, 1), (1, 0), (1, 1)],
            "orientations": 1,
            "name": "Block4"
        }
        self.line4 = {
            "id": LINE4,
            "blocks": [(0, 0), (0, 1), (0, 2), (0, 3)],
            "orientations": 2,
            "name": "Line4"
        }
        self.l4 = {
            "id": L4,
            "blocks": [(0, 0), (0, 1), (0, 2), (1, 2)],
            "orientations": 4,
            "name": "L4"
        }
        self.j4 = {
            "id": J4,
            "blocks": [(1, 0), (1, 1), (1, 2), (0, 2)],
            "orientations": 4,
            "name": "J4"
        }
        self.t4 = {
            "id": T4,
            "blocks": [(0, 0), (1, 0), (2, 0), (1, 1)],
            "orientations": 4,
            "name": "T4"
        }
        self.s4 = {
            "id": S4,
            "blocks": [(1, 0), (2, 0), (0, 1), (1, 1)],
            "orientations": 4,
            "name": "S4"
        }
        self.z4 = {
            "id": Z4,
            "blocks": [(0, 0), (1, 0), (1, 1), (2, 1)],
            "orientations": 4,
            "name": "Z4"
        }
        self.line5 = {
            "id": LINE5,
            "blocks": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            "orientations": 2,
            "name": "Line5"
        }
        self.l5 = {
            "id": L5,
            "blocks": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
            "orientations": 4,
            "name": "L5"
        }
        self.t5 = {
            "id": T5,
            "blocks": [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
            "orientations": 4,
            "name": "T5"
        }
        self.u5 = {
            "id": U5,
            "blocks": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
            "orientations": 4,
            "name": "U5"
        }
        self.plus5 = {
            "id": PLUS5,
            "blocks": [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
            "orientations": 1,
            "name": "Plus5"
        }