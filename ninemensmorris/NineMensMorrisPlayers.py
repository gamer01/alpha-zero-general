import numpy as np
from readline import clear_history, add_history

from ninemensmorris.NineMensMorrisLogic import Board

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # always as player one, as the board is in canonical form
        valids = self.game.getValidMoves(board, 1)
        a = np.random.choice(np.where(valids == 1)[0])
        return a


class HumanMorrisPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # board is in canonical form, so we use player 1
        valid = self.game.getValidMoves(board, 1).reshape((8, 3))
        moves = []
        for (ringpos, z), index_valid in np.ndenumerate(valid):
            if index_valid:
                move = "{} {} {}".format(*Board.actionToPos[ringpos], z)
                moves.append(move)
        print("; ".join(moves))
        while True:
            clear_history()
            [add_history(move) for move in moves]
            a = input()

            try:
                y, x, z = [int(x) for x in a.strip().split(' ')]
                if (x != 1 or y != 1) and valid[Board.actionToPos.inv[(y, x)], z]:
                    break
                else:
                    print('Invalid')
            except ValueError:
                print('Invalid')

        return np.ravel_multi_index((Board.actionToPos.inv[(y, x)], z), (8, 3))
