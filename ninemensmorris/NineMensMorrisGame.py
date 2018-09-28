from __future__ import print_function
import sys
from collections import defaultdict

from functools import lru_cache
sys.path.append('..')
from Game import Game
from .NineMensMorrisLogic import Board
import numpy as np


class NineMensMorrisGame(Game):
    def __init__(self):
        super().__init__()
        self.boardstateOccurances = defaultdict(set)
        self.turn = 0

    def getInitBoard(self):
        # return initial board (numpy board)
        return Board().toTensor()

    def getBoardSize(self):
        # (y,x,z) z being the rings from inside out plus the additional planes for the specific states
        return Board().toTensor().shape

    def getActionSize(self):
        return Board.actionSpaceCardinality

    def getNextState(self, board: np.ndarray, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(board.copy())
        nextPlayer = b.executeAction(action, player)
        # this function gets queried very often, if the board has not changed since last query, it is not a new state!
        self.boardstateOccurances[repr(b.board)].add(b.turn)
        b.identicalStatesCount = len(self.boardstateOccurances[repr(b.board)])
        # return after state and next player
        return (b.toTensor(), nextPlayer)

    def getValidMoves(self, board: np.ndarray, player):
        # return a fixed size binary vector
        return Board(board.copy()).getLegalMoves(player)

    def getGameEnded(self, board: np.ndarray, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        return Board(board.copy()).hasWon(player)

    def getCanonicalForm(self, board: np.ndarray, player):
        # return state if player==1, else return -state if player==-1
        b = Board(board.copy())
        if player == -1:
            b.board *= -1
            b.playerWithTurn *= -1
            b.whitePrisonerCount, b.blackPrisonerCount = b.blackPrisonerCount, b.whitePrisonerCount
        return b.toTensor()

    def getSymmetries(self, board: np.ndarray, pi):
        """
        mirror on south east diagonal and rotate by multiples of 90 degrees
        :param board:
        :param pi:
        :return:
        """
        pi_board = np.reshape(pi, (8, 3))
        l = []
        for i in range(4):
            newB = np.rot90(board, i)
            newPi = np.roll(pi_board, -2 * i, axis=0)
            for j in [False, True]:
                if j:
                    # flip along top left rigth bottom diagonal
                    newB = np.rot90(np.flip(newB, 0), -1)
                    newPi = np.concatenate((newPi[:1, :], newPi[:0:-1, :]), axis=0)
                l.append((newB, list(newPi.ravel())))
        return l

    def stringRepresentation(self, board):
        # used for hashing in the MCTS
        return repr(Board(board))
