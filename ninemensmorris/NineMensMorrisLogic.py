import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Iterable

from bidict import bidict
from itertools import zip_longest

folder = Path(__file__).parent


class Board:
    """
    white stones will be "1", black stones "-1", if a stone is selected for movement it will be 2 and -2 respective.
    """

    """
    three circles starting with the inner most circle, clockwise from x,y position 0,0
    2╶─────────╴5╶─────────╴8
    │           │           │
    │   1╶─────╴4╶─────╴7   │
    │   │       │       │   │
    │   │   0╶─╴3╶─╴6   │   │
    │   │   │       │   │   │
    23─╴22─╴21      9╶─10╶─11
    │   │   │       │   │   │
    │   │  18╶─15╶─12   │   │
    │   │       │       │   │
    │  19╶─────16╶─────13   │
    │           │           │
    20─────────17╶─────────14
    """
    actionSpaceCardinality = 8 * 3
    actionToPos = bidict({k: v for k, v in enumerate(((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)))})
    history_length = 12

    def __init__(self, boardtensor=None):
        n_planes_history = 9 * (Board.history_length-1)
        if boardtensor is None:
            # history includes current state
            self.history = np.zeros((3, 3, n_planes_history+9))

            # last state
            self.board = np.zeros((3, 3, 3))
            self.isSelecting = False
            self.isPlacing = True
            self.isImprisoning = False
            self.whitePrisonerCount = 0
            self.blackPrisonerCount = 0
            self.identicalStatesCount = 1

            # global
            self.turnsWithoutMills = 0
            self.turn = 0
            self.playerWithTurn = 1

        else:
            # history includes current state
            self.history = boardtensor[:, :, :n_planes_history+9]

            # last state
            # (y,x,z)
            self.board = boardtensor[:, :, n_planes_history:n_planes_history + 3]
            self.isSelecting = bool(boardtensor[0, 0, n_planes_history + 3])
            self.isPlacing = bool(boardtensor[0, 0, n_planes_history + 4])
            self.isImprisoning = bool(boardtensor[0, 0, n_planes_history + 5])
            self.whitePrisonerCount = int(boardtensor[0, 0, n_planes_history + 6])
            self.blackPrisonerCount = int(boardtensor[0, 0, n_planes_history + 7])
            self.identicalStatesCount = int(boardtensor[0, 0, n_planes_history + 8])

            # global
            self.turnsWithoutMills = int(boardtensor[0, 0, n_planes_history + 9])
            self.turn = int(boardtensor[0, 0, n_planes_history + 10])
            self.playerWithTurn = int(boardtensor[0, 0, n_planes_history + 11])  # white player starts

    @staticmethod
    def _create_const_plane(value):
        return np.full((3, 3, 1), int(value))

    def toTensor(self) -> np.ndarray:
        if np.array_equal(self.board,self.history[:, :, 9 * (Board.history_length-1):9 * (Board.history_length-1)+3]):
            # state has not changed
            hist = self.history[:, :, :9 * (Board.history_length-1)]
        else:
            # drop last history item (first index)
            hist = self.history[:, :, 9:]

        return np.concatenate((hist, self.board, *map(self._create_const_plane,
                                                              [self.isSelecting, self.isPlacing, self.isImprisoning,
                                                               self.whitePrisonerCount, self.blackPrisonerCount,
                                                               self.identicalStatesCount, self.turnsWithoutMills,
                                                               self.turn,
                                                               self.playerWithTurn])), axis=2)

    def __str__(self):
        with open(folder / "board.txt") as file:
            board_to_print = [list(line.strip()) for line in file]
        for (y, x, z), val in np.ndenumerate(self.board):
            stridex = ((z + 1) * 3 + z) + 1
            offsetx = 12 - stridex
            stridey = ((z + 1) + z) + 1
            offsety = 6 - stridey
            if val != 0:
                board_to_print[offsety + y * stridey][offsetx + x * stridex] = (
                    "◇" if val > 1 else "○") if val > 0 else ("◆" if val < -1 else "●")
        infos = ["{:<26s} {}".format(k + ":", v) for k, v in {
            "State": "selecting" if self.isSelecting else
            "placing" if self.isPlacing else
            "imprisoning" if self.isImprisoning else "error",
            "white prisoner count": self.whitePrisonerCount,
            "black prisoner count": self.blackPrisonerCount,
            "identical positions count": self.identicalStatesCount,
            "turns without mills": self.turnsWithoutMills,
            "turns": self.turn,
            "player in turn": "white" if self.playerWithTurn == 1 else "black"}.items()]
        return "\n".join(b + "  " + i for b, i in
                         zip_longest(["".join(line) for line in board_to_print], infos, fillvalue=""))

    def __repr__(self):
        return str(self.toTensor().tostring())

    def hasWon(self, player):
        """
        :param player:
        :return: 1 if has won, -1 if has lost, 0 if game not ended, small value if draw
        """
        if self.identicalStatesCount >= 3 or self.turnsWithoutMills >= 50:
            return .1  # draw
        if self.isImprisoning:
            # if i have imprisoned 6 stones and its my turn to imprison
            if self._ownPrisonerCount(player) == 6:
                return 1
            # if opponents turn and i have only 3 stones left 
            elif self._opponentPrisonerCount(player) == 6:
                return -1
        if np.count_nonzero(self.getLegalMoves(player) > 0) == 0:
            return -1
        elif np.count_nonzero(self.getLegalMoves(-player) > 0) == 0:
            return 1
        return 0

    def _getFreeNeighbourFields(self, ringpos: int, ringindex: int) -> Iterable:
        freeNeigbourFields = []
        if ringpos % 2 != 0:
            # corner, cant move ring up or down
            for offset in [1, -1]:
                y, x = Board.actionToPos[ringpos]
                # if the new ringindex is within board boundaries and field is empty
                z = ringindex + offset
                if 0 <= z < 3 and self.board[y, x, z] == 0:
                    freeNeigbourFields.append((y, x, z))
        for offset in [1, -1]:
            y, x = Board.actionToPos[(ringpos + offset) % 8]
            if self.board[y, x, ringindex] == 0:
                freeNeigbourFields.append((y, x, ringindex))
        return freeNeigbourFields

    def _isInMill(self, ringpos: int, ringindex: int) -> bool:
        """
        summing value of 3 fields one can easily determine if they contain 3 stones from the same color, by checking if the absolute value is >=3
        :param ringpos:
        :param ringindex:
        :return:
        """
        y, x = Board.actionToPos[ringpos]
        # no corner, check connections to outer rings
        # if not vertically in the middle, check horizontally on the current ring
        # if not horizontally in the middle, check vertically on the current ring
        return ringpos % 2 == 1 and np.abs(np.sum(self.board[y, x, :])) >= 3 \
               or y != 1 and np.abs(np.sum(self.board[y, :, ringindex])) >= 3 \
               or x != 1 and np.abs(np.sum(self.board[:, x, ringindex])) >= 3

    def _ownPrisonerCount(self, player=None):
        if player is None:
            player = self.playerWithTurn
        return self.whitePrisonerCount if player == 1 else self.blackPrisonerCount

    def _opponentPrisonerCount(self, player=None):
        if player is None:
            player = self.playerWithTurn
        return self.blackPrisonerCount if player == 1 else self.whitePrisonerCount

    def getLegalMoves(self, player):
        legalMoves = np.zeros((8, 3))
        if player != self.playerWithTurn:
            # if player would be in phase 1: placing
            if np.count_nonzero(self._ownStonesMask(player)) + self._opponentPrisonerCount(player) < 9:
                for (y, x, z) in zip(*np.where(self.board == 0)):
                    # leave out the position of the board, as it has no real interpretation
                    if not y == x == 1:
                        legalMoves[Board.actionToPos.inv[(y, x)], z] = 1
            else:
                # player is in phase 2 or 3: selecting
                for (y, x, z) in zip(*np.where(self._ownStonesMask(player))):  # filter own stones
                    ringpos = Board.actionToPos.inv[(y, x)]
                    # if he can fly, all stones can be moved, otherwise if only stones with a free adjacent field are valid
                    # since we are in phase 2 or 3, we dont have to count our stones on the board
                    if self._opponentPrisonerCount(player) == 6 or self._getFreeNeighbourFields(ringpos, z):
                        legalMoves[ringpos, z] = 1
        else:
            if self.isSelecting:
                # subturn 1 in phase 2&3: pick any own stone that can be moved
                for (y, x, z) in zip(*np.where(self._ownStonesMask(player))):  # filter own stones
                    ringpos = Board.actionToPos.inv[(y, x)]
                    # if he can fly, all stones can be moved, otherwise if only stones with a free adjacent field are valid
                    # since we are in phase 2 or 3, we dont have to count our stones on the board
                    if self._opponentPrisonerCount(player) == 6 or self._getFreeNeighbourFields(ringpos, z):
                        legalMoves[ringpos, z] = 1
            elif self.isPlacing:
                # can fly to
                if np.count_nonzero(self._ownStonesMask(player)) + self._opponentPrisonerCount(player) < 9 \
                        or self._opponentPrisonerCount(player) == 6:
                    # every free board position
                    for (y, x, z) in zip(*np.where(self.board == 0)):
                        # leave out the position of the board, as it has no real interpretation
                        if not y == x == 1:
                            legalMoves[Board.actionToPos.inv[(y, x)], z] = 1
                else:
                    # moving to
                    # find selected stone, check if there are ajecent fields
                    # assumption: at any given time only one stone is active and it is the stone of the current player, because after each complete turn there are no selected stones left
                    y, x, z = (int(i) for i in np.where(np.abs(self.board) == 2))
                    ringpos = Board.actionToPos.inv[(y, x)]
                    for (y, x, z) in self._getFreeNeighbourFields(ringpos, z):
                        legalMoves[Board.actionToPos.inv[(y, x)], z] = 1
            elif self.isImprisoning:
                opponents_stones = list(zip(*np.where(self._opponentStonesMask(player))))
                # we should not get into the imprison state if the opponend has only 3 stones, because than the player has already won!!!
                assert self._ownPrisonerCount() <= 6
                for (y, x, z) in opponents_stones:
                    ringpos = Board.actionToPos.inv[(y, x)]
                    if not self._isInMill(ringpos, z):
                        legalMoves[ringpos, z] = 1
                if np.count_nonzero(legalMoves) == 0:
                    # all of the opponents stones are in mills, so the player may take stones inside mills!
                    for (y, x, z) in opponents_stones:
                        ringpos = Board.actionToPos.inv[(y, x)]
                        legalMoves[ringpos, z] = 1
            else:
                # here we should never be able to come, so we will make sure that an error is raised
                raise ValueError("The game state is in an illegal state", self, sep="\n")

        return legalMoves.ravel()

    def _ownStonesMask(self, player):
        ownStonesMask = np.abs(self.board + player) >= 2
        return ownStonesMask

    def executeAction(self, action: np.ndarray, player):
        """
        if the action triggers a new subturn for the player we will return the input player, otherwise the opponent is returned
        sideeffect: changes board stage after action
        :param action:
        :param player:
        :return: nextPlayer
        """
        ringpos, z = np.unravel_index(action, (8, 3))
        y, x = Board.actionToPos[ringpos]
        end_of_turn = False
        if self.isSelecting:
            self.isSelecting = False
            self.isPlacing = True
            # highlight selected stone by doubling its value
            self.board[y, x, z] *= 2
        elif self.isPlacing:
            self.isPlacing = False
            # remove highlighted stone
            self.board[np.abs(self.board) == 2] = 0
            # place new stone of players color at "to"-position
            self.board[y, x, z] = player
            # check if "to" position is in a mill
            #  (it has been closed, as otherwise moving a stone there would have been invalid)
            if self._isInMill(ringpos, z):
                self.turnsWithoutMills = 0
                self.isImprisoning = True
            else:
                self.turnsWithoutMills += 1
                end_of_turn = True
        elif self.isImprisoning:
            self.isImprisoning = False
            if player == 1:
                self.whitePrisonerCount += 1
            else:
                self.blackPrisonerCount += 1
            end_of_turn = True
            # remove imprisoned stone
            self.board[y, x, z] = 0
        else:
            # here we should never be able to come, so we will make sure that an error is raised
            raise ValueError("The game state is in an illegal state", self, sep="\n")

        # if end of turn (including all subturns)
        if end_of_turn:
            self.turn += 1
            # inverted, because we are about to flip turns
            if (np.count_nonzero(self._opponentStonesMask(player)) + self._ownPrisonerCount(player)) < 9:
                # opponent is in phase 1 -> placing
                self.isPlacing = True
            else:
                # opponent in phase 2 or 3
                self.isSelecting = True
            self.playerWithTurn *= -1
        return self.playerWithTurn

    def _opponentStonesMask(self, player):
        opponentStonesMask = (self.board + player) == 0
        return opponentStonesMask
