import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Iterable

from bidict import bidict

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

    def __init__(self, boardtensor=None):
        # (y,x,z)
        self.board = np.zeros((3, 3, 3)) if boardtensor is None else boardtensor[:, :, 0:3]
        self.isSelectingStone = False if boardtensor is None else bool(boardtensor[0, 0, 3])
        self.isSelectingToPosition = True if boardtensor is None else bool(boardtensor[0, 0, 4])
        self.isImprisoning = False if boardtensor is None else bool(boardtensor[0, 0, 5])
        self.canFly = True if boardtensor is None else bool(boardtensor[0, 0, 6])
        self.canOpponentFly = True if boardtensor is None else bool(boardtensor[0, 0, 7])
        self.identicalStatesCount = 0 if boardtensor is None else boardtensor[0, 0, 8]
        self.turnsWithoutMills = 0 if boardtensor is None else boardtensor[0, 0, 9]
        self.playerWithTurn = 1 if boardtensor is None else boardtensor[0, 0, 10]  # white player starts

    @staticmethod
    def _create_const_plane(value):
        return np.full((3, 3, 1), int(value))

    def toTensor(self) -> np.ndarray:
        return np.concatenate((self.board, *map(self._create_const_plane,
                                                [self.isSelectingStone, self.isSelectingToPosition, self.isImprisoning,
                                                 self.canFly, self.canOpponentFly, self.identicalStatesCount,
                                                 self.turnsWithoutMills, self.playerWithTurn])), axis=2)

    def __str__(self):
        with open(folder / "board.txt") as file:
            board_to_print = [list(line.strip()) for line in file]
        for (y, x, z), val in np.ndenumerate(self.board):
            stridex = ((z + 1) * 3 + z) + 1
            offsetx = 12 - stridex
            stridey = ((z + 1) + z) + 1
            offsety = 6 - stridey
            if val != 0:
                board_to_print[offsety + y * stridey][offsetx + x * stridex] = "○" if val > 0 else "●"
        return "\n".join(["".join(line) for line in board_to_print])

    def __repr__(self):
        return str(self.toTensor().tostring())

    def hasWon(self, player):
        """
        :param player:
        :return: 1 if has one, -1 if has lost, 0 if game not ended, small value if draw
        """
        if self.identicalStatesCount == 3 or self.turnsWithoutMills == 50:
            return .1  # draw
        if self.isImprisoning:
            # if opponent has only 3 stones left and and its my turn to imprison 
            if player == self.playerWithTurn and np.where(self.board >= -1 * player)[0].shape[0] <= 3:
                return 1
            # if opponents turn and i have only 3 stones left 
            elif player != self.playerWithTurn and np.where(self.board >= player)[0].shape[0] <= 3:
                return -1
        # TODO if any of the players cannot move, the player imobilized player has lost (maybe use np.count_nonzero(getlegalmoves)>0)
        return 0

    def executeAction(self, action, player):
        """
        if the action triggers a new subturn for the player we will return the input player, otherwise the opponent is returned
        sideeffect: changes board stage after action
        :param action:
        :param player:
        :return: nextPlayer
        """
        mayImprison = False
        if self.isSelectingStone:
            # highlight selected stone by doubleing its value
            pass
        if self.isSelectingToPosition:
            # remove highlited stone
            # place new stone at "to"-position
            # if mill closed and opponent has more than 3 stones or he has stones that are not inside a mill, he may imprison
            pass

        # TODO at the end of the turn (all subturns for the player) count in default dict how often this constellation was already there
        # if a mill was closed after flying or moving, the turn ends after imprisoning, otherwise it ends after

    def _getFreeNeighbourFields(self, ringpos: int, ringindex: int) -> Iterable:
        freeNeigbourFields = []
        if ringpos % 2 == 0:
            # corner, cant move ring up or down
            for offset in [1, -1]:
                y, x = Board.actionToPos[ringpos + offset]
                if self.board[y, x, ringindex] == 0:
                    freeNeigbourFields.append((y, x, ringindex))
        for offset in [1, -1]:
            y, x = Board.actionToPos[ringpos]
            # if the new ringindex is within board boundaries and field is empty
            if 0 <= ringindex + offset < 3 and self.board[y, x, ringindex] == 0:
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

    def getLegalMoves(self, player):
        legalMoves = np.zeros((8, 3))
        if self.isSelectingStone:
            # subturn 1 in phase 2&3: pick any own stone that can be moved
            for (y, x, z) in zip(*np.where(np.abs(self.board + player) >= 2)):  # filter own stones
                # if he can fly, all stones can be moved, otherwise if only stones with a free ajecent field are valid
                ringpos = Board.actionToPos.inv[(y, x)]
                if self.canFly or self._getFreeNeighbourFields(ringpos, z):
                    legalMoves[ringpos, z] = 1
        elif self.isSelectingToPosition:
            if self.canFly:
                # every free board position
                for (y, x, z) in zip(*np.where(self.board == 0)):
                    # leave out the position of the board, as it has no real interpretation
                    if not y == x == 1:
                        legalMoves[Board.actionToPos.inv[(y, x)], z] = 1
            else:
                # moving to
                # find selected stone, check if there are ajecent fields
                # assumption: at any given time only one stone is active and it is the stone of the current player, because after each complete turn there are no selected stones left
                y, x, z = self.board[np.abs(self.board) == 2]
                ringpos = Board.actionToPos.inv[(y, x)]
                for action in self._getFreeNeighbourFields(ringpos, z):
                    legalMoves[action] = 1
        elif self.isImprisoning:
            opponents_stones = list(zip(*np.where(self.board - player == 0)))
            # we should not get into the imprison state if the opponend has only 3 stones, because than the player has already won!!!
            assert len(opponents_stones) > 3
            for (y, x, z) in opponents_stones:
                ringpos = Board.actionToPos.inv[(y, x)]
                if not self._isInMill(ringpos, z):
                    legalMoves[ringpos, z] = 1
        else:
            assert False

        return legalMoves.ravel()
