import unittest
from numpy.testing import *
import numpy as np

from ninemensmorris.NineMensMorrisGame import NineMensMorrisGame as Game
from ninemensmorris.NineMensMorrisLogic import Board
from ninemensmorris.tensorflow.NNet import NNetWrapper as nn


class BoardTestCase(unittest.TestCase):
    def test_Boardconstructor(self):
        b = Board()
        b2 = Board(b.toTensor())
        self.assertEqual(repr(b), repr(b2))


class GameTestCase(unittest.TestCase):
    def test_canonicalform(self):
        b = Board()
        b.board[2, 2, 1] = -1
        self.assertIsNone(assert_array_equal(Game().getCanonicalForm(b.toTensor(), 1), b.toTensor()))

    def test_symmetries(self):
        b = Board()
        b.board[0, 0, 2] = -1
        b.board[0, 1, 2] = 1
        for (board,pi) in Game().getSymmetries(b.toTensor(),np.arange(25)):
            print(Board(board),pi)


if __name__ == '__main__':
    unittest.main()
