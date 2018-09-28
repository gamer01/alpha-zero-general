import Arena
from MCTS import MCTS
from ninemensmorris.NineMensMorrisGame import NineMensMorrisGame as Game
from ninemensmorris.NineMensMorrisPlayers import *
from ninemensmorris.tensorflow.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
if __name__ == "__main__":
    g = Game()

    # all players
    rp = RandomPlayer(g).play
    rp2 = RandomPlayer(g).play
    hp = HumanMorrisPlayer(g).play
    """
    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./training/morris/','best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

   
    n2 = NNet(g)
    n2.load_checkpoint('./temp/','checkpoint_5.pth.tar')
    args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    """

    arena = Arena.Arena(rp, rp2, g, display=lambda board: print((Board(board))))
    print(arena.playGames(1, verbose=True))
