from Coach import Coach
from ninemensmorris.NineMensMorrisGame import NineMensMorrisGame as Game
from ninemensmorris.tensorflow.NNet import NNetWrapper as nn
#from othello.tensorflow.NNet import NNetWrapper as nn
#from othello.OthelloGame import OthelloGame as Game
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': '/mnt/data/morris/training/1/',
    'load_model': False,
    'load_folder_file': ('./training/morris', 'checkpoint_2.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    g = Game(ignore_board_repetitions=True)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
