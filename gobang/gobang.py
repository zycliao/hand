# -*- coding: utf-8 -*-

import time
from chessboard import chessboard
from evaluation import evaluation
from searcher import searcher
from gamemain import gamemain

SIZE = 19  # The number of locations/cross-points in row or column
TEXT = '  A B C D E F G H I J K L M N O P Q R S'
# SIZE = 15
# TEXT = '  A B C D E F G H I J K L M N O'

# ----------------------------------------------------------------------
# psyco speedup
# ----------------------------------------------------------------------


def psyco_speedup():
    try:
        import psyco
        psyco.bind(chessboard)
        psyco.bind(evaluation)
    except:
        pass
    return 0

psyco_speedup()


# ----------------------------------------------------------------------
# testing case
# ----------------------------------------------------------------------

if __name__ == '__main__':
    gamemain(SIZE, TEXT)



