# -*- coding: utf-8 -*-

from chessboard import chessboard
from searcher import searcher
from evaluation import evaluation


# ----------------------------------------------------------------------
# main game
# ----------------------------------------------------------------------


class GoBang(object):
    def __init__(self):

        self.SIZE = 19  # The number of locations/cross-points in row or column
        self.TEXT = '  A B C D E F G H I J K L M N O P Q R S'

        self.b = chessboard(SIZE=self.SIZE, TEXT=self.TEXT)
        self.s = searcher(self.SIZE)
        self.s.board = self.b.board()

        # opening = ['2:JJ']
        #
        # import random
        # openid = random.randint(0, len(opening) - 1)
        # self.b.loads(opening[openid])
        # turn = 2
        self.history = []
        # self.undo = False

        # 设置难度
        self.DEPTH = 2
        # speed up
        self._psyco_speedup()

    def _psyco_speedup(self):
        try:
            import psyco
            psyco.bind(chessboard)
            psyco.bind(evaluation)
        except:
            pass
        return 0

    def one_round(self, text):
    # def one_round(self):
        # input the location of people, i.e., text = [row, col]

        self.b.show()   # show the result of last round
        # text = raw_input().strip('\r\n\t')

        tr, tc = text
        # tr = ord(tr) - ord('a')
        # tc = ord(tc) - ord('a')
        row, col = tr, tc
        self.history.append(self.b.dumps())
        self.b[row][col] = 1

        if self.b.check() == 1:
            self.b.show()
            print(self.b.dumps())
            print('')
            print('YOU WIN !!')
            return None, None, 0

        print('robot is thinking now ...')
        score, row, col = self.s.search(2, self.DEPTH)  # ******* 输出接口（机器人落棋位置） ******
        cord = '%s%s' % (chr(ord('A') + row), chr(ord('A') + col))
        print('robot move to %s (%d)' % (cord, score))
        self.b[row][col] = 2
        game_state = 1

        if self.b.check() == 2:
            self.b.show()
            print(self.b.dumps())
            print('')
            print('YOU LOSE.')
            game_state = 0  # 0 means stop and 1 means continue

        return row, col, game_state  # ******* 输出接口（机器人落棋位置，量化坐标） ******



        # if len(text) == 2:
        #     tr, tc = text
        # else:
        #     print 'error input locations!'
        #
        # if tr >= 0 and tc >= 0 and tr < self.SIZE and tc < self.SIZE:
        #     if self.b[tr][tc] == 0:
        #         row, col = tr, tc
        # else:
        #     print 'error input locations!'
        #
        # if self.undo == True:
        #     self.undo = False
        #     if len(self.history) == 0:
        #         print('no history to undo')
        #     else:
        #         print('rollback from history ...')
        #         move = self.history.pop()
        #         self.b.loads(move)
        # else:
        #     self.history.append(self.b.dumps())
        #     self.b[row][col] = 1
        #
        #     if self.b.check() == 1:
        #         self.b.show()
        #         print(self.b.dumps())
        #         print('')
        #         print('YOU WIN !!')
        #         return None, None, 0
        #
        #     print('robot is thinking now ...')
        #     score, row, col = self.s.search(2, self.DEPTH)  # ******* 输出接口（机器人落棋位置） ******
        #     cord = '%s%s' % (chr(ord('A') + row), chr(ord('A') + col))
        #     print('robot move to %s (%d)' % (cord, score))
        #     self.b[row][col] = 2
        #     game_state = 1
        #
        #     if self.b.check() == 2:
        #         self.b.show()
        #         print(self.b.dumps())
        #         print('')
        #         print('YOU LOSE.')
        #         game_state = 0  # 0 means stop and 1 means continue
        #
        #     return row, col, game_state  # ******* 输出接口（机器人落棋位置，量化坐标） ******
