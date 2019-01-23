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

    # def test1():
    #     b = chessboard()
    #     b[10][10] = 1
    #     b[11][11] = 2
    #     for i in range(4):
    #         b[5 + i][2 + i] = 2
    #     for i in range(4):
    #         b[7 - 0][3 + i] = 2
    #     print(b)
    #     print('check', b.check())
    #     return 0
    #
    # def test2():
    #     b = chessboard()
    #     b[7][7] = 1
    #     b[8][8] = 2
    #     b[7][9] = 1
    #     eva = evaluation()
    #     for l in eva.POS: print(l)
    #     return 0
    #
    # def test3():
    #     e = evaluation()
    #     line = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    #     record = []
    #     e.analysis_line(line, record, len(line), 6)
    #     print(record[:10])
    #     return 0
    #
    # def test4():
    #     b = chessboard()
    #     b.loads('2:DF 1:EG 2:FG 1:FH 2:FJ 2:GG 1:GH 1:GI 2:HG 1:HH 1:IG 2:IH 1:JF 2:JI 1:KE')
    #     b.loads('2:CE 2:CK 1:DF 1:DK 2:DL 1:EG 1:EI 1:EK 2:FG 1:FH 1:FI 1:FJ 1:FK 2:FL 1:GD 2:GE 2:GF 2:GG 2:GH 1:GI 1:GK 2:HG 1:HH 2:HJ 2:HK 2:IG 1:JG 2:AA')
    #     eva = evaluation()
    #     print(b)
    #     score = 0
    #     t = time.time()
    #     for i in range(10000):
    #         score = eva.evaluate(b.board(), 2)
    #     # eva.test(b.board())
    #     t = time.time() - t
    #     print(score, t)
    #     print(eva.textrec(3))
    #     return 0
    #
    # def test5():
    #     import profile
    #     profile.run("test4()", "prof.txt")
    #     import pstats
    #     p = pstats.Stats("prof.txt")
    #     p.sort_stats("time").print_stats()
    #
    # def test6():
    #     b = chessboard()
    #     b.loads('1:CJ 2:DJ 1:dk 1:DL 1:EH 1:EI 2:EJ 2:EK 2:FH 2:FI 2:FJ 1:FK 2:FL 1:FM 2:GF 1:GG 2:GH 2:GI 2:GJ 1:GK 1:GL 2:GM 1:HE 2:HF 2:HG 2:HH 2:HI 1:HJ 2:HK 2:HL 1:IF 1:IG 1:IH 2:II 1:IJ 2:IL 2:JG 1:JH 1:JI 1:JJ 1:JK 2:JL 1:JM 1:KI 2:KJ 1:KL 1:LJ 2:MK')
    #     # b.loads('1:HH,1:HI,1:HJ,1:HK')
    #     s = searcher(SIZE)
    #     s.board = b.board()
    #     t = time.time()
    #     score, row, col = s.search(2, 3)
    #     t = time.time() - t
    #     b[row][col] = 2
    #     print(b)
    #     print(score, t)
    #     print(chr(ord('A') + row) + chr(ord('A') + col))
    #
    # def test7():
    #     b = chessboard(SIZE=SIZE)
    #     s = searcher(SIZE)
    #     s.board = b.board()
    #     b.loads('2:HH 1:JF')
    #     turn = 2
    #     while 1:
    #         score, row, col = s.search(2, 2)
    #         print('robot move %s%s (%d)'%(chr(ord('A') + row), chr(ord('A') + col), score))
    #         b[row][col] = 2
    #         if b.check() == 2:
    #             print(b)
    #             print(b.dumps())
    #             print('you lose !!')
    #             return 0
    #         while 1:
    #             print(b)
    #             print('your move (pos):'),
    #             text = input().strip('\r\n\t ')
    #             if len(text) == 2:
    #                 tr = ord(text[0].upper()) - ord('A')
    #                 tc = ord(text[1].upper()) - ord('A')
    #                 if tr >= 0 and tc >= 0 and tr < SIZE and tc < SIZE:
    #                     if b[tr][tc] == 0:
    #                         row, col = tr, tc
    #                         break
    #                     else:
    #                         print('can not move there')
    #                 else:
    #                     print('bad position')
    #             elif text.upper() == 'Q':
    #                 print(b.dumps())
    #                 return 0
    #         b[row][col] = 1
    #         if b.check() == 1:
    #             print(b)
    #             print(b.dumps())
    #             print('you win !!')
    #             return 0
    #     return 0

    # test7()
    gamemain(SIZE, TEXT)



