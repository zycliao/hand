# -*- coding: utf-8 -*-

import sys
from chessboard import chessboard
from searcher import searcher
from ipdb import *

# ----------------------------------------------------------------------
# main game
# ----------------------------------------------------------------------


def gamemain(SIZE, TEXT):
    b = chessboard(SIZE=SIZE, TEXT=TEXT)
    s = searcher(SIZE)
    s.board = b.board()

    opening = [
        '2:JJ'
        # '1:HH 2:II',
        # '2:IG 2:GI 1:HH',
        # '1:IH 2:GI',
        # '1:HG 2:HI',
        # '2:HG 2:HI 1:HH',
        # '1:HH 2:IH 2:GI',
        # '1:HH 2:IH 2:HI',
        # '1:HH 2:IH 2:HJ',
        # '1:HG 2:HH 2:HI',
        # '1:GH 2:HH 2:HI',
    ]

    import random
    openid = random.randint(0, len(opening) - 1)
    b.loads(opening[openid])
    turn = 2
    history = []
    undo = False

    # 设置难度
    DEPTH = 1

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'hard':
            DEPTH = 2

    while 1:
        print('')
        while 1:
            print('<ROUND %d>' % (len(history) + 1))
            b.show()
            print('Your move (u:undo, q:quit):'),
            text = raw_input().strip('\r\n\t ')    # ******* 输入接口（新来棋子） *******
            # set_trace()
            if len(text) == 2:
                tr = ord(text[0].upper()) - ord('A')
                tc = ord(text[1].upper()) - ord('A')
                if tr >= 0 and tc >= 0 and tr < SIZE and tc < SIZE:
                    if b[tr][tc] == 0:
                        row, col = tr, tc
                        break
                    else:
                        print('can not move there')
                else:
                    print('bad position')
            elif text.upper() == 'U':
                undo = True
                break
            elif text.upper() == 'Q':
                print(b.dumps())
                return 0

        if undo == True:
            undo = False
            if len(history) == 0:
                print('no history to undo')
            else:
                print('rollback from history ...')
                move = history.pop()
                b.loads(move)
        else:
            history.append(b.dumps())
            b[row][col] = 1

            if b.check() == 1:
                b.show()
                print(b.dumps())
                print('')
                print('YOU WIN !!')
                return 0

            print('robot is thinking now ...')
            score, row, col = s.search(2, DEPTH)    # ******* 输出接口（机器人落棋位置） ******
            cord = '%s%s' % (chr(ord('A') + row), chr(ord('A') + col))
            print('robot move to %s (%d)' % (cord, score))
            b[row][col] = 2

            if b.check() == 2:
                b.show()
                print(b.dumps())
                print('')
                print('YOU LOSE.')
                return 0