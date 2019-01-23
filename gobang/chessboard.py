# -*- coding: utf-8 -*-
import sys

# ----------------------------------------------------------------------
# chessboard: 棋盘类，简单从字符串加载棋局或者导出字符串，判断输赢等
# ----------------------------------------------------------------------


class chessboard(object):

    def __init__(self, SIZE, forbidden=0, TEXT=None):
        self.__board = [[0 for n in range(SIZE)] for m in range(SIZE)]
        self.__forbidden = forbidden
        self.__dirs = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
        self.DIRS = self.__dirs
        self.won = {}
        self.SIZE = SIZE
        self.TEXT = TEXT

    # 清空棋盘
    def reset(self):
        for j in range(self.SIZE):
            for i in range(self.SIZE):
                self.__board[i][j] = 0
        return 0

    # 索引器
    def __getitem__(self, row):
        return self.__board[row]

    # 将棋盘转换成字符串
    def __str__(self):
        text = self.TEXT + '\n'  # '  A B C D E F G H I J K L M N O\n'
        # text = '  A B C D E F G H I J K L M N O\n'
        mark = ('. ', 'O ', 'X ')
        nrow = 0
        for row in self.__board:
            line = ''.join([mark[n] for n in row])
            text += chr(ord('A') + nrow) + ' ' + line
            nrow += 1
            if nrow < self.SIZE:
                text += '\n'
        return text

    # 转成字符串
    def __repr__(self):
        return self.__str__()

    def get(self, row, col):
        if row < 0 or row >= self.SIZE or col < 0 or col >= self.SIZE:
            return 0
        return self.__board[row][col]

    def put(self, row, col, x):
        if row >= 0 and row < self.SIZE and col >= 0 and col < self.SIZE:
            self.__board[row][col] = x
        return 0

    # 判断输赢，返回0（无输赢），1（白棋赢），2（黑棋赢）
    def check(self):
        board = self.__board
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if board[i][j] == 0:
                    continue
                id = board[i][j]
                for d in dirs:
                    x, y = j, i
                    count = 0
                    for k in range(5):
                        if self.get(y, x) != id:
                            break
                        y += d[0]
                        x += d[1]
                        count += 1
                    if count == 5:
                        self.won = {}
                        r, c = i, j
                        for z in range(5):
                            self.won[(r, c)] = 1
                            r += d[0]
                            c += d[1]
                        return id
        return 0

    # 返回数组对象
    def board(self):
        return self.__board

    # 导出棋局到字符串
    def dumps(self):
        try:
            from StringIO import StringIO
        except ImportError:
            from io import StringIO

        sio = StringIO()
        board = self.__board
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                stone = board[i][j]
                if stone != 0:
                    ti = chr(ord('A') + i)
                    tj = chr(ord('A') + j)
                    sio.write('%d:%s%s ' % (stone, ti, tj))
        return sio.getvalue()

    # 从字符串加载棋局
    def loads(self, text):
        self.reset()
        board = self.__board
        for item in text.strip('\r\n\t ').replace(',', ' ').split(' '):
            n = item.strip('\r\n\t ')
            if not n:
                continue
            n = n.split(':')
            stone = int(n[0])
            i = ord(n[1][0].upper()) - ord('A')
            j = ord(n[1][1].upper()) - ord('A')
            board[i][j] = stone
        return 0

    # 设置终端颜色
    def console(self, color):
        if sys.platform[:3] == 'win':
            try:
                import ctypes
            except:
                return 0

            kernel32 = ctypes.windll.LoadLibrary('kernel32.dll')
            GetStdHandle = kernel32.GetStdHandle
            SetConsoleTextAttribute = kernel32.SetConsoleTextAttribute
            GetStdHandle.argtypes = [ ctypes.c_uint32 ]
            GetStdHandle.restype = ctypes.c_size_t
            SetConsoleTextAttribute.argtypes = [ ctypes.c_size_t, ctypes.c_uint16 ]
            SetConsoleTextAttribute.restype = ctypes.c_long
            handle = GetStdHandle(0xfffffff5)
            if color < 0: color = 7
            result = 0
            if (color & 1): result |= 4
            if (color & 2): result |= 2
            if (color & 4): result |= 1
            if (color & 8): result |= 8
            if (color & 16): result |= 64
            if (color & 32): result |= 32
            if (color & 64): result |= 16
            if (color & 128): result |= 128
            SetConsoleTextAttribute(handle, result)
        else:
            if color >= 0:
                foreground = color & 7
                background = (color >> 4) & 7
                bold = color & 8
                sys.stdout.write(" \033[%s3%d;4%dm" % (bold and "01;" or "", foreground, background))
                sys.stdout.flush()
            else:
                sys.stdout.write(" \033[0m")
                sys.stdout.flush()
        return 0

    # 彩色输出
    def show(self):
        print(self.TEXT)  # '  A B C D E F G H I J K L M N O'
        # print('  A B C D E F G H I J K L M N O')
        mark = ('. ', 'O ', 'X ')
        nrow = 0
        self.check()
        color1 = 10
        color2 = 13
        for row in range(self.SIZE):
            print(chr(ord('A') + row)),
            for col in range(self.SIZE):
                ch = self.__board[row][col]
                if ch == 0:
                    self.console(-1)
                    print('.'),
                elif ch == 1:
                    if (row, col) in self.won:
                        self.console(9)
                    else:
                        self.console(10)
                    print('O'),
                # self.console(-1)
                elif ch == 2:
                    if (row, col) in self.won:
                        self.console(9)
                    else:
                        self.console(13)
                    print('X'),
                # self.console(-1)
            self.console(-1)
            print('')
        return 0

