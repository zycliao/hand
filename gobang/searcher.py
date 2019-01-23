# -*- coding: utf-8 -*-
from evaluation import evaluation

# ----------------------------------------------------------------------
# DFS: 博弈树搜索
# ----------------------------------------------------------------------


class searcher(object):

    # 初始化
    def __init__(self, SIZE):
        self.evaluator = evaluation(SIZE)
        self.board = [[0 for n in range(SIZE)] for i in range(SIZE)]
        self.gameover = 0
        self.overvalue = 0
        self.maxdepth = 3
        self.SIZE = SIZE

    # 产生当前棋局的走法
    def genmove(self, turn):
        moves = []
        board = self.board
        POSES = self.evaluator.POS
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if board[i][j] == 0:
                    score = POSES[i][j]
                    moves.append((score, i, j))
        moves.sort()
        moves.reverse()
        return moves

    # 递归搜索：返回最佳分数
    def __search(self, turn, depth, alpha=-0x7fffffff, beta=0x7fffffff):

        # 深度为零则评估棋盘并返回
        if depth <= 0:
            score = self.evaluator.evaluate(self.board, turn)
            return score

        # 如果游戏结束则立马返回
        score = self.evaluator.evaluate(self.board, turn)
        if abs(score) >= 9999 and depth < self.maxdepth:
            return score

        # 产生新的走法
        moves = self.genmove(turn)
        bestmove = None

        # 枚举当前所有走法
        for score, row, col in moves:

            # 标记当前走法到棋盘
            self.board[row][col] = turn

            # 计算下一回合该谁走
            nturn = turn == 1 and 2 or 1

            # 深度优先搜索，返回评分，走的行和走的列
            score = - self.__search(nturn, depth - 1, -beta, -alpha)

            # 棋盘上清除当前走法
            self.board[row][col] = 0

            # 计算最好分值的走法
            # alpha/beta 剪枝
            if score > alpha:
                alpha = score
                bestmove = (row, col)
                if alpha >= beta:
                    break

        # 如果是第一层则记录最好的走法
        if depth == self.maxdepth and bestmove:
            self.bestmove = bestmove

        # 返回当前最好的分数，和该分数的对应走法
        return alpha

    # 具体搜索：传入当前是该谁走(turn=1/2)，以及搜索深度(depth)
    def search(self, turn, depth=3):
        self.maxdepth = depth
        self.bestmove = None
        score = self.__search(turn, depth)
        if abs(score) > 8000:
            self.maxdepth = depth
            score = self.__search(turn, 1)
        row, col = self.bestmove
        return score, row, col