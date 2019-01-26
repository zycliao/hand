# coding=utf-8
import numpy as np
import cv2
from wuzirobot import WuziRobot
from camera import Camera
from utils.logger import logger_init
from gobang.gamemain import GoBang
import time


class Controller(object):
    def __init__(self, robot_first):
        self.robot_first = robot_first
        self.logger = logger_init()
        self.robot = WuziRobot(self.logger)
        self.robot.prepare()
        self.robot.move_to_init()

        self.camera = Camera()
        self.gobang = GoBang()

        self.next_fetch = self.camera.quant2pixel(18, 0)
        self.game_state = 1

    def is_end(self):
        if self.game_state == 0:
            return True
        else:
            return False

    def first(self):
        if self.robot_first:  # robot
            print 'Robot first !'
            decide_row, decide_col, self.game_state = self.gobang.robot_first()
            # self.camera.center_array.append([decide_col, decide_row])
            self.camera.register_new_chess(decide_col, decide_row, 'black')
            # < add robot points to array >
            self.next_step = (decide_col, decide_row)
            _, _, _, black_pixel_out = self.camera.get_gomoku_out()
            self.next_fetch = black_pixel_out[0]
            print self.next_step
        else:  # people
            print 'People first !'

    def decide(self, human_chess):
        """

        """
        # while 1:
        #     self.camera.update_chessboard()
        #     quant_pos, white_out, black_out = self.camera.get_frame()
        #     if quant_pos is None:
        #         return None
        #
        #     if self.robot_first == 1:
        #         self.next_fetch = black_out
        #     else:
        #         self.next_fetch = white_out
        #     if self.next_fetch is not None:
        #         break


        # self.camera.center_array.append([decide_col, decide_row])
        # < add robot points to array >
        # decide_row, decide_col, self.game_state = self.gobang.one_round()
        self.next_step = (decide_col, decide_row)
        print self.next_step
        return 1

    def action(self, human_chess):
        """
        :param human_chess: the chess put by human
        catch a chess and release it.
        self.next_fetch: the pixel where the chess that will be fetched
        self.next_step: the quant location where the chess that will be released
        """
        if human_chess is None: # the robot put the first chess
            decide_row, decide_col, game_state = self.gobang.robot_first()
        else:
            decide_row, decide_col, game_state = self.gobang.one_round([human_chess[1], human_chess[0]])
        if decide_row is None and decide_col is None:
            return None
        _, _, white_pixel_out, black_pixel_out = self.camera.get_gomoku_out()

        dst_x, dst_y = self.camera.quant2pixel(decide_col+3, decide_row+3)

        if self.robot_first:
            self.camera.register_new_chess(decide_col, decide_row, 'black')
            self.robot.catch_chess(black_pixel_out[0][0], black_pixel_out[0][1])
        else:
            self.camera.register_new_chess(decide_col, decide_row, 'white')
            self.robot.catch_chess(white_pixel_out[0][0], white_pixel_out[0][1])
        self.robot.release_chess(dst_x, dst_y)
        print("the robot put the chess in {}".format([decide_col, decide_row]))
        if game_state == 0:
            return None
        else:
            return 1

    def wait_human(self):
        while 1:
            rgb_img, depth_img = self.camera.capture()
            white_quant_in, black_quant_in, _, _ = self.camera.get_gomoku_out()
            white_sub_in = self.camera.full2sub(white_quant_in)
            black_sub_in = self.camera.full2sub(black_quant_in)

            if self.robot_first:
                new_chess = self.camera.one_more_chess(white_sub_in, 'white')
            else:
                new_chess = self.camera.one_more_chess(black_sub_in, 'black')
            without_hand = depth_img[np.logical_and(depth_img<4500., depth_img>1500.)].size < 1
            if new_chess is not None and without_hand:
                break
        cv2.waitKey(1000)
        if self.robot_first:
            self.camera.register_new_chess(new_chess[0], new_chess[1], 'white')
        else:
            self.camera.register_new_chess(new_chess[0], new_chess[1], 'black')
        print("the human put the chess in {}".format(new_chess))
        return new_chess

    def disconnect(self):
        self.robot.move_to_init()
        self.robot.disconnect()
        if self.robot.connected:
            self.robot.robot_shutdown()
            self.robot.disconnect()
            print("Disconnected")
        WuziRobot.uninitialize()

    def say_hello(self):
        pass


if __name__ == '__main__':
    robot_first = True
    controller = Controller(True)

    if robot_first:
        controller.action(None)
    new_chess = controller.wait_human()

    while 1:
        if controller.action(new_chess) is None:
            break
        new_chess = controller.wait_human()
    controller.disconnect()
