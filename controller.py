# coding=utf-8
import numpy as np
import cv2
from wuzirobot import WuziRobot
from camera import Camera
from utils.logger import logger_init
from gobang.gamemain import GoBang
import time


class Controller(object):
    def __init__(self):
        self.logger = logger_init()
        self.robot = WuziRobot(self.logger)
        self.robot.prepare()
        self.robot.move_to_init()

        self.camera = Camera()
        self.gobang = GoBang()

        self.next_step = (0, 0)
        self.next_fetch = self.camera.quant2pixel(18, 0)
        self.game_state = 1

    def is_end(self):
        if self.game_state == 0:
            return True
        else:
            return False

    def first(self, first_id):
        self.first_id = first_id
        if first_id == 1:  # robot
            print 'Robot first !'
            decide_row, decide_col, self.game_state = self.gobang.robot_first()
            self.camera.center_array.append([decide_col, decide_row])
            # < add robot points to array >
            self.next_step = (decide_col, decide_row)
            _, _, _, black_pixel_out = self.camera.get_gomoku_location()
            self.next_fetch = black_pixel_out[0]
            print self.next_step
        else:  # people
            print 'People first !'

    def decide(self):
        while 1:
            self.camera.update_chessboard()
            quant_pos, white_out, black_out = self.camera.get_frame()
            if quant_pos is None:
                return None

            if self.first_id == 1:
                self.next_fetch = black_out
            else:
                self.next_fetch = white_out
            if self.next_fetch is not None:
                break

        print "quant_pos: {}".format(quant_pos)
        decide_row, decide_col, self.game_state = self.gobang.one_round([quant_pos[1], quant_pos[0]])  # row, col, 0 or 1
        self.camera.center_array.append([decide_col, decide_row])
        # < add robot points to array >
        # decide_row, decide_col, self.game_state = self.gobang.one_round()
        self.next_step = (decide_col, decide_row)
        print self.next_step
        return 1

    def action(self):
        # to_x, to_y = self.camera.quant2pixel(self.next_fetch[0], self.next_fetch[1])
        to_x, to_y = self.next_fetch[0], self.next_fetch[1]
        print("Action: {}".format(self.next_step))
        dst_x, dst_y = self.camera.quant2pixel(self.next_step[0]+3, self.next_step[1]+3)

        self.robot.catch_chess(to_x, to_y)
        self.robot.release_chess(dst_x, dst_y)

    def wait_human(self):
        while 1:
            rgb_img, depth_img = self.camera.capture()
            new_array, _, _ = self.camera.get_frame(False)

            depth_mask = np.zeros_like(depth_img)
            depth_mask_size = 400
            depth_mask[(480 - depth_mask_size) / 2: depth_mask_size + (480 - depth_mask_size) / 2,
            (640 - depth_mask_size) / 2: depth_mask_size + (640 - depth_mask_size) / 2] = 1
            # depth_img *= depth_mask
            without_hand = depth_img[np.logical_and(depth_img<4500., depth_img>1500.)].size < 1
            if new_array is not None and without_hand:
                break
        cv2.waitKey(1000)
        # return input('wait for human')

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
    controller = Controller()

    # first hand
    first_id = 1  # 1 means robot and 0 means people
    controller.first(first_id)

    if first_id == 1:
        controller.action()
        controller.wait_human()
        # < 等机器人回原位并且人下完棋子再继续下面的程序 >

    while 1:
        while controller.decide() is None:
            pass
        if not controller.is_end():
            controller.action()
            if controller.wait_human() == 'q':
                break
        else:
            break
    controller.disconnect()
