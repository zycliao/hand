from wuzirobot import WuziRobot
from camera import Camera
from utils.logger import logger_init
from gobang.gamemain import GoBang
import time


class Controller(object):
    def __init__(self):
        self.camera = Camera()
        self.gobang = GoBang()

        self.logger = logger_init()
        self.robot = WuziRobot(self.logger)
        self.robot.prepare()
        self.robot.move_to_init()

        self.next_step = (0, 0)
        self.game_state = 1

    def is_end(self):
        if self.game_state == 0:
            return True
        else:
            return False

    def decide(self):
        quant_pos = self.camera.get_frame()
        print "quant_pos: {}".format(quant_pos)
        decide_row, decide_col, self.game_state = self.gobang.one_round([quant_pos[1], quant_pos[0]])  # row, col, 0 or 1
        # decide_row, decide_col, self.game_state = self.gobang.one_round()
        self.next_step = (decide_col, decide_row)
        print self.next_step

    def action(self):
        to_x, to_y = self.camera.quant2pixel(18, 0)
        dst_x, dst_y = self.camera.quant2pixel(self.next_step[0], self.next_step[1])
        self.robot.catch_chess(to_x, to_y)
        self.robot.release_chess(dst_x, dst_y)

    def wait_human(self):
        return input()

    def disconnect(self):
        self.robot.move_to_init()
        self.robot.disconnect()
        if self.robot.connected:
            self.robot.robot_shutdown()
            self.robot.disconnect()
            print("Disconnected")
        WuziRobot.uninitialize()


if __name__ == '__main__':
    controller = Controller()
    # ....

    while 1:
        controller.decide()
        if not controller.is_end():
            controller.action()
            if controller.wait_human() == 'q':
                break
        else:
            break
    controller.disconnect()
