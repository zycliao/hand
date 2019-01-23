from wuzirobot import WuziRobot
from camera import Camera
from utils.logger import logger_init


class Controller(object):
    def __init__(self):
        self.camera = Camera()

        self.logger = logger_init()
        self.robot = WuziRobot(self.logger)
        self.robot.prepare()
        self.robot.move_to_init()



    def win(self):
        pass

    def round(self):
        pass


if __name__ == '__main__':
    controller = Controller()
    # ....

    while not controller.win():
        controller.round()