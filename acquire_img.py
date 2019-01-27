# coding=utf-8

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from wuzirobot import WuziRobot
from robotcontrol import RobotError
from utils.logger import logger
from utils.cam2tool import cam2tool
from faceio import FaceRecog
from controller import Controller


xxx, yyy = 0.1882, 0.1407


class _Controller(object):
    def __init__(self, robot, win_name='RealSense'):
        self.robot = robot
        self.win_name = win_name
        self.num = 20
        self.img = None
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense', self.move_hand)

    def show(self, images):
        self.img = images
        cv2.imshow('RealSense', images)
        return cv2.waitKey(1)

    def move_hand(self, event, x, y, flags, params):
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     cur_x, cur_y, cur_z = robot.current_waypoint['pos']
        #     X, Y, Z = cam2tool(x, y, cur_x, cur_y)
        #     ik_result = self.robot.inverse_kin(self.robot.current_waypoint['joint'], (X, Y, Z),
        #                                        self.robot.current_waypoint['ori'])
        #     self.robot.move_joint(ik_result['joint'])
        #     import time
        #     time.sleep(5)
        #     # self.robot.move_to_coord(x, y)
        #     self.robot.move_to_init()
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.imwrite('./images/{}.jpg'.format(self.num), self.img)
            np.savetxt('./images/{}_pos.txt'.format(self.num), np.array(self.robot.current_waypoint['pos']))
            np.savetxt('./images/{}_ori.txt'.format(self.num), np.array(self.robot.current_waypoint['ori']))
            self.num += 1


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


WuziRobot.initialize()

# 创建机械臂控制类
robot = WuziRobot(logger)
controller = _Controller(robot)
try:
    robot.prepare()
    print robot.current_waypoint
    # robot.move_to_init()
    robot.move_hello()

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        # mask = np.zeros_like(depth_image)
        # mask[120: 320, 220: 420] = 1
        # depth_image *= mask

        color_image = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        k = controller.show(color_image)
        if k == ord('q'):
            cv2.destroyAllWindows()
            robot.disconnect()
            break

        face = FaceRecog(folder='./faceio/our_faces', sampleCount=8, modelPath="./faceio/deploy.prototxt.txt",
                         weightPath="./faceio/res10_300x300_ssd_iter_140000.caffemodel", confidence=0.8)
        face.predict(color_image)


        # todo 拍不到人脸，可能报错

        # import time
        # time.sleep(1)

        # todo 识别人脸后，调用Controller

        robot.move_to_init()

        break

except RobotError, e:
    logger.error("{0} robot Event:{1}".format(robot.get_local_time(), e))

finally:
    # 断开服务器链接
    if robot.connected:
        # 关闭机械臂
        robot.robot_shutdown()
        # 断开机械臂链接
        robot.disconnect()
        print("Disconnected")
    # 释放库资源
    WuziRobot.uninitialize()
    logger.info("{0} test completed.".format(WuziRobot.get_local_time()))


# Stop streaming
pipeline.stop()

import time
time.sleep(1)


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
