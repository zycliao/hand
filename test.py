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


xxx, yyy = 0.1882, 0.1407


class Controller(object):
    def __init__(self, robot, win_name='RealSense'):
        self.robot = robot
        self.win_name = win_name
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense', self.move_hand)

    def show(self, images):
        cv2.circle(images, (320, 240), 2, (0, 255, 0), 1)
        cv2.circle(images, (640, 240), 2, (0, 255, 0), 1)
        cv2.circle(images, (320, 0), 2, (0, 255, 0), 1)
        cv2.imshow('RealSense', images)
        # print(self.robot.current_waypoint)

        return cv2.waitKey(1)

    def move_hand(self, event, x, y, flags, params):
        print('move hand!')
        if event == cv2.EVENT_LBUTTONDOWN:
            self.robot.move_to_coord(x, y)
            # print(self.robot.current_waypoint['pos'])
            self.robot.move_to_init()


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
controller = Controller(robot)
try:
    robot.prepare()
    print robot.current_waypoint
    robot.move_to_init()

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        mask = np.zeros_like(depth_image)
        mask[120: 320, 220: 420] = 1
        depth_image *= mask
        # print(np.min(depth_image[depth_image > 0]))
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        k = controller.show(images)
        if k == ord('q'):
            cv2.destroyAllWindows()
            robot.disconnect()
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