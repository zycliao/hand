# coding=utf-8


import pyrealsense2 as rs
import numpy as np
import cv2


class Camera(object):
    def __init__(self):
        # 配置
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 打开摄像头
        self.pipeline.start(config)

        self.center_array = []
        self.orig_x, self.orig_y = 320., 240.
        self.x_num, self.y_num = 300, 300

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 判定条件
        depth_image[depth_image == 0] = 100000
        if np.min(depth_image) > 30000:
            center_array = self._detection(color_image)
        new_array = [i for i in center_array if i not in self.center_array]
        self.center_array = center_array
        return new_array

    def _detection(self, image):
        res = []
        img1 = image
        # cv2.imwrite("001.jpg", img1)

        # 裁剪原图
        # image_crop=src[:,1000:2500]
        image_out = img1[1000:2500, :]
        print(len(img1), len(img1[0]))
        # 转灰度处理
        gray = cv2.cv2tColor(image_out, cv2.COLOR_BGR2GRAY)
        circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=32, minRadius=30,
                                    maxRadius=150)

        circles = circles1[0, :, :]

        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(image_out, (i[0], i[1]), i[2], (255, 0, 0), 5)
            cv2.circle(image_out, (i[0], i[1]), 2, (255, 0, 255), 10)
            # cv2.rectangle(image_out,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)

            res.append([int((i[0] - self.orig_x) / self.x_num), int((i[1] - self.orig_y) / self.y_num)])
            # print("圆心坐标",i[0],i[1])
            cv2.circle(image_out, (i[0], i[1]), 7, (255, 255, 255), -1)
        return res
