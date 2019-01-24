# coding=utf-8


import pyrealsense2 as rs
import numpy as np
import cv2
import time


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
        self.orig_x, self.orig_y = 70., 44.
        self.b_x,self.b_y=(499.-self.orig_x)/18.,(454.-self.orig_y)/18.
        self.x_num, self.y_num = 24, 24
        # time.sleep(3)


    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #a=cv2.imwrite('1.jpg',color_image)
        #cv2.imshow('1.jpg',a)

        # 判定条件
        #depth_image[depth_image == 0] = 100000
        # if np.min(depth_image) > 30000:
        #cv2.imshow("img",color_image)
        #cv2.waitKey()

        if 1:
            _,center_array1 = self._detection_white(color_image)
            _,center_array2 = self._detection_black(color_image)
            center_array=center_array1+center_array2
        new_array = [i for i in center_array if i not in self.center_array]
        self.center_array = center_array
        #print color_image
        print new_array
        return new_array[0]

    def quant2pixel(self, col, row):
        return col*self.b_x+self.orig_x,row*self.b_y+self.orig_y


    def edge(self,image):
        blurred = cv2.GaussianBlur(image , (3 , 3) , 0)
        gray=cv2.cvtColor(blurred,cv2.COLOR_RGB2GRAY)
        xgrad=cv2.Sobel(gray,cv2.CV_16SC1,1,0)
        ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
        edge_output=cv2.Canny(xgrad,ygrad,40,120)
        return edge_output

    #检测黑棋
    def _detection_white(self, image):
        #cv2.imshow('1',image)
        #cv2.waitKey()
        mask = (image[:, :, 0: 1] < 100).astype(np.uint8)
        a = mask[:, :, 0] * 255

        #cv2.imshow('a',a)
        #cv2.waitKey()
        res_int = []
        res_float = []
        image_out = a
        # cv2.imwrite("001.jpg", img1)

        # 裁剪原图
        # image_crop=src[:,1000:2500]
        #image_out = img1[:,:]
        #image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
         # 转灰度处理
        #gray = cv2.cvtColor(image_out, cv2.COLOR_BGR2GRAY)
        circles1 = cv2.HoughCircles(image_out, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=10, minRadius=8,
                                    maxRadius=20)
        if circles1 is None:
            return [],[]
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))

        for i in circles[:]:
            cv2.circle(image_out, (i[0], i[1]), i[2], (0, 100, 0), -1)
            cv2.circle(image_out, (i[0], i[1]), 2, (0, 255, 0), -1)
            # cv2.rectangle(image_out,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)
            res_float.append([i[0], i[1]])
            res_int.append([int(np.round((i[0] - self.orig_x)/self.b_x)), int(np.round((i[1] - self.orig_y) / self.b_y))])
            print("圆心坐标", i[0], i[1])


            #cv2.circle(image_out, (i[0], i[1]), 7, (255, 255, 255), -1)

        return res_float,res_int

    def _detection_black(self, image):
        #cv2.imshow('1',image)
        #cv2.waitKey()
        mask = (image[:, :, 0: 1] < 100).astype(np.uint8)
        image*=mask
        res_int = []
        res_float = []
        img1 = image
        # cv2.imwrite("001.jpg", img1)

        # 裁剪原图
        # image_crop=src[:,1000:2500]
        image_out = img1[:,:]
         # 转灰度处理
        gray = cv2.cvtColor(image_out, cv2.COLOR_BGR2GRAY)
        circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=25, minRadius=8,
                                    maxRadius=20)
        if circles1 is None:
            return [],[]
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(image_out, (i[0], i[1]), i[2], (0, 100, 0), -1)
            cv2.circle(image_out, (i[0], i[1]), 2, (0, 255, 0), -1)
            # cv2.rectangle(image_out,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)
            res_float.append([i[0], i[1]])
            res_int.append([int(np.round((i[0] - self.orig_x) / self.b_x)), int(np.round((i[1] - self.orig_y) / self.b_y))])

            print("圆心坐标",i[0],i[1])
            #cv2.circle(image_out, (i[0], i[1]), 7, (255, 255, 255), -1)

        return res_float,res_int
