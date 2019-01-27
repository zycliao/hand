# coding=utf-8


import pyrealsense2 as rs
import numpy as np
import cv2
from find_chessboard import find_chessboard
import time


class Camera(object):
    def __init__(self):
        # 配置
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.lu, self.ld, self.ru, self.rd = [], [], [], []
        # 打开摄像头
        self.pipeline.start(config)

        self.center_array = []
        time.sleep(2)
        self.update_chessboard()

        self.circle_mask = np.zeros([23, 23], dtype=np.uint8)

        cv2.circle(self.circle_mask, (11, 11), 11, (255, 255, 255), -1)
        self.circle_mask /= 255

        # these two arrays store quant coord that has been subtracted by 3
        self.black_array = []
        self.white_array = []

        # the length of the sub area in the chessboard
        self.sub_len = 13
        self.sub_pad = (19 - self.sub_len) / 2

    def capture(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def update_chessboard(self):
        image, _ = self.capture()
        self.lu, self.ru, self.rd, self.ld = find_chessboard(image, debug=True)

        # self.orig_x, self.orig_y = lu[0], lu[1]
        # self.b_x, self.b_y = (ru[0] - self.orig_x) / 18., (rd[1] - self.orig_y) / 18.

    def get_gomoku_inside(self):
        color_image, _ = self.capture()
        white_pixel_ = self.detection_white(color_image)
        black_pixel_ = self.detection_black(color_image)
        # white_quant = self.pixel2quant(white_pixel_)
        # black_quant = self.pixel2quant(black_pixel)

        white_quant_in = []
        black_quant_in = []
        white_pixel_out = []
        black_pixel_out = []
        for w_pixel in white_pixel_:
            w_quant = self.pixel2quant(w_pixel)
            if 0 <= w_quant[0] <= 18 and 0 <= w_quant[1] <= 18:
                if self.sub_pad <= w_quant[0] <= 18 - self.sub_pad and self.sub_pad <= w_quant[1] <= 18 - self.sub_pad:
                    white_quant_in.append(w_quant)
                else:
                    white_pixel_out.append(w_pixel)

        for b_pixel in black_pixel_:
            b_quant = self.pixel2quant(b_pixel)
            if 0 <= b_quant[0] <= 18 and 0 <= b_quant[1] <= 18:
                if self.sub_pad <= b_quant[0] <= 18 - self.sub_pad and self.sub_pad <= b_quant[1] <= 18 - self.sub_pad:
                    black_quant_in.append(b_quant)
                else:
                    black_pixel_out.append(b_pixel)

        # white_pixel = [k for k in white_pixel_ if 0 <= k[0] <= 18 and 0 <= k[1] <= 18]
        # black_pixel = [k for k in black_pixel if 0 <= k[0] <= 18 and 0 <= k[1] <= 18]

        # black_in = [k for k in black_pixel if 3 <= k[0] <= 15 and 3 <= k[1] <= 15]
        # white_in = [k for k in white_pixel if 3 <= k[0] <= 15 and 3 <= k[1] <= 15]
        return white_quant_in, black_quant_in, white_pixel_out, black_pixel_out

    def get_gomoku_out(self):
        color_image, _ = self.capture()
        white_pixel_ = self.detection_white(color_image)
        black_pixel_ = self.detection_black(color_image)
        # white_quant = self.pixel2quant(white_pixel_)
        # black_quant = self.pixel2quant(black_pixel)

        white_quant_in = []
        black_quant_in = []
        white_pixel_out = []
        black_pixel_out = []
        for w_pixel in white_pixel_:
            w_quant = self.pixel2quant(w_pixel)
            if 0 <= w_quant[0] <= 18 and 0 <= w_quant[1] <= 18:
                if self.sub_pad <= w_quant[0] <= 18 - self.sub_pad and self.sub_pad <= w_quant[1] <= 18 - self.sub_pad:
                    white_quant_in.append(w_quant)
                else:
                    white_pixel_out.append(w_pixel)

        for b_pixel in black_pixel_:
            b_quant = self.pixel2quant(b_pixel)
            if 0 <= b_quant[0] <= 18 and 0 <= b_quant[1] <= 18:
                if self.sub_pad <= b_quant[0] <= 18 - self.sub_pad and self.sub_pad <= b_quant[1] <= 18 - self.sub_pad:
                    black_quant_in.append(b_quant)
            else:
                if b_pixel[0] > np.minimum(self.ru[0], self.rd[0]):
                    black_pixel_out.append(b_pixel)

        # white_pixel = [k for k in white_pixel_ if 0 <= k[0] <= 18 and 0 <= k[1] <= 18]
        # black_pixel = [k for k in black_pixel if 0 <= k[0] <= 18 and 0 <= k[1] <= 18]

        # black_in = [k for k in black_pixel if 3 <= k[0] <= 15 and 3 <= k[1] <= 15]
        # white_in = [k for k in white_pixel if 3 <= k[0] <= 15 and 3 <= k[1] <= 15]

        # TODO: sort it so that the first element is the min y
        black_pixel_out = sorted(black_pixel_out, key=lambda x: x[1])
        return white_quant_in, black_quant_in, white_pixel_out, black_pixel_out

    # def pixel2quants(self,pixels , flag_in=False):
    #     quant_list=[]
    #     for pixel in pixels:
    #         quant_x = int(np.round((pixel[0] - self.orig_x) / self.b_x))
    #         quant_y = int(np.round((pixel[1] - self.orig_y) / self.b_y))
    #         if flag_in:
    #             quant_x-=3
    #             quant_y-=3
    #
    #         quant_list.append([quant_x,quant_y])
    #     return  quant_list

    def pixel2quant(self, pixel):
        if not self.point_inside(pixel, self.lu, self.ru, self.rd, self.ld):
            return [50., 50.]
        # if self.lu[0] - self.ld[0] == 0:
        #     a_x = 1
        #     b_x = 0
        # else:
        #     a_x = (self.lu[1] - self.ld[1]) / (self.lu[0] - self.ld[0])
        #     b_x = self.lu[1] - a_x * self.lu[0]
        # d_pixel_x = abs((pixel[0] * a_x - pixel[1] + b_x) / (np.sqrt(1 + a_x * a_x)))
        #
        # a_y = (self.ru[1] - self.lu[1]) / (self.ru[0] - self.lu[0])
        # b_y = self.ru[1] - a_y * self.ru[0]
        # d_pixel_y = abs((pixel[0] * a_y - pixel[1] + b_y) / (np.sqrt(1 + a_y * a_y)))
        d_pixel_x = self.point2line(pixel, self.lu, self.ld)
        d_pixel_y = self.point2line(pixel, self.ru, self.lu)

        d_all_x = np.sqrt(np.square(self.ru[0] - self.lu[0]) + np.square(self.ru[1] - self.lu[1])) + np.sqrt(
            np.square(self.rd[0] - self.ld[0]) + np.square(self.rd[1] - self.ld[1]))
        d_all_y = np.sqrt(np.square(self.lu[0] - self.ld[0]) + np.square(self.lu[1] - self.ld[1])) + np.sqrt(
            np.square(self.ru[0] - self.rd[0]) + np.square(self.ru[1] - self.rd[1]))
        d_all_x /= 2.
        d_all_y /= 2.

        # print "QUANT: {} {}".format(d_pixel_x / (d_all_x / 18.), d_pixel_y / (d_all_y / 18.))
        quant_x = int(np.round(d_pixel_x / (d_all_x / 18.)))
        quant_y = int(np.round(d_pixel_y / (d_all_y / 18.)))

        return [quant_x, quant_y]

    def point_inside(self, p, p1, p2, p3, p4):
        """
        p1, p2, p3, p4 have to be sorted
        p1 to p4 is four points of the quad, p is the target point
        """
        true_area = self.triangle_area(p1, p2, p3) + self.triangle_area(p3, p4, p1)
        area = self.triangle_area(p, p1, p2) + self.triangle_area(p, p2, p3) + self.triangle_area(p, p3,
                                                                                                  p4) + self.triangle_area(
            p, p4, p1)
        if np.abs(true_area - area) < 1:
            return True
        else:
            # print("{} is outside the quad".format(p))
            return False

    def triangle_area(self, p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        bottom_line = np.sqrt(np.sum(np.square(p3 - p2)))
        height = self.point2line(p1, p2, p3)
        return height * bottom_line / 2.

    def point2line(self, p, l1, l2):
        """
        l1, l2 is two points on the line
        p is the point
        y = ax + b
        """
        if l1[0] == l2[0]:
            a = 1
            b = 0
        else:
            a = (l1[1] - l2[1]) / (l1[0] - l2[0])
            b = l1[1] - a * l1[0]
        try:
            dis = abs((p[0] * a - p[1] + b) / (np.sqrt(1 + a * a)))
        except IndexError as e:
            print e
        return dis

    def quant2pixel(self, quant_x, quant_y):
        quant_x_up = self.lu[0] + (self.ru[0] - self.lu[0]) * quant_x / 18.
        quant_x_down = self.ld[0] + (self.rd[0] - self.ld[0]) * quant_x / 18.

        quant_y_up = self.lu[1] + (self.ru[1] - self.lu[1]) * quant_x / 18.
        quant_y_down = self.ld[1] + (self.rd[1] - self.ld[1]) * quant_x / 18.

        # quant_y_left = self.lu[1] + (self.ld[1] - self.lu[1]) * quant_y / 18
        # quant_y_right = self.ru[1] + (self.rd[1] - self.ru[1]) * quant_y / 18
        pixel_x = quant_x_up + (quant_x_down - quant_x_up) * quant_y / 18.
        pixel_y = quant_y_up + (quant_y_down - quant_y_up) * quant_y / 18.
        return [pixel_x, pixel_y]

    # def get_frame(self, update=True):
    #     white_quant_in, black_quant_in, white_pixel_out, black_pixel_out = self.get_gomoku_out()
    #     white_fix_in = [[i[0] - 3, i[1] - 3] for i in white_quant_in]
    #     black_fix_in = [[i[0] - 3, i[1] - 3] for i in black_quant_in]
    # center_array = white_fix_in + black_fix_in
    #
    # new_array = [i for i in center_array if i not in self.center_array]

    # if update:
    #     self.center_array = center_array
    #     print("Center array: {}".format(self.center_array))
    #
    #     # draw all quant
    #     quant_img, _ = self.capture()
    #     quant_img = self.draw_all_quant(quant_img)
    #     cv2.imshow('quant', quant_img)

    # print color_image
    # print 'new_array %s' % new_array
    # return new_array[0] if len(new_array) > 0 else None, white_pixel_out[0] if len(white_pixel_out)>0 else None, \
    #        black_pixel_out[0] if len(black_pixel_out)>0 else None

    # def quant2pixel(self, col, row):
    #     return col * self.b_x + self.orig_x, row * self.b_y + self.orig_y
    def draw_all_quant(self, img):
        img = np.copy(img)
        for i in range(19):
            for j in range(19):
                x, y = self.quant2pixel(i, j)
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
        return img

    def edge(self, image):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
        ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
        edge_output = cv2.Canny(xgrad, ygrad, 40, 120)
        return edge_output

    def detection_white(self, image):
        image = np.copy(image)

        mask = (image[:, :, 0: 1] < 150).astype(np.uint8)
        image_out = mask[:, :, 0] * 255

        res_float = []
        circles1 = cv2.HoughCircles(image_out, cv2.HOUGH_GRADIENT, 1, 15, param1=100, param2=10, minRadius=8,
                                    maxRadius=20)
        if circles1 is None:
            return [], []
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))

        for i in circles[:]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 100, 0), -1)
            cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), -1)
            # cv2.rectangle(image_out,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)
            res_float.append([i[0], i[1]])

            # print("圆心坐标", i[0], i[1])

            cv2.circle(image, (i[0], i[1]), 7, (255, 0, 0), -1)
        cv2.imshow('white', image)
        cv2.imshow('white_mask', image_out)

        return res_float

    # def detection_white_out(self, image):
    #     image = np.copy(image)
    #
    #     # mask = (image[:, :, 0: 1] < 150).astype(np.uint8)
    #     # image_out = mask[:, :, 0] * 255
    #     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     res_float = []
    #     circles1 = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 15, param1=100, param2=15, minRadius=8,
    #                                 maxRadius=20)
    #     if circles1 is None:
    #         return [], []
    #     circles = circles1[0, :, :]
    #     circles = np.uint16(np.around(circles))
    #
    #     for i in circles[:]:
    #         if self.filter_black(image[cir_center_int[1] - rad: cir_center_int[1] + rad + 1,
    #                              cir_center_int[0] - rad: cir_center_int[0] + rad + 1]):
    #         cv2.circle(image, (i[0], i[1]), i[2], (0, 100, 0), -1)
    #         cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), -1)
    #         # cv2.rectangle(image_out,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)
    #         res_float.append([i[0], i[1]])
    #
    #         # print("圆心坐标", i[0], i[1])
    #
    #         cv2.circle(image, (i[0], i[1]), 7, (255, 0, 0), -1)
    #     # filter black chess
    #
    #     cv2.imshow('white', image)
    #     return res_float


    def detection_black(self, image):
        # cv2.imshow('1',image)
        # cv2.waitKey()
        image = np.copy(image)
        draw_image = np.copy(image)
        mask = (np.mean(image, -1) > 60).astype(np.uint8)
        image_out = mask * 255
        res_float = []
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('black_mask', image_gray)
        circles1 = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 15, param1=100, param2=12, minRadius=8,
                                    maxRadius=15)
        if circles1 is None:
            return [], []
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        rad = 8
        for circle in circles[:]:
            cir_center_int = np.around(np.array(circle)).astype(int)
            if self.filter_black(image[cir_center_int[1] - rad: cir_center_int[1] + rad + 1,
                                 cir_center_int[0] - rad: cir_center_int[0] + rad + 1]):
                circle[0], circle[1] = self.optimize_center(image, circle[0], circle[1])
                cv2.circle(draw_image, (circle[0], circle[1]), circle[2], (0, 100, 0), -1)
                cv2.circle(draw_image, (circle[0], circle[1]), 2, (0, 255, 0), -1)
                # cv2.rectangle(image_out,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)
                res_float.append([circle[0], circle[1]])
                # print("圆心坐标", i[0], i[1])
                cv2.circle(draw_image, (circle[0], circle[1]), 7, (255, 0, 0), -1)

        cv2.imshow('black', draw_image)
        return res_float

    def filter_black(self, mat):
        """
        return True if most pixel in mat is black
        """
        mat = np.copy(mat)
        mean_rgb = np.mean(mat, (0, 1))
        if np.mean(mean_rgb) < 50:
            return True
        else:
            return False

    def filter_white(self, mat):
        """
        return True if most pixel in mat is black
        """
        mat = np.copy(mat)
        mean_rgb = np.mean(mat, (0, 1))
        if np.mean(mean_rgb) > 130:
            return True
        else:
            return False

    def optimize_center(self, img, x, y):
        search_width = 5
        neighb_width = 11
        best_x, best_y = 0, 0
        min_color = 1000.

        for i in range(-search_width, search_width + 1):
            for j in range(-search_width, search_width + 1):
                mat = img[y + j - neighb_width: y + j + neighb_width + 1,
                      x + i - neighb_width: x + i + neighb_width + 1]
                if mat.shape[0] == mat.shape[1] == 23:
                    mat = mat * np.expand_dims(self.circle_mask, -1)
                    mean_mat = np.mean(mat)
                    if mean_mat < min_color:
                        best_x = i
                        best_y = j
                        min_color = mean_mat
        return x + best_x, y + best_y

    def register_new_chess(self, col, row, color):
        """
        register a new chess
        """
        assert [col, row] not in self.black_array and [col, row] not in self.white_array
        if color == 'black':
            print("Black array: {}".format(self.black_array))
            self.black_array.append([col, row])
        elif color == 'white':
            print("White array: {}".format(self.white_array))
            self.white_array.append([col, row])
        print("register {}: {}, {}".format(color, col, row))

    def one_more_chess(self, chess_quants, color):
        """
        judge whether chess_quants have one more chess than black_array(white array)
        color could only be 'black' or 'white'
        """
        if color == 'black':
            new_chess = [k for k in chess_quants if k not in self.black_array]
            # new_chess = list(set(chess_quants) - set(self.black_array))
            if len(new_chess) == 1 and \
                    len(chess_quants) - len(self.black_array) == 1:
                return new_chess[0]
            else:
                return None
        elif color == 'white':
            new_chess = [k for k in chess_quants if k not in self.white_array]
            if len(new_chess) == 1 and \
                    len(chess_quants) - len(self.white_array) == 1:
                return new_chess[0]
            else:
                return None
        else:
            raise ValueError

    def full2sub(self, quants):
        """
        convert the full chessboard quant coords to sub chessboard quant coords
        """
        return [[i[0] - self.sub_pad, i[1] - self.sub_pad] for i in quants]
