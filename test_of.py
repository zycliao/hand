# coding=utf-8
import numpy as np
from camera import Camera
import cv2

if __name__ == '__main__':
    # i1 = cv2.imread('ref.jpg')
    # i2 = cv2.imread('ref1.jpg')
    # i1_grey = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    # i2_grey = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    # corner_coord = [(19, 78), (459, 542)]

    # h, w, _ = i1.shape
    camera=Camera()
    new_array=camera.get_frame()
    # image=cv2.imread('对角.jpg')

    #cv2.circle(image, (296, 244), 7, (0,0,0),-1)
    #cv2.imshow('a',image)
    #cv2.waitKey()





    exit()

    new_i = np.zeros_like(i1)
    for hh in range(h):
        for ww in range(w):
            dx, dy = flow[hh, ww]
            if 0 < int(hh+dy) < h and 0 < int(ww+dx) < w:

                new_i[hh, ww] = i1[int(hh+dy), int(ww+dx)]
    cv2.imshow('a', new_i)
    cv2.waitKey()