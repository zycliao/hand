# coding=utf-8
import cv2
import numpy as np


def intersection(r1, t1, r2, t2):
    c1 = np.cos(t1)
    s1 = np.sin(t1)
    c2 = np.cos(t2)
    s2 = np.sin(t2)
    y = (r2-r1*c2/c1)/(s2-c1*c2/c1)
    x = (r1-y*s1)/c1
    return x, y


def find_cont(img):
    img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('a', img)
    cv2.waitKey()
    cont, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_num = np.array([c.shape[0] for c in cont])
    cont_idx = np.argmax(cont_num)
    return cont[cont_idx: cont_idx+1]


if __name__ == '__main__':

    ref_img = cv2.imread('对角.jpg').astype(np.float32)
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    mask = np.zeros_like(ref_hsv)
    mask = np.logical_and(ref_hsv<45, 30<ref_hsv).astype(np.int)
    mask2 = np.logical_and(ref_img[:, :, 0]<115, ref_img[:, :, 0]>30)
    mask3 = np.mean(ref_img, -1)>100
    mask = np.logical_and(mask, mask2)
    mask = np.logical_and(mask, mask3)
    # ref_hsv *= mask
    ref_hsv[ref_hsv>0] = 255
    masked_img = ref_img * np.expand_dims(mask, -1)
    cont = find_cont(mask)

    ref_img = ref_img.astype(np.uint8)
    cv2.imshow('ref', ref_img.astype(np.uint8))

    black_img = np.zeros_like(masked_img)
    edges = cv2.drawContours(black_img, cont, -1, (255, 255, 255)).astype(np.uint8)
    # m = mask.astype(np.uint8)*255
    # edges = cv2.Canny(m, 50, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    cv2.imshow('edges', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)[:, 0, :]
    # lines = [k for k in lines if k[1] != 0 and np.abs(k[1]-1.5708)>0.001]
    # lines=lines[1: 2]
    line_num = len(lines)
    draw_line = False
    if draw_line:
        for line in lines:
            rho = line[0]
            theta = line[1]
            print rho
            print theta
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
                pt1 = (int(rho / np.cos(theta)), 0)
                pt2 = (int((rho - ref_img.shape[0] * np.sin(theta)) / np.cos(theta)), ref_img.shape[0])
                cv2.line(ref_img, pt1, pt2, (255))
            else:
                pt1 = (0, int(rho / np.sin(theta)))
                pt2 = (ref_img.shape[1], int((rho - ref_img.shape[1] * np.cos(theta)) / np.sin(theta)))
                cv2.line(ref_img, pt1, pt2, (255), 1)

    # line_gen = []
    cross_p = []
    # for line in lines:
    #     line_gen.append(general_equ(line[0], line[1]))
    for i in range(line_num):
        for j in range(line_num):
            theta1 = lines[i][1]
            theta2 = lines[j][1]
            rho1 = lines[i][0]
            rho2 = lines[j][0]
            theta_diff = theta1 - theta2
            if theta_diff<-np.pi/2:
                theta_diff += np.pi
            if theta_diff>np.pi/2:
                theta_diff -= np.pi
            if np.abs(theta1 - theta2) < (np.pi/9):
                continue
            if np.abs(theta1 - theta2) > (8*np.pi/9):
                continue
            # p = intersection(line_gen[i], line_gen[j])
            p = intersection(rho1, theta1, rho2, theta2)
            if 0 < p[0] < 640 and 0 < p[1] < 480:
                cross_p.append((int(p[0]), int(p[1])))

    # left up
    cp = np.array(cross_p)
    xy = np.sum(cp, -1)
    lu = np.argmin(xy)
    # right down
    rd = np.argmax(xy)
    xfy = cp[:, 0]-cp[:,1]
    ru = np.argmax(xfy)
    ld = np.argmin(xfy)

    # print line_gen
    # cv2.circle(ref_img, tuple(cp[lu]), 4, (0, 0, 255), 3)
    # cv2.circle(ref_img, tuple(cp[rd]), 4, (0, 255, 255), 3)
    # cv2.circle(ref_img, tuple(cp[ru]), 4, (255, 0, 255), 3)
    # cv2.circle(ref_img, tuple(cp[ld]), 4, (255, 255, 255), 3)
    cv2.circle(ref_img, tuple(cp[lu]+[20, 15]), 4, (0, 0, 255), 3)
    cv2.circle(ref_img, tuple(cp[rd]+[-20, -15]), 4, (0, 255, 255), 3)
    cv2.circle(ref_img, tuple(cp[ru]+[-20, 15]), 4, (255, 0, 255), 3)
    cv2.circle(ref_img, tuple(cp[ld]+[20, -15]), 4, (255, 255, 255), 3)
    # for p in cross_p:
    #     if 0<p[0]<640 and 0<p[1]<480:
    #         cv2.circle(ref_img, p, 2, (0, 255, 0), 1)

    cv2.imshow('pred', ref_img)
    cv2.waitKey()
    exit()