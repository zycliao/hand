# coding=utf-8
import cv2
import numpy as np

EPS = 1e-6

def intersection(r1, t1, r2, t2):
    c1 = np.cos(t1)
    s1 = np.sin(t1)
    c2 = np.cos(t2)
    s2 = np.sin(t2)
    # y = (r2-r1*c2/c1)/(EPS+s2-c1*c2/c1)
    # x = (r1-y*s1)/(c1+EPS)
    x = (r1*s2-r2*s1)/(np.sin(t2-t1))
    y = r2/s2-x*c2/s2
    return x, y


def find_cont(img, get_max=True):
    img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cont, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if get_max:
        max_span = np.array([np.max(c[:, 0, 0])-np.min(c[:, 0, 0]) for c in cont])
        # cont_num = np.array([c.shape[0] for c in cont])
        cont_idx = np.argmax(max_span)
        cont = cont[cont_idx: cont_idx+1]
    return cont


def find_mask(img):
    img = img.astype(np.float32)
    ref_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    mask = np.logical_and(ref_hsv < 45, 30 < ref_hsv).astype(np.int)
    mask2 = np.logical_and(img[:, :, 0] < 115, img[:, :, 0] > 30)
    mask3 = np.mean(img, -1) > 100
    mask = np.logical_and(mask, mask2)
    mask = np.logical_and(mask, mask3)
    mask = mask.astype(np.uint8)
    return mask


def find_chessboard(ref_img, debug=False):
    mask = find_mask(ref_img)
    # ref_hsv *= mask
    masked_img = ref_img * np.expand_dims(mask, -1)
    cont = find_cont(mask * 255)
    black_img = np.zeros_like(masked_img)
    cont_img = cv2.drawContours(black_img, cont, -1, (255, 255, 255)).astype(np.uint8)

    if debug:
        pass
        # cv2.imshow('mask', mask * 255)
        # cv2.imshow('ref', ref_img)
        # cv2.imshow('cont', black_img)
    cont_img = cv2.cvtColor(cont_img, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLines(cont_img, 1, np.pi / 180, 118)[:, 0, :]

    # lines = lines[:2]
    line_num = len(lines)
    if debug:
        for line in lines:
            rho = line[0]
            theta = line[1]
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
            if theta_diff < -np.pi / 2:
                theta_diff += np.pi
            if theta_diff > np.pi / 2:
                theta_diff -= np.pi
            if np.abs(theta1 - theta2) < (np.pi / 9):
                continue
            if np.abs(theta1 - theta2) > (8 * np.pi / 9):
                continue
            # p = intersection(line_gen[i], line_gen[j])
            p = intersection(rho1, theta1, rho2, theta2)
            if 0 < p[0] < 640 and 0 < p[1] < 480:
                cross_p.append(np.array([p[0], p[1]]))

    for c in cross_p:
        # print(c)
        cv2.circle(ref_img, tuple(c.astype(int)), 2, (0, 255, 0), 1)

    # left up
    assert len(cross_p) > 0
    cp = np.array(cross_p)
    xy = np.sum(cp, -1)
    lu = np.argmin(xy)
    # right down+
    rd = np.argmax(xy)
    xfy = cp[:, 0] - cp[:, 1]
    ru = np.argmax(xfy)
    ld = np.argmin(xfy)

    inner_lu = cp[lu] + [20, 15]
    inner_rd = cp[rd] + [-25, -20]
    inner_ru = cp[ru] + [-20, 15]
    inner_ld = cp[ld] + [20, -15]

    if debug:
        cv2.circle(ref_img, tuple(cp[lu].astype(int)), 2, (0, 0, 255), 3)
        cv2.circle(ref_img, tuple(cp[ru].astype(int)), 2, (0, 255, 255), 3)
        cv2.circle(ref_img, tuple(cp[rd].astype(int)), 2, (255, 0, 255), 3)
        cv2.circle(ref_img, tuple(cp[ld].astype(int)), 2, (255, 255, 255), 3)
        # cv2.circle(ref_img, tuple(np.mean(np.stack([cp[lu], cp[ld], cp[ru], cp[rd]], 0), 0).astype(int)), 2,
        #            (255, 255, 255), 3)
        cv2.circle(ref_img, tuple(inner_lu.astype(int)), 4, (0, 0, 255), 3)
        cv2.circle(ref_img, tuple(inner_ru.astype(int)), 4, (0, 255, 255), 3)
        cv2.circle(ref_img, tuple(inner_rd.astype(int)), 4, (255, 0, 255), 3)
        cv2.circle(ref_img, tuple(inner_ld.astype(int)), 4, (255, 255, 255), 3)
        # for p in cross_p:
        #     if 0<p[0]<640 and 0<p[1]<480:
        #         cv2.circle(ref_img, p, 2, (0, 255, 0), 1)

        cv2.imshow('pred', ref_img)
    return inner_lu, inner_ru, inner_rd, inner_ld


if __name__ == '__main__':

    ref_img = cv2.imread('对角.jpg')
    find_chessboard(ref_img, debug=True)