# import cv2
import numpy as np

intrisinc = [[6.011667775530881e+02, 0, 3.223857746398750e+02, 0],
             [0, 6.023642035372599e+02, 2.310343184963895e+02, 0], [0, 0, 1, 0]]
intrisinc = np.array(intrisinc)

# -0.377940875184106 0.342232882889716 -0.947695031131854
Q = np.mat('-0.007069939682902   0.007067231898927  -0.000264926719918  -0.247940875184106; \
     -0.007071711057244  -0.007060107484196   0.000381818064614   0.250232882889716;\
     0.000082798568777   0.000457291590126   0.009989195603181   -0.977695031131854;\
     0 0 0 0.01')

Q *= 100.

P = np.dot(intrisinc, Q)

####################
intrisinc = [[6.011667775530881e+02, 0, 3.223857746398750e+02],
             [0, 6.023642035372599e+02, 2.310343184963895e+02], [0, 0, 1]]
intrisinc = np.array(intrisinc)

k1 = None
k2 = None


def cam2tool(u, v, cur_x, cur_y, Zc=600.46):

    tmp_position = Zc * np.dot(np.linalg.pinv(intrisinc), np.array([u, v, 1]).reshape(3, 1))
    # print(tmp_position)
    tmp_position = np.concatenate([tmp_position, np.array([[1]])])
    # flange coordinate
    res = np.dot(Q, tmp_position)
    return [float(res[0][0]/1000.)+cur_x, -float(res[1][0]/1000.)+cur_y, 0.18]


if __name__ == '__main__':
    print(cam2tool(10, 240, 0.23, 0.35))
