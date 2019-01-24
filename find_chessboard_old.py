import cv2
import numpy as np
import tensorflow as tf


def find_cont(img):
    cont, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_num = np.array([c.shape[0] for c in cont])
    cont_idx = np.argmax(cont_num)
    return cont[cont_idx: cont_idx+1]


def calc_dis(pq, ls):
    # pq is the distance between the point and one point in the line (point_num, line_num, 2)
    # ls is the unit vector of the line
    point_num, line_num, _ = pq.get_shape().as_list()
    zero_pad = tf.zeros([point_num, line_num, 1], dtype=tf.float32)
    pq = tf.concat((pq, zero_pad), 2)
    ls = tf.concat((ls, zero_pad), 2)
    cross_ret = tf.cross(pq, ls)
    dis = tf.reduce_sum(tf.square(cross_ret), -1)
    return dis


if __name__ == '__main__':

    # ref_img = cv2.imread('ref.jpg').astype(np.float32)
    dst_img = cv2.imread('ref.jpg').astype(np.float32)
    #
    # ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    # mask = np.logical_and(ref_hsv<45, 30<ref_hsv).astype(np.int)
    # ref_hsv *= mask
    # ref_hsv[ref_hsv>0] = 255
    # ref_hsv = cv2.threshold(ref_hsv, 45, 255, cv2.THRESH_BINARY)[1]
    dst_hsv = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    mask = np.zeros_like(dst_hsv)
    mask = np.logical_and(dst_hsv < 45, 30 < dst_hsv).astype(np.int)
    dst_hsv *= mask
    dst_hsv[dst_hsv > 0] = 255

    # cv2.imshow('ref', ref_hsv.astype(np.uint8))
    # cv2.imshow('dst', dst_hsv.astype(np.uint8))
    # cv2.waitKey()

    # flow = cv2.calcOpticalFlowFarneback(ref_hsv[:, :, 0], dst_hsv[:, :, 0], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # cont = find_cont(ref_hsv)
    # cv2.drawContours(ref_img, cont, -1, (255, 0, 0))
    # cv2.imshow('pred', ref_img.astype(np.uint8))
    # p = np.ones([100, 100], dtype=np.uint8)*255
    #
    # for vec in vecs:
    #     cv2.circle(p, tuple((vec*10+50).astype(int)), 3, (255, 0, 0))
    # cv2.imshow('a', p)
    # cv2.waitKey()
    # exit()
    cont = find_cont(dst_hsv)
    cnum = cont[0].shape[0]
    dst_img = dst_img.astype(np.uint8)
    cv2.drawContours(dst_img, cont, -1, (255, 0, 0))
    # cv2.imshow('pred_dst', dst_img)
    # cv2.waitKey()

    center = tf.Variable(np.array([320, 240], dtype=np.float32))
    deg_var = tf.Variable(0.01)
    side_d = tf.Variable(0.01)
    deg = tf.cast(deg_var, tf.float32)
    cont_point_ph = tf.placeholder(tf.float32, [cnum, 2])
    side_len = tf.add(side_d, 466.)
    half_diag = 444 * np.sqrt(2) / 2
    p1 = center + tf.stack([half_diag * tf.cos(np.pi / 4 + deg), -half_diag * tf.sin(np.pi / 4 + deg)], 0)
    p2 = center + tf.stack([half_diag * tf.cos(np.pi * 3 / 4 + deg), -half_diag * tf.sin(np.pi * 3 / 4 + deg)], 0)
    p3 = center + tf.stack([half_diag * tf.cos(np.pi * 5 / 4 + deg), -half_diag * tf.sin(np.pi * 5 / 4 + deg)], 0)
    p4 = center + tf.stack([half_diag * tf.cos(np.pi * 7 / 4 + deg), -half_diag * tf.sin(np.pi * 7 / 4 + deg)], 0)
    l1 = (p2 - p1) / tf.sqrt(tf.reduce_sum(tf.square(p2-p1)))
    l2 = (p3 - p2) / tf.sqrt(tf.reduce_sum(tf.square(p3-p2)))
    l3 = (p4 - p3) / tf.sqrt(tf.reduce_sum(tf.square(p4 - p3)))
    l4 = (p1 - p4) / tf.sqrt(tf.reduce_sum(tf.square(p1 - p4)))
    ps = tf.expand_dims(tf.stack([p1, p2, p3, p4], 0), 0)
    ls = tf.tile(tf.expand_dims(tf.stack([l1, l2, l3, l4], 0), 0), [cnum, 1, 1])
    cont_point_exp = tf.expand_dims(cont_point_ph, 1)
    pq = cont_point_exp - ps
    dis = calc_dis(pq, ls)
    min_dis = tf.reduce_min(dis, 1)

    def filter_noise(min_dis):
        sorted_dis = sorted(list(min_dis), reverse=True)
        thresh = sorted_dis[int(cnum*0.1)]
        mask = (min_dis < thresh).astype(np.float32)
        return mask

    mask = tf.py_func(filter_noise, [min_dis], [tf.float32])[0]
    mask.set_shape([cnum])
    min_dis = min_dis * mask

    loss = tf.reduce_mean(min_dis)
    optimizer = tf.train.AdamOptimizer(3e-1)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        _, loss_, p1_, p2_, p3_, p4_, deg_, min_dis_ = sess.run([train_op, loss, p1, p2, p3, p4, deg_var, min_dis], feed_dict={cont_point_ph: np.array(cont[0][:, 0, :])})
        print("step {}, loss: {}".format(i, loss_))
        print(deg_)
        if i % 100 == 0:
            show_img = np.copy(dst_img)
            cv2.line(show_img, tuple(p1_.astype(int)), tuple(p2_.astype(int)), (0, 255, 0), 2)
            cv2.line(show_img, tuple(p2_.astype(int)), tuple(p3_.astype(int)), (0, 255, 0), 2)
            cv2.line(show_img, tuple(p3_.astype(int)), tuple(p4_.astype(int)), (0, 255, 0), 2)
            cv2.line(show_img, tuple(p4_.astype(int)), tuple(p1_.astype(int)), (0, 255, 0), 2)
            cv2.imshow('a', show_img)
            cv2.waitKey()

