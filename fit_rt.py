import numpy as np
import tensorflow as tf
import cv2


if __name__ == '__main__':
    # src_coord = np.array([
    #     [119., 238], [230, 144], [339, 51], [208, 341], [318, 247], [427, 155], [297, 448], [407, 352], [517, 258]])
    src_coord = np.array([[72, 47], [143, 113], [285, 114], [428, 114], [215, 182], [357, 182], [285, 249],
                          [213, 317], [357, 318], [139, 388], [284, 386], [429, 387], [504, 459], [69, 250], [501, 249], [117, 448], [335, 441], [556, 437]], dtype=np.float32)
    src_coord -= np.array([[320, 240]], dtype=np.float32)
    src_coord /= np.array([[320, 240]], dtype=np.float32)
    annot = False
    if annot:
        coord = []
        def get_coord(event, x, y, flags, params):
            # print('move hand!')
            if event == cv2.EVENT_LBUTTONDOWN:
                coord.append([x, y])
                print len(coord)


        a_img = cv2.imread('annot2.jpg')
        cv2.namedWindow('annot', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('annot', get_coord)
        cv2.imshow('annot', a_img)
        k = cv2.waitKey()
        print coord
        exit()

    # src_coord[:, 0] *= (0.3145/320.)
    # src_coord[:, 1] *= (0.2276/240.)
    # src_coord = np.concatenate((src_coord, 0.61146*np.ones([9, 1])), 1)
    #
    # dst_coord = np.array([[-0.184860, -0.546631, 0.175872], [-0.327943, -0.535427, 0.179833],
    #                      [-0.470288, -0.524934, 0.181080],
    #                      [-0.175538, -0.411586, 0.176415], [-0.317580, -0.400943, 0.179436],
    #                      [-0.460004, -0.389861, 0.181078],
    #                      [-0.164628, -0.276345, 0.179116], [-0.309285, -0.265510, 0.181086],
    #                      [-0.451690, -0.254356, 0.181692]], dtype=np.float32)

    dst_coord = np.array([[-0.28066, -0.7170, 0.18], [-0.28433, -0.618265, 0.18],
                          [-0.387274, -0.519417, 0.18], [-0.4892, -0.4167, 0.18], [-0.28774, -0.51994, 0.18],
                          [-0.390706, -0.422247, 0.18], [-0.29203, -0.42349, 0.18], [-0.19268, -0.426557, 0.18],
                          [-0.295426, -0.321935, 0.18], [-0.09546, -0.4291, 0.18], [-0.19647, -0.327306, 0.18],
                          [-0.29941, -0.224357, 0.18], [-0.30343, -0.12624, 0.18],
                          [-0.13754, -0.5721, 0.18], [-0.4455, -0.271525, 0.18], [-0.038514, -0.397597, 0.18],
                          [-0.195868, -0.25191, 0.18], [-0.35391, -0.10458, 0.18]], dtype=np.float32)

    # src_coord = src_coord[-3:]
    # dst_coord = dst_coord[-3:]

    # src_i = np.ones([480, 640, 3], np.uint8)*255
    # dst_i = np.ones([400, 400, 3 ], np.uint8) * 255
    # for sp,dp in zip(src_coord, dst_coord):
    #     dp = (400*(dp+0.9)).astype(int)[:2]
    #     cv2.circle(src_i, tuple(sp), 3, (255, 0, 0), 2)
    #     cv2.circle(dst_i, tuple(dp), 3, (255, 0, 0), 2)
    #     cv2.imshow('src', src_i)
    #     cv2.imshow('dst', dst_i)
    #     cv2.waitKey()
    # exit()

    # dst_coord -= np.array([[-0.2863222836800378, -0.3641882948297106, 0.6854579419026046]])
    # dst_coord[:, 1:] *= -1

    # src_coord = src_coord[:16]
    # dst_coord = dst_coord[:16]

    R = tf.Variable(np.eye(3, dtype=np.float32))
    T = tf.Variable(np.zeros([3], dtype=np.float32))
    xf = tf.Variable(0.3145, trainable=False)
    yf = tf.Variable(0.2276, trainable=False)
    # bs = len(dst_coord)
    bs = 1

    src_coord_ph = tf.placeholder(tf.float32, [bs, 2])
    dst_coord_ph = tf.placeholder(tf.float32, [bs, 3])

    # intrinsic
    two_order = False
    intri = tf.Variable(np.array([[0.3145, 0., 0., ], [0., 0.2276, 0.], [0., 0., 1.]], dtype=np.float32))

    src_coord_ = src_coord_ph * tf.stack([tf.ones([1]), -1 * tf.ones([1])], 1)

    # src_coord_ = tf.stack([src_coord_[:, 0]*xf, src_coord_[:, 1]*yf, -0.61146*tf.ones([bs])], 1)
    src_coord_ = tf.stack([src_coord_[:, 0], src_coord_[:, 1], -0.61146 * tf.ones([bs])], 1)
    cam_coord = tf.einsum('uv,nv->nu', intri, src_coord_)

    if two_order:
        src_coord_2_ = tf.square(src_coord_)
        intri2 = tf.Variable(np.array([[0.3145, 0., 0., ], [0., 0.2276, 0.], [0., 0., 1.]], dtype=np.float32))
        cam_coord += tf.einsum('uv,nv->nu', intri2, src_coord_2_)


    # extrinsic
    pred_coord = tf.einsum('uv,nv->nu', R, cam_coord) + tf.expand_dims(T, 0)

    pred_coord -= tf.constant([[-0.2863222836800378, -0.3641882948297106, 0.6854579419026046]], dtype=tf.float32)

    loss = tf.reduce_sum(tf.square((dst_coord_ph - pred_coord)[:, :3]))
    dis = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((dst_coord_ph-pred_coord)[:, :2]), -1)))
    opti = tf.train.AdamOptimizer(1e-2)
    train_op = opti.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    min_dis = 100.
    for i in range(3000):
        idx = np.random.randint(len(dst_coord))
        if bs == 1:
            feed_dict = {dst_coord_ph: dst_coord[idx:idx+1], src_coord_ph: src_coord[idx:idx+1]}
        else:
            feed_dict = {dst_coord_ph: dst_coord, src_coord_ph: src_coord}
        R_, T_, _, loss_, xf_, yf_, intr_, dis_ = sess.run([R, T, train_op, loss, xf, yf, intri, dis], feed_dict=feed_dict)
        # print("step {}, loss: {}, dist: {}".format(i, loss_, dis_))
        if i % 10 == 0:
            total_loss, total_dis = 0., 0.
            if bs == 1:
                for b in range(len(dst_coord)):
                    loss_, dis_, pred_coord_ = sess.run([loss, dis, pred_coord], feed_dict={dst_coord_ph: dst_coord[b: b+1], src_coord_ph: src_coord[b: b+1]})
                    total_loss += loss_
                    total_dis += dis_
                    # print "{}: {}".format(b, dis_)
                total_loss = total_loss / len(dst_coord)
                total_dis = total_dis/len(dst_coord)

            else:
                total_loss, total_dis = sess.run([loss, dis], feed_dict={dst_coord_ph: dst_coord, src_coord_ph: src_coord})
            if total_dis < min_dis:
                min_R, min_T, min_xf, min_yf, min_intr = R_, T_, xf_, yf_, intr_
                min_dis = total_dis
            print("total loss: {} total_dis: {}".format(total_loss, total_dis))
    print(min_R)
    print(min_T)
    print(min_xf)
    print(min_yf)
    print(min_intr)

