import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    src_coord = np.array([
        [119., 238], [230, 144], [339, 51], [208, 341], [318, 247], [427, 155], [297, 448], [407, 352], [517, 258]])
    src_coord -= np.array([[320, 240]], dtype=np.float32)
    src_coord[:, 0] *= (0.3145/320.)
    src_coord[:, 1] *= (0.2276/240.)
    src_coord = np.concatenate((src_coord, 0.61146*np.ones([9, 1])), 1)

    dst_coord = np.array([[-0.184860, -0.546631, 0.175872], [-0.327943, -0.535427, 0.179833],
                         [-0.470288, -0.524934, 0.181080],
                         [-0.175538, -0.411586, 0.176415], [-0.317580, -0.400943, 0.179436],
                         [-0.460004, -0.389861, 0.181078],
                         [-0.164628, -0.276345, 0.179116], [-0.309285, -0.265510, 0.181086],
                         [-0.451690, -0.254356, 0.181692]], dtype=np.float32)

    dst_coord -= np.array([[-0.2863222836800378, -0.3641882948297106, 0.6854579419026046]])
    dst_coord[:, 1:] *= -1

    R = tf.Variable(np.eye(3, dtype=np.float32))
    T = tf.Variable(np.zeros([3], dtype=np.float32))
    src_coord_ph = tf.placeholder(tf.float32, [1, 3])
    dst_coord_ph = tf.placeholder(tf.float32, [1, 3])

    pred_coord = tf.einsum('uv,nv->nu', R, src_coord_ph) + tf.expand_dims(T, 0)
    loss = tf.reduce_sum(tf.square(dst_coord_ph - pred_coord))
    opti = tf.train.AdamOptimizer(1e-1)
    train_op = opti.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(300):
        idx = np.random.randint(9)
        feed_dict = {dst_coord_ph: dst_coord[idx: idx+1], src_coord_ph: src_coord[idx: idx+1]}
        R_, T_, _, loss_ = sess.run([R, T, train_op, loss], feed_dict=feed_dict)
        print("step {}, loss: {}".format(i, loss_))
        if i % 10 == 0:
            print(R_)
            print(T_)
