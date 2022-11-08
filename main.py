import copy
import math
from math import cos, sin

import numpy as np
from PIL import Image


def normalise(I_pos, il, jl):
    I_pos[:, 0] = I_pos[:, 0] / (il - 1)
    I_pos[:, 1] = I_pos[:, 1] / (jl - 1)
    return I_pos


def denormalise(I_pos, il, jl):
    I_pos[:, 0] = I_pos[:, 0] * (il - 1)
    I_pos[:, 1] = I_pos[:, 1] * (jl - 1)
    return I_pos


def simulate_views(I1, I2, t1, t2, ts, img_f):
    # get list of pixel coordinates for I1 and I2
    i_c, j_c = np.meshgrid(range(I1.shape[0]), range(I1.shape[1]), indexing='ij')
    pixel_pos = np.array(list(zip(j_c.reshape(-1), i_c.reshape(-1))), dtype=np.float64)
    I1_pos = normalise(copy.deepcopy(pixel_pos), I1.shape[1], I1.shape[0])
    I1_pos = np.c_[I1_pos, np.ones(I1_pos.shape[0])]  # homogeneous z
    I2_pos = normalise(copy.deepcopy(pixel_pos), I1.shape[1], I1.shape[0])
    I2_pos = np.c_[I2_pos, np.ones(I2_pos.shape[0])]  # homogeneous z
    R1_T = np.array([[cos(t1), 0, sin(t1)],
                     [0, 1, 0],
                     [-sin(t1), 0, cos(t1)]]).T
    R2_T = np.array([[cos(t2), 0, sin(t2)],
                     [0, 1, 0],
                     [-sin(t2), 0, cos(t2)]]).T
    for t in ts:
        # compute homographic transform onto I
        R = np.array([[cos(t), 0, sin(t)],
                      [0, 1, 0],
                      [-sin(t), 0, cos(t)]])
        H1 = R @ R1_T
        H2 = R @ R2_T
        q1 = H1 @ I1_pos.reshape(-1, 3, 1)
        q2 = H2 @ I2_pos.reshape(-1, 3, 1)
        q1 = q1.reshape(-1, 3)[:, :2]
        q2 = q2.reshape(-1, 3)[:, :2]
        q1 = np.round(denormalise(q1, I1.shape[1], I1.shape[0])).astype(int)
        q2 = np.round(denormalise(q2, I2.shape[1], I2.shape[0])).astype(int)
        # generate view I
        I = np.zeros(I1.shape, dtype=np.uint8)
        q1 = q1[(q1[:, 1] < I.shape[0]) & (q1[:, 1] >= 0)]
        q1 = q1[(q1[:, 0] < I.shape[1]) & (q1[:, 0] >= 0)]
        q2 = q2[(q2[:, 1] < I.shape[0]) & (q2[:, 1] >= 0)]
        q2 = q2[(q2[:, 0] < I.shape[1]) & (q2[:, 0] >= 0)]
        I[q1.T[1], q1.T[0], :] = I1[:, :q1.shape[0] // I1.shape[0], :].reshape(-1, 3)
        I[q2.T[1], q2.T[0], :] = I2[:, I2.shape[1] - q2.shape[0] // I2.shape[0]:, :].reshape(-1, 3)
        PI = Image.fromarray(I)
        PI.save('./out/' + img_f + '-t={:.2f}.jpg'.format(t))


if __name__ == '__main__':
    # load input
    img_f = '4'
    # original_img = np.asarray(Image.open('./data/' + img_f + '.jpg'))
    # oh, ow, chs = original_img.shape

    # divide input into I1 and I2, leaving 1/3 overlapping corresponding points
    # save cropped images
    # I1 = original_img[:, :ow * 2 // 3, :]
    # I2 = original_img[:, ow // 3:, :]
    I1 = np.asarray(Image.open('./test_pic/0.pic.jpg'))
    I2 = np.asarray(Image.open('./test_pic/10.pic.jpg'))
    PI1 = Image.fromarray(I1)
    PI2 = Image.fromarray(I2)
    PI1.save('./out/' + img_f + '-I1.jpg')
    PI2.save('./out/' + img_f + '-I2.jpg')

    # simulate theta generated image at 10 locations
    theta_1 = 0
    theta_2 = math.pi / 180 * 10
    thetas = np.arange(theta_1, theta_2+1e-5, (theta_2 - theta_1) / 20)
    simulate_views(I2, I1, theta_1, theta_2, thetas, img_f)

    # out
    print('Check the out folder for results.')
