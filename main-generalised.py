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


def simulate_views(I1, I2, I3, I4, t1, t2, g1, g2, ts, gs, img_f):
    # get list of pixel coordinates for I1 and I2
    i_c, j_c = np.meshgrid(range(I1.shape[0]), range(I1.shape[1]), indexing='ij')
    pixel_pos = np.array(list(zip(j_c.reshape(-1), i_c.reshape(-1))), dtype=np.float64)
    I1_pos = normalise(copy.deepcopy(pixel_pos), I1.shape[1], I1.shape[0])
    I1_pos = np.c_[I1_pos, np.ones(I1_pos.shape[0])]  # homogeneous z
    I2_pos = copy.deepcopy(I1_pos)
    I3_pos = copy.deepcopy(I1_pos)
    I4_pos = copy.deepcopy(I1_pos)
    R1y_T = np.array([[cos(t1), 0, sin(t1)],
                      [0, 1, 0],
                      [-sin(t1), 0, cos(t1)]]).T
    R2y_T = np.array([[cos(t2), 0, sin(t2)],
                      [0, 1, 0],
                      [-sin(t2), 0, cos(t2)]]).T
    R1x_T = np.array([[1, 0, 0],
                      [0, cos(g1), -sin(g1)],
                      [0, sin(g1), cos(g1)]]).T
    R2x_T = np.array([[1, 0, 0],
                      [0, cos(g2), -sin(g2)],
                      [0, sin(g2), cos(g2)]]).T

    for t in ts:
        for g in gs:
            # compute homographic transform onto I
            Ry = np.array([[cos(t), 0, sin(t)],
                           [0, 1, 0],
                           [-sin(t), 0, cos(t)]])
            Rx = np.array([[1, 0, 0],
                           [0, cos(g), -sin(g)],
                           [0, sin(g), cos(g)]])
            H1 = Ry @ R1y_T
            H2 = Ry @ R2y_T
            H3 = Ry @ R1y_T
            H4 = Ry @ R2y_T
            q1 = H1 @ I1_pos.reshape(-1, 3, 1)
            q2 = H2 @ I2_pos.reshape(-1, 3, 1)
            q3 = H3 @ I3_pos.reshape(-1, 3, 1)
            q4 = H4 @ I4_pos.reshape(-1, 3, 1)
            q1 = q1.reshape(-1, 3)[:, :2]
            q2 = q2.reshape(-1, 3)[:, :2]
            q3 = q3.reshape(-1, 3)[:, :2]
            q4 = q4.reshape(-1, 3)[:, :2]
            q1 = np.round(denormalise(q1, I1.shape[1], I1.shape[0])).astype(int)
            q2 = np.round(denormalise(q2, I2.shape[1], I2.shape[0])).astype(int)
            q3 = np.round(denormalise(q3, I3.shape[1], I3.shape[0])).astype(int)
            q4 = np.round(denormalise(q4, I4.shape[1], I4.shape[0])).astype(int)
            # generate view I1-I2 and view I3-I4
            I1_I2 = np.zeros(I1.shape, dtype=np.uint8)
            I3_I4 = np.zeros(I1.shape, dtype=np.uint8)
            q10 = q1.T[0][0]
            q20 = q2.T[0][0]
            q30 = q3.T[0][0]
            q40 = q4.T[0][0]
            q1 = q1[(q1[:, 1] < I1_I2.shape[0]) & (q1[:, 1] >= 0)]
            q1 = q1[(q1[:, 0] < I1_I2.shape[1]) & (q1[:, 0] >= 0)]
            q2 = q2[(q2[:, 1] < I1_I2.shape[0]) & (q2[:, 1] >= 0)]
            q2 = q2[(q2[:, 0] < I1_I2.shape[1]) & (q2[:, 0] >= 0)]
            q3 = q3[(q3[:, 1] < I1_I2.shape[0]) & (q3[:, 1] >= 0)]
            q3 = q3[(q3[:, 0] < I1_I2.shape[1]) & (q3[:, 0] >= 0)]
            q4 = q4[(q4[:, 1] < I1_I2.shape[0]) & (q4[:, 1] >= 0)]
            q4 = q4[(q4[:, 0] < I1_I2.shape[1]) & (q4[:, 0] >= 0)]
            I1_I2[q1.T[1], q1.T[0], :] = I1[q1.T[1], q1.T[0] - q10, :].reshape(-1, 3)
            I1_I2[q2.T[1], q2.T[0], :] = I2[q2.T[1], q2.T[0] - q20, :].reshape(-1, 3)
            I3_I4[q3.T[1], q3.T[0], :] = I3[q3.T[1], q3.T[0] - q30, :].reshape(-1, 3)
            I3_I4[q4.T[1], q4.T[0], :] = I4[q4.T[1], q4.T[0] - q40, :].reshape(-1, 3)

            # generate I from I1-I2 and I3-I4
            I = np.zeros(I1.shape, dtype=np.uint8)
            H12 = Rx @ R1x_T
            H34 = Rx @ R2x_T
            q1 = H12 @ I1_pos.reshape(-1, 3, 1)
            q2 = H34 @ I2_pos.reshape(-1, 3, 1)
            q1 = q1.reshape(-1, 3)[:, :2]
            q2 = q2.reshape(-1, 3)[:, :2]
            q1 = np.round(denormalise(q1, I1_I2.shape[1], I1_I2.shape[0])).astype(int)
            q2 = np.round(denormalise(q2, I3_I4.shape[1], I3_I4.shape[0])).astype(int)
            q11 = q1.T[1][0]
            q21 = q2.T[1][0]
            q1 = q1[(q1[:, 1] < I1_I2.shape[0]) & (q1[:, 1] >= 0)]
            q1 = q1[(q1[:, 0] < I1_I2.shape[1]) & (q1[:, 0] >= 0)]
            q2 = q2[(q2[:, 1] < I1_I2.shape[0]) & (q2[:, 1] >= 0)]
            q2 = q2[(q2[:, 0] < I1_I2.shape[1]) & (q2[:, 0] >= 0)]
            I[q1.T[1], q1.T[0], :] = I1_I2[q1.T[1] - q11, q1.T[0], :].reshape(-1, 3)
            I[q2.T[1], q2.T[0], :] = I3_I4[q2.T[1] - q21, q2.T[0], :].reshape(-1, 3)

            PI = Image.fromarray(I)
            PI.save('./out/' + img_f + '-t={:.2f}-g={:.2f}.jpg'.format(t, g))


if __name__ == '__main__':
    # load input
    img_f = '6'
    original_img = np.asarray(Image.open('./data/' + '3.jpg'))
    oh, ow, chs = original_img.shape

    # divide input into I1 and I2, leaving 1/3 overlapping corresponding points
    # save cropped images
    I1 = original_img[oh * 1 // 3:, :ow * 2 // 3, :]
    I2 = original_img[oh * 1 // 3:, ow // 3:, :]
    I3 = original_img[:oh * 2 // 3, :ow * 2 // 3, :]
    I4 = original_img[:oh * 2 // 3, ow // 3:, :]
    # I1 = np.asarray(Image.open('./test_pic/4stone/angle-15_-15.jpg'))
    # I2 = np.asarray(Image.open('./test_pic/4stone/angle15_-15.jpg'))
    # I3 = np.asarray(Image.open('./test_pic/4stone/angle-15_15.jpg'))
    # I4 = np.asarray(Image.open('./test_pic/4stone/angle15_15.jpg'))
    PI1 = Image.fromarray(I1)
    PI2 = Image.fromarray(I2)
    PI1.save('./out/' + img_f + '-I1.jpg')
    PI2.save('./out/' + img_f + '-I2.jpg')
    PI3 = Image.fromarray(I3)
    PI4 = Image.fromarray(I4)
    PI3.save('./out/' + img_f + '-I3.jpg')
    PI4.save('./out/' + img_f + '-I4.jpg')

    # simulate theta generated image at 10 locations
    theta_1 = 0
    gamma_1 = 0
    # theta_2 = math.pi / 180 * 29.225
    theta_2 = math.pi / 180 * 29.225
    gamma_2 = math.pi / 180 * 29.225
    thetas = np.arange(theta_1, theta_2 + 1e-5, (theta_2 - theta_1) / 5)
    gammas = np.arange(gamma_1, gamma_2 + 1e-5, (gamma_2 - gamma_1) / 5)
    simulate_views(I4, I3, I2, I1, theta_1, theta_2, gamma_1, gamma_2, thetas, gammas, img_f)

    # out
    print('Check the out folder for results.')
