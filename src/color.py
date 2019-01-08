import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os, sys

from math import *

IMG_DIR = "../img/"
OUT_DIR = "../out/"


def im2double(im):
    info = np.iinfo(im.dtype)                # Get the data type of the input image
    return im.astype(np.double) / info.max   # Divide all values by the largest possible value in the dtype


def double2im(im):
    im = im * 255
    return im.astype(np.uint8)


def colorize(img, mrk):
    # double type
    d_img = im2double(img)
    d_mrk = im2double(mrk)
    print("d_src.shape:", d_img.shape, d_img.dtype)

    # diff between src & mrk
    mark_map = np.sum((np.abs(d_img - d_mrk)), 2) > 0.01
    mark_map = mark_map.astype(np.float64)
    print("colorIm.shape:", mark_map.shape, mark_map.dtype)
    # print(colorIm)

    # to HSV space
    s_img = im2double(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    s_mrk = im2double(cv2.cvtColor(mrk, cv2.COLOR_BGR2HSV))
    print("s_img.shape:", s_img.shape, s_img.dtype)

    hsv_img = np.copy(s_mrk)
    hsv_img[:, :, 2] = s_img[:, :, 2]     # copy brightness

    # resize
    max_d = floor(log(min([hsv_img.shape[0], hsv_img.shape[1]]), 2) - 2)
    iu = int(floor(hsv_img.shape[0] / pow(2, max_d - 1)) * pow(2, max_d - 1))
    ju = int(floor(hsv_img.shape[1] / pow(2, max_d - 1)) * pow(2, max_d - 1))
    mark_map = mark_map[0:iu, 0:ju]
    hsv_img = hsv_img[0:iu, 0:ju, :]
    print("Resized:")
    print("colorIm.shape:", mark_map.shape)
    print("hsvIm.shape:", hsv_img.shape)

    # get color
    return set_color(mark_map, hsv_img)


def set_color(color, hsv):
    height = hsv.shape[0]
    width = hsv.shape[1]
    size = height * width

    # find color anchor
    element_idx = np.reshape([x for x in range(0, size)], [width, height]).transpose()
    mark_idx_pair = np.nonzero(color)
    mark_idx_linear = mark_idx_pair[1] * height + mark_idx_pair[0]

    wd = 1
    vec_len = int(size * pow(2 * wd + 1, 2))
    col_idx = np.zeros([vec_len, 1])
    row_idx = np.zeros([vec_len, 1])
    vals = np.zeros([vec_len, 1])
    gvals = np.zeros([1, int(pow(2 * wd + 1, 2))])

    # print(element_idx.shape, mark_idx_linear, col_idx.shape, gvals.shape)

    # calculating
    print("Calculating...")
    llen, consts_len = 0, 0
    for j in range(0, width):
        for i in range(0, height):
            # spread color
            if not color[i, j]:
                tlen = 0
                for ii in range(max((0, i - wd)), min((height, i + wd + 1))):
                    for jj in range(max((0, j - wd)), min((width, j + wd + 1))):
                        if ii != i or jj != j:
                            row_idx[llen] = consts_len
                            col_idx[llen] = element_idx[ii, jj]
                            gvals[0, tlen] = hsv[ii, jj, 2]       # brightness
                            llen += 1
                            tlen += 1

                tval = hsv[i, j, 2]
                gvals[0, tlen] = tval
                cvar = np.mean(np.power(gvals[0, 0:tlen+1] - np.mean(gvals[0, 0:tlen+1]), 2))
                csig = cvar * 0.6
                mgv = np.min(np.power(gvals[0, 0:tlen] - tval, 2))

                # control the spread of color
                if csig < (-mgv / log(0.01)):
                    csig = -mgv / log(0.01)
                if csig < 0.0000005:
                    csig = 0.0000005

                gvals[0, 0:tlen] = np.exp(-np.power(gvals[0, 0:tlen] - tval, 2) / csig)
                gvals[0, 0:tlen] = gvals[0, 0:tlen] / np.sum(gvals[0, 0:tlen])
                vals[llen-tlen:llen] = -np.reshape(gvals[0, 0:tlen], [tlen, 1])
            # end of not color(i,j)

            # set idx
            row_idx[llen] = consts_len
            col_idx[llen] = element_idx[i, j]
            vals[llen] = 1
            consts_len += 1
            llen += 1
    # end of for

    vals = np.reshape(vals[0:llen], -1)
    col_idx = np.reshape(col_idx[0:llen], -1)
    row_idx = np.reshape(row_idx[0:llen], -1)

    # sparse matrix, else too large
    A = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(consts_len, size))
    b = np.zeros([consts_len, 1])

    # write output
    out = np.copy(hsv)
    for t in range(0, 2):
        plain = hsv[:, :, t]
        b[mark_idx_linear] = np.reshape(plain[mark_idx_pair], [mark_idx_linear.shape[0], 1])
        res = spsolve(A, b)
        out[:, :, t] = np.reshape(res, [width, height]).transpose()

    out = double2im(out)
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\t%s src_img marked_img [output_img]" % sys.argv[0])
        exit(0)

    src_img = IMG_DIR + sys.argv[1] + ".bmp"
    mrk_img = IMG_DIR + sys.argv[2] + ".bmp"

    if len(sys.argv) == 4:
        out_img = OUT_DIR + sys.argv[-1] + ".bmp"
    else:
        out_img = OUT_DIR + sys.argv[1] + "_out.bmp"

    print("Reading src img: %s" % src_img)
    img = cv2.imread(src_img, cv2.IMREAD_COLOR)
    mrk = cv2.imread(mrk_img, cv2.IMREAD_COLOR)
    cv2.imshow("src", img)
    cv2.imshow("mark", mrk)
    out = colorize(img, mrk)

    cv2.imshow('out', out)
    cv2.imwrite(out_img, out)

    # finished
    cv2.waitKey()


if __name__ == '__main__':
    main()
