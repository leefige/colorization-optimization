import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, sys

IMG_DIR = "../img/"
OUT_DIR = "../out/"


def im2double(im):
    info = np.iinfo(im.dtype)                # Get the data type of the input image
    return im.astype(np.double) / info.max   # Divide all values by the largest possible value in the dtype


def colorize(img, mrk):
    # double type
    d_img = im2double(img)
    print("d_src.shape:", d_img.shape, d_img.dtype)
    # print(img)

    d_mrk = im2double(mrk)
    # print(mrk)

    # diff between src & mrk
    colorIm = np.sum((np.abs(d_img - d_mrk)), 2) > 0.01
    colorIm = colorIm.astype(np.float64)
    print("colorIm.shape:", colorIm.shape, colorIm.dtype)
    # print(colorIm)

    # to YUV space
    s_img = im2double(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
    s_mrk = im2double(cv2.cvtColor(mrk, cv2.COLOR_RGB2YUV))
    print("s_img.shape:", s_img.shape, s_img.dtype)



def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\t%s src_img marked_img [output_img]" % sys.argv[0])
        exit(0)

    src_img = IMG_DIR + sys.argv[1] + ".bmp"
    mrk_img = IMG_DIR + sys.argv[2] + ".bmp"

    if len(sys.argv) == 4:
        out_img = IMG_DIR + sys.argv[-1] + ".bmp"
    else:
        out_img = IMG_DIR + sys.argv[1] + "_out.bmp"

    print("Reading src img: %s" % src_img)
    img = cv2.imread(src_img, cv2.IMREAD_COLOR)
    mrk = cv2.imread(mrk_img, cv2.IMREAD_COLOR)

    colorize(img, mrk)

    cv2.imshow('src', img)

    # finished
    cv2.waitKey()


if __name__ == '__main__':
    main()