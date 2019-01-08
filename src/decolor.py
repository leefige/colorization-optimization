import cv2, sys
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\t%s src_img dst_img" % sys.argv[0])
        exit(0)

    src_img = "../img/" + sys.argv[1] + ".bmp"
    out_img = "../img/" + sys.argv[2] + ".bmp"

    img = cv2.imread(src_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 0] = np.zeros([img.shape[0], img.shape[1]])
    img[:, :, 1] = np.zeros([img.shape[0], img.shape[1]])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    cv2.imshow('out', img)
    cv2.imwrite(out_img, img)
    # finished
    cv2.waitKey()


if __name__ == '__main__':
    main()
