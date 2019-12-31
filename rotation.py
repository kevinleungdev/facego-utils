import os
import cv2

from os.path import join, isdir, splitext

DEFAULT_DATA_DIR = 'D:/Faces/data'

for cls in os.listdir(DEFAULT_DATA_DIR):
    sub_dir = join(DEFAULT_DATA_DIR, cls)

    if isdir(sub_dir):
        # list image files
        for item in os.listdir(sub_dir):
            basename, ext = splitext(item)

            if ext in ('.jpg', '.jpeg', '.png'):
                img = cv2.imread(join(sub_dir, item))

                # flip image horizontally
                horizontal_img = cv2.flip(img, 1)

                # write new image to disk
                new_img = basename + '_h' + ext
                print 'Save new image: ', new_img

                cv2.imwrite(join(sub_dir, new_img), horizontal_img)
