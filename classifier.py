# -*- coding: utf-8 -*-

import os
import cv2
import sys
import argparse
import numpy as np
import cPickle as pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import isdir, join, splitext
from mego.face import api

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEFAULT_DATA_DIR = "D:/Faces/data"
DEFAULT_MODEL_PATH = "D:/Faces/classifier.pkl"


def split_dataset(data_dir, min_nrof_images_per_class, nrof_train_images_per_class):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for cls in os.listdir(data_dir):
        sub_dir = join(data_dir, cls)

        # skip files in data directory
        if not isdir(sub_dir):
            continue

        # count number
        cnt = 0

        for img in os.listdir(sub_dir):
            _, ext = splitext(img)

            if ext not in ('.jpg', '.jpeg', '.png'):
                continue

            img_path = join(sub_dir, img)
            img_buf = cv2.imread(img_path)

            rgb_img = cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB)

            # detect faces
            face_locations = api.detect_faces(rgb_img)

            # detect face feature
            if len(face_locations) == 1:
                # face encodings
                reps = api.face_encodings(rgb_img, known_face_locations=face_locations)[0]

                # log some info
                print 'cls: {}, reps[0:5]: {}'.format(cls, reps[0:5])

                if cnt < nrof_train_images_per_class:
                    train_x.append(reps)
                    train_y.append(cls)
                else:
                    test_x.append(reps)
                    test_y.append(cls)

                cnt += 1
            else:
                print 'No face or more than one face detected: ', img_path

        if cnt < min_nrof_images_per_class:
            print '{} class only have {} pics'.format(cls, cnt)

    return train_x, train_y, test_x, test_y


def get_data(data_dir):
    x = []
    y = []

    for cls in os.listdir(data_dir):
        sub_dir = join(data_dir, cls)

        # skip files in data directory
        if not isdir(sub_dir):
            continue

        for img in os.listdir(sub_dir):
            _, ext = splitext(img)

            if ext not in ('.jpg', '.jpeg', '.png'):
                continue

            img_path = join(sub_dir, img)
            img_buf = cv2.imread(img_path)

            rgb_img = cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB)

            # detect faces
            face_locations = api.detect_faces(rgb_img)

            # detect face feature
            if len(face_locations) == 1:
                # face encodings
                reps = api.face_encodings(rgb_img, known_face_locations=face_locations)[0]
                # log some info
                print 'cls: {}, reps[0:5]: {}'.format(cls, reps[0:5])

                x.append(reps)
                y.append(cls)
            else:
                print 'No face or more than one face detected: ', img_path

    return x, y


def train_svm(data_dir, min_nrof_images_per_class, nrof_train_images_per_class, classifier_filename):
    # split train dataset and test dataset
    train_x, train_y, test_x, test_y = split_dataset(data_dir, min_nrof_images_per_class, nrof_train_images_per_class)

    class_names = list(np.unique(train_y))
    print '{} classes, {} train samples, {} test samples'.format(len(class_names), len(train_x), len(test_x))

    # start to training classifier
    print 'Training classifier'

    import time
    start = time.time()

    # model = SVC(kernel='linear', probability=True)
    # model.fit(train_x, train_y)

    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']}
    ]
    model = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5).fit(train_x, train_y)

    duration = (time.time() - start) / 1000
    print 'Train classifier done! it takes %d seconds' % duration

    # test classifier
    if len(test_x) > 0 and len(test_y) > 0:
        print 'Testing classifier, test_y: {}'.format(test_y)

        predictions = model.predict_proba(test_x)
        print 'predictions: {}, shape: {}'.format(predictions, predictions.shape)

        best_class_indices = np.argmax(predictions, axis=1)
        print 'best_class_indices: {}'.format(best_class_indices)

        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        test_indices = [class_names.index(cls) for cls in test_y]
        accuracy = np.mean(np.equal(best_class_indices, test_indices))

        print('Accuracy: %.3f' % accuracy)

    # saving classifier model
    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print 'Saved classifier model to file "%s"' % classifier_filename


def show_tsne(data_dir):
    x, y = get_data(data_dir)

    if len(x) == 0 or len(y) == 0:
        print 'No data'
        return

    x_pca = PCA(n_components=50).fit_transform(x, x)
    tsne = TSNE(n_components=2, init='random', random_state=0)
    x_r = tsne.fit_transform(x_pca)

    y_vals = list(np.unique(y))
    colors = cm.rainbow(np.linspace(0, 1, len(y_vals)))

    plt.figure()
    for c, i in zip(colors, y_vals):
        indices = [idx for idx, value in enumerate(y) if value == i]

        plt.scatter(x_r[indices, 0], x_r[indices, 1], c=c, label=i)
        plt.legend()

    plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'TSNE'], default='TRAIN')

    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing face patches.', default=DEFAULT_DATA_DIR)

    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=10)

    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for test', default=20)

    parser.add_argument('--classifier_filename', type=str,
                        help='Classifier model file name as a pickle (.pkl) file', default=DEFAULT_MODEL_PATH)

    return parser.parse_args(argv)


def main(args):
    if args.mode == 'TRAIN':
        train_svm(args.data_dir, args.min_nrof_images_per_class
                  , args.nrof_train_images_per_class, args.classifier_filename)
    elif args.mode == 'TSNE':
        show_tsne(args.data_dir)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
