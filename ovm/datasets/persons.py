from __future__ import print_function
import os, os.path, errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from dlcommon.datasets.dataset import Dataset
import cv2

class Persons(Dataset):
    base_folder = 'Anno'
    train_list = [
        ['5w+10w+non_person/train.txt'],
    ]

    test_list = [
        ['5w+10w+non_person/filtered-10000_out.txt'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.input_names = ['data']
            self.label_names = ['softmax_label']

            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file_ = os.path.join(root, self.base_folder, f)
                fd = open(file_, 'r')
                for line in fd:
                    image_path, label_str = line.strip().split(' ')
                    self.train_data.append(image_path)
                    self.train_labels.append(int(label_str))
                fd.close()
        else:
            self.input_names = ['data']
            self.label_names = ['softmax_label']

            self.test_data = []
            self.test_labels = []
            f = self.test_list[0][0]
            file_ = os.path.join(root, self.base_folder, f)
            fd = open(file_, 'r')
            for line in fd:
                image_path, label_str = line.strip().split(' ')
                self.test_data.append(image_path)
                self.test_labels.append(int(label_str))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        im = cv2.imread(img)[:,:,(2,1,0)]

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

