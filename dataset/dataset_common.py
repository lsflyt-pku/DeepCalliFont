import torch
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import os
import numpy as np
from utils.data_utils import *


class FontDataset(Dataset):
    def __init__(self, path):
        super(FontDataset, self).__init__()

        # read the list of data (image and sequence)
        f = open(path, 'r')
        self.data_list = f.readlines()
        f.close()
        self.data_list = [x.strip().split(' ') for x in self.data_list]
        self.len = len(self.data_list)
        print(self.len)

        # read mean skeletons
        # convert relative coordinates to absolute coordinates
        seq_source_file = np.load('data/seq/seq_mean.npz')
        self.seq_source_file = {}
        for key in seq_source_file.keys():
            seq_source = seq_source_file[key]
            l = len(seq_source)
            for i in range(1, l):
                seq_source[i, :2] += seq_source[i-1, :2]
            seq_source = format3_zero_pad_to_max_len_transformer([seq_source], 110)
            self.seq_source_file[key] = seq_source

        self.ref_num = 3

        # read reference character list
        # the file of data list is the same as the train list
        f = open(path, 'r')
        tmp_data_list = f.readlines()
        f.close()
        tmp_data_list = [x.strip().split(' ') for x in tmp_data_list]
        self.content_list = {}
        for font, char in tmp_data_list:
            if int(font) not in self.content_list.keys():
                self.content_list[int(font)] = []
            self.content_list[int(font)].append(int(char))

    def __getitem__(self, index):
        font, char = self.data_list[index]
        font = int(font)
        char = int(char)

        # get reference character id
        ref_char = np.random.randint(0, len(self.content_list[font]), self.ref_num)
        ref_char = [self.content_list[font][x] for x in ref_char]
        while char in ref_char: 
            ref_char = np.random.randint(0, len(self.content_list[font]), self.ref_num)
            ref_char = [self.content_list[font][x] for x in ref_char]

        # read source, target, reference images
        image_source = cv.imread('data/image/source/0_{}.png'.format(char))
        image_source = cv.resize(image_source, (128, 128))
        _, image_source = cv.threshold(image_source, 127, 255, cv.THRESH_BINARY)
        image_source = 1. - torch.tensor(image_source).float().permute(2, 0, 1) / 255.

        image_target = cv.imread('data/image/common/{}_{}.png'.format(font, char))
        image_target = cv.resize(image_target, (128, 128))
        _, image_target = cv.threshold(image_target, 127, 255, cv.THRESH_BINARY)
        image_target = 1. - torch.tensor(image_target).float().permute(2, 0, 1) / 255.

        image_ref = torch.ones((self.ref_num, 3, 128, 128)).float()
        for i in range(self.ref_num):
            temp = cv.imread('data/image/common/{}_{}.png'.format(font, ref_char[i]))
            temp = cv.resize(temp, (128, 128))
            _, temp = cv.threshold(temp, 127, 255, cv.THRESH_BINARY)
            image_ref[i] = 1. - torch.tensor(temp).float().permute(2, 0, 1) / 255.
       
        # read source, target, reference sequences
        seq_source = self.seq_source_file['GB'+str(char)]
        seq_source = torch.tensor(seq_source).float().squeeze()
        seq_source[:, :2] = seq_source[:, :2] / 200.

        seq_target = np.load('data/seq/common/{}_{}.npy'.format(font, char))

        l = len(seq_target)
        seq_target = torch.tensor(seq_target).float()
        for i in range(l):
            if seq_target[i, -2] == 0:
                seq_target[i, :2] = seq_target[i, :2] * 2 - 1
        seq_target[0, :2] = 0

        ref = []
        for i in range(self.ref_num):
            seq = np.load('data/seq/common/{}_{}.npy'.format(font, ref_char[i]))
            l = len(seq)
            for i in range(l):
                if seq[i, -2] == 0:
                    seq[i, :2] = seq[i, :2] * 2 - 1
            seq[0, :2] = 0
            ref.append(seq[None, :, :])
        seq_ref = np.concatenate(ref, 0)
        seq_ref[:, :, 2] = seq_ref[: ,:, 3] + seq_ref[:, :, 5]
        seq_ref = seq_ref[:, :, :3]
        seq_ref = torch.tensor(seq_ref).float()

        # in our dataset, the id of font and character is start at 1
        return image_source, image_target, image_ref, seq_source, seq_target, seq_ref, font-1, char-1

    def __len__(self):
        return self.len
