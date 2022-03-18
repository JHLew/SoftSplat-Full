from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import torch.utils.data as data
import os
from random import randint


class Vimeo90k(data.Dataset):
    def __init__(self, path, is_train=True):
        super(Vimeo90k, self).__init__()
        train_list = os.path.join(path, 'tri_trainlist.txt')
        test_list = os.path.join(path, 'tri_testlist.txt')
        self.frame_list = os.path.join(path, 'sequences')
        self.is_train = is_train
        if self.is_train:
            triplet_list = train_list
        else:
            triplet_list = test_list
        with open(triplet_list) as triplet_list_file:
            triplet_list = triplet_list_file.readlines()
            triplet_list_file.close()
        self.triplet_list = triplet_list[:-1]

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        triplet_path = self.triplet_list[idx]
        if triplet_path[-1:] == '\n':
            triplet_path = triplet_path[:-1]
        try:
            vid_no, seq_no = triplet_path.split('/')
            name = vid_no + '_' + seq_no
        except:
            name = triplet_path
        triplet_path = os.path.join(self.frame_list, triplet_path)
        im1 = os.path.join(triplet_path, 'im1.png')
        im2 = os.path.join(triplet_path, 'im2.png')
        im3 = os.path.join(triplet_path, 'im3.png')

        # read image file
        im1 = Image.open(im1).convert('RGB')
        im2 = Image.open(im2).convert('RGB')
        im3 = Image.open(im3).convert('RGB')

        # data augmentation - random flip / sequence flip
        if self.is_train:
            flip_flag = randint(0, 1)
            if flip_flag == 1:
                im1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
                im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
                im3 = im3.transpose(Image.FLIP_LEFT_RIGHT)

            order_reverse = randint(0, 1)
            if order_reverse == 1:
                tmp = im1
                im1 = im3
                im3 = tmp

        im1 = to_tensor(im1)
        im2 = to_tensor(im2)
        im3 = to_tensor(im3)

        return torch.stack([im1, im3], dim=1), im2, 0.5, name
