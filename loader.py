import cv2
import random
import os
import numpy as np
import glob
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from utils import create_character_mask, create_affinity_mask


class SynthText(Dataset):
    def __init__(self, data_folder, save_img, save_char, save_aff):
        self.label_file = glob.glob(os.path.join(data_folder, '*.mat'))
        mat = scipy.io.loadmat(self.label_file[0])

        self.images =  []
        self.char_masks = []
        self.aff_masks = []
        charBB = mat['charBB']
        string = mat['txt']
        imnames = mat['imnames']

        for i in range(100):
            # Load image
            self.images.append(os.path.join(save_img, str(i) + '.png'))
            img = cv2.imread(os.path.join(data_folder, imnames[0, i][0]))
            cv2.imwrite(os.path.join(save_img, str(i) + '.png'), img)
            # Gauss image
            gauss = cv2.imread('gauss.jpg')

            # Character mask
            self.char_masks.append(os.path.join(save_char, str(i) + '.png'))
            char_mask = create_character_mask(img, gauss, charBB[0, i])
            cv2.imwrite(os.path.join(save_char, str(i) + '.png'), char_mask)

            # Aff mask
            self.aff_masks.append(os.path.join(save_aff, str(i) + '.png'))
            aff_mask = create_affinity_mask(img, gauss, charBB[0, i], string[0, i])
            cv2.imwrite(os.path.join(save_aff, str(i) + '.png'), aff_mask)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        h, w, _ = np.shape(image)

        if h % 2 == 1:
            h = h - 1
        
        if w  % 2 == 1:
            w = w - 1

        image = cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        char_mask = cv2.imread(self.char_masks[idx])
        char_mask = cv2.resize(char_mask, (int(w / 2), int(h /2)))
        char_mask = cv2.cvtColor(char_mask, cv2.COLOR_BGR2GRAY)
        char_mask = np.expand_dims(char_mask, axis=2)

        aff_mask = cv2.imread(self.aff_masks[idx])
        aff_mask = cv2.resize(aff_mask, (int(w / 2), int(h / 2)))
        aff_mask = cv2.cvtColor(aff_mask, cv2.COLOR_BGR2GRAY)
        aff_mask = np.expand_dims(aff_mask, axis=2)

        # Convert to tensor
        tranforms = ToTensor()
        image = tranforms(image)
        char_mask = tranforms(char_mask)
        aff_mask = tranforms(aff_mask)

        # image = image.permute(2, 0, 1)
        # char_mask = char_mask.permute(2, 0, 1)
        # aff_mask = aff_mask.permute(2, 0, 1)
        return image, char_mask, aff_mask


class SynthTextLoader(object):
    def __init__(self, batch_size, shuffle, data_folder, save_img, save_char, save_aff):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = SynthText(data_folder, save_img, save_char, save_aff)

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)



if __name__ == '__main__':
    save_img = '/mnt/data/hades/source/sekiwa_rnd/dataset/synthtext/images'
    save_char = '/mnt/data/hades/source/sekiwa_rnd/dataset/synthtext/char_mask'
    save_aff = '/mnt/data/hades/source/sekiwa_rnd/dataset/synthtext/aff_mask'

    data_loader = SynthTextLoader(1, True, '/mnt/data/hades/source/ocr_rnd/datasets/SynthText', save_img, save_char, save_aff)
    data_loader = data_loader.loader()