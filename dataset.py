from torch.utils.data import Dataset
import torch
import numpy as np
import imageio.v3
import os
import torch.nn.functional as F
import cv2
from utils import augment_hsv


def img_normalize(image):
    if len(image.shape) == 2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel, channel, channel], axis=2)
    else:
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))) \
                / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image


class TrainDataset(Dataset):
    def __init__(self, paths):
        self.image = []
        self.label = []
        self.count = {}
        for path in paths:
            self.list = os.listdir(os.path.join(path, "Imgs"))
            for i in self.list:
                self.image.append(os.path.join(path, "Imgs", i))
                self.label.append(os.path.join(path, "GT", os.path.splitext(i)[0] + ".png"))
        print("Datasetsize:", len(self.image))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        hsv = True
        img = imageio.v3.imread(self.image[item]).astype(np.float32)/255.
        label = imageio.v3.imread(self.label[item]).astype(np.float32)/255.
        # if hsv:
        #     augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)
        # img = img/255.

        # img = img2ycbcr(img) / 255.
        ration = np.random.rand()
        if ration < 0.25:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
        elif ration < 0.5:
            img = cv2.flip(img, 0)
            label = cv2.flip(label, 0)
        elif ration < 0.75:
            img = cv2.flip(img, -1)
            label = cv2.flip(label, -1)

        if len(label.shape) == 3:
            label = label[:, :, 0]
        label = label[:, :, np.newaxis]

        return {"img": torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0),
                "label": torch.from_numpy(label).permute(2,0,1).unsqueeze(0)}
        #
        # img = DCT_embedding(img, N=8, title=0)
        # label = DCT_embedding(label, N=8, title=1)
        # return {"img": torch.from_numpy(img).unsqueeze(0),
        #         "label": torch.from_numpy(label).permute(2, 0, 1).unsqueeze(0)}


class TestDataset(Dataset):
    def __init__(self, path, size):
        self.size = size
        self.image = []
        self.label = []
        self.list = os.listdir(os.path.join(path, "Imgs"))
        self.count = {}
        for i in self.list:
            self.image.append(os.path.join(path, "Imgs", i))
            self.label.append(os.path.join(path, "GT", os.path.splitext(i)[0] + ".png"))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        img = imageio.v3.imread(self.image[item]).astype(np.float32) / 255.  # ndarray
        # print("image load")
        label = imageio.v3.imread(self.label[item]).astype(np.float32) / 255.
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]

        return {"img": F.interpolate(torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0),
                 (self.size, self.size), mode='bilinear', align_corners=True).squeeze(0),
                "label": torch.from_numpy(label).permute(2,0,1),
                'name': self.label[item]}

        # img = DCT_embedding(img, N=8, title=0)
        # label = DCT_embedding(label, N=8, title=1)
        # return {"img": torch.from_numpy(img),
        #         "label": torch.from_numpy(label).permute(2, 0, 1),
        #         'name': self.label[item]}



def my_collate_fn(batch):
    size = 384
    imgs = []
    labels = []
    for item in batch:
        imgs.append(F.interpolate(item['img'], (size, size), mode='bilinear'))  # 缩放
        labels.append(F.interpolate(item['label'], (size, size), mode='bilinear'))
        # imgs.append(item['img'])
        # labels.append(item['label'])
    return {'img': torch.cat(imgs, 0),
            'label': torch.cat(labels, 0)}


def DCT_embedding(image, N=8, title=0):
    """
    image:input RGB image [420,632,3]
    return: normalized dct;ndarray [3,420,632]
    title: 0 denotes img; 1 denotes GT
    """
    size = 384
    if title == 0:
        _, _, c = image.shape
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        dct_list = []
        for index in range(c):
            img_perchannel = np.float32(ycbcr[:, :, index])
            img_perchannel = cv2.resize(img_perchannel, (size,size), interpolation=cv2.INTER_LINEAR)
            dct_coeff = np.zeros((size, size), dtype=np.float32)
            # 8x8 DCT
            for row in np.arange(0, size, N):
                for col in np.arange(0, size, N):
                    block = np.array(img_perchannel[row:(row + N), col:(col + N)], dtype=np.float32)
                    dct_coeff[row:(row + N), col:(col + N)] = cv2.dct(block)
            dct_list.append(dct_coeff)
        dct = np.stack(dct_list, axis=0)

    else:
        img_perchannel = np.float32(image[:, :, 0])
        img_perchannel = cv2.resize(img_perchannel, (size,size), interpolation=cv2.INTER_LINEAR)
        dct_coeff = np.zeros((size, size), dtype=np.float32)
        for row in np.arange(0, size, N):
            for col in np.arange(0, size, N):
                block = np.array(img_perchannel[row:(row + N), col:(col + N)], dtype=np.float32)
                dct_coeff[row:(row + N), col:(col + N)] = cv2.dct(block)
        if len(dct_coeff.shape) == 2:
            dct = dct_coeff[:, :, np.newaxis]
    return dct


def img2ycbcr(image):
    """
    image:input RGB image [420,632,3]
    return: ycbcr [420,632,3]
    """
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return ycbcr

