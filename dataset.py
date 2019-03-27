import os
from PIL import Image
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from bg import Bwdist

class ImageData(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """

    def __init__(self, img_root, label_root, transform=None, t_transform=None, filename=None, require_name=False, test=False):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            self.label_path = list(
                map(lambda x: os.path.join(img_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        else:
            lines = [line.rstrip('\n') for line in open(filename)]
            if test:
                self.image_path = list(map(lambda x: os.path.join(img_root, x.split(' ')[0][:-3]+'jpg'), lines))
                self.label_path = list(map(lambda x: os.path.join(label_root, x.split(' ')[0][:-3]+'png'), lines))
            else:
                self.image_path = list(map(lambda x: os.path.join(img_root, x.split(' ')[0][:-3]+'jpg'), lines))
                self.label_path = list(map(lambda x: os.path.join(label_root, x.split(' ')[1][:-3]+'png'), lines))
        self.require_name = require_name
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        label = Image.open(self.label_path[item]).convert('L')
        bg, fg = Bwdist(label)
        
        # img_label = {'image': image, 'label': label}
        if self.transform is not None:
            # img_label = self.transform(img_label)
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
            bg = self.t_transform(bg)
            fg = self.t_transform(fg)
        if self.require_name:
            return image, label, bg, fg, self.image_path[item].split('/')[-1]
        else:
            return image, label, bg, fg

    def __len__(self):
        return 10
        return len(self.image_path)

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}
                
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, filename=None, num_thread=4, pin=True, test=False):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])
    t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
    dataset = ImageData(img_root, label_root, transform=transform, t_transform=t_transform, filename=filename, test=test)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                    pin_memory=pin)
    return data_loader



if __name__ == '__main__':
    import numpy as np
    img_root = '/home/hanqi/dataset/DUTS/DUTS-TR/'
    filename = '/home/hanqi/dataset/DUTS/DUTS-TR/train_pair.lst'
    # # loader = get_loader(img_root, 1, filename=filename)
    # for image, label in loader:
    #     print(np.array(image).shape)
    #     break
