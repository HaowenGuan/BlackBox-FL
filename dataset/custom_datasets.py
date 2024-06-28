import torch.utils.data as data
from PIL import Image
import numpy as np

class CustomDataset(data.Dataset):
    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data = np.array(data)
        self.target = np.array(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)