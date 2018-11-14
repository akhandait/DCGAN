from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class CelebA(Dataset):

    def __init__(self, imgDirectory):

        self.imgPaths = []
        for img in os.listdir(imgDirectory):
            self.imgPaths.append(imgDirectory + img)

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.imgPaths[index]))
        # Return in the form (C, H, W) and normalized.
        return np.transpose(img, (2, 0, 1)) / 255 * 2 - 1
