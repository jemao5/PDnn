from torch.utils import data
import numpy as np


class DataHandler(data.Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        self.get_random_slice()
        label = np.squeeze(self.labels[self.slice])
        image = np.squeeze(self.images[self.slice, :, :], axis=0)
        return label, image

    def get_random_slice(self):
        self.slice = np.random.randint(0, self.labels.shape, 1)