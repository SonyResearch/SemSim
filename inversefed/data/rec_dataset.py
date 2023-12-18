import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

class RecDataset(Dataset):
    def __init__(self, data_folder, transform=None, target_transform=None):
        self.data_folder = data_folder
        self.images = []
        self.images_paths = []
        self.labels = []
        self.filenames = []
        self.transform = transform
        self.target_transform = target_transform

        for filename in os.listdir(data_folder):
            if filename.endswith(".png"):
                image = Image.open(os.path.join(data_folder, filename))
                image = np.array(image)
                self.images.append(image)
                label = int(filename.split("_")[1])
                self.labels.append(label)
                images_path =os.path.join(self.data_folder, filename)
                self.images_paths.append(images_path)

                self.filenames.append(filename)

        self.images = np.stack(self.images, axis=0)
        self.labels = np.array(self.labels)

    def get_filename(self):
        return self.filenames

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # image1 = self.images[idx]
        # image1 = self.transform(image1)
        label = self.labels[idx]
        image_path = self.images_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label




if __name__=='__main__':
    recset = RecDataset('/home/sunxx/project/ATSPrivacy/benchmark/images/test/rec', transform=transforms.ToTensor())
    recset2 = RecDataset('/home/sunxx/project/ATSPrivacy/benchmark/images/test/ori', transform=transforms.ToTensor())
    recset[0]
    recset2[0]

    #
    # recset = CustomDataset('/home/sony/data/human_anno_id/train_with_ori', transform=transforms.ToTensor())
    # recset[0]