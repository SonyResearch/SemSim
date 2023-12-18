import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RE_ID(Dataset):
    def __init__(self, data_folder, transform=None, target_transform=None):
        self.root_dir = data_folder
        self.image_files = os.listdir(os.path.join(self.root_dir, "img"))
        self.remove()
        self.transformed_dir =os.path.join(self.root_dir, "img0")
        self.distorted_dir = os.path.join(self.root_dir, "img1")
        self.transform = transform
        self.target_transform = target_transform

    def remove(self):
        for fichier in self.image_files:  # filelist[:] makes a copy of filelist.
            if not (fichier.endswith(".png")):
                self.image_files.remove(fichier)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # load original image
        image_name = self.image_files[index]
        image_path = os.path.join(self.root_dir, "img", image_name)
        image = Image.open(image_path).convert('RGB')

        # load transformed image for positive sample
        pos_image_names = os.listdir(os.path.join(self.transformed_dir, image_name.split('_')[0]))
        pos_image_name =  random.choice(pos_image_names)
        pos_image_path = os.path.join(self.root_dir, "img0",image_name.split('_')[0], pos_image_name)
        pos_image = Image.open(pos_image_path).convert('RGB')

        # randomly select a distorted image for negative sample
        if os.path.exists(os.path.join(self.distorted_dir, '-' + image_name.split('_')[0])):
            distorted_image_files = os.listdir(os.path.join(self.distorted_dir, '-' + image_name.split('_')[0]))
            neg_image_name = random.choice(distorted_image_files)
            neg_image_path = os.path.join(self.root_dir, "img1",'-' + image_name.split('_')[0], neg_image_name)
            neg_image = Image.open(neg_image_path).convert('RGB')
        else:
            other_image_files = os.listdir(os.path.join(self.root_dir, 'others'))
            neg_image_name = random.choice(other_image_files)
            neg_image_path = os.path.join(self.root_dir, 'others', neg_image_name)
            neg_image = Image.open(neg_image_path).convert('RGB')


        if self.transform:
            image = self.transform(image)
            pos_image =  self.transform(pos_image)
            neg_image =  self.transform(neg_image)

        # return the images and their labels
        return (image, pos_image, neg_image), torch.tensor(index)



if __name__=='__main__':

    recset = RE_ID('/home/sunxx/data/human_anno_id', transform=transforms.ToTensor())
    recset[0]
