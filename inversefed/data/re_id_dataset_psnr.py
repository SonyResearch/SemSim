import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import math
class RE_ID_PSNR(Dataset):
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
    def psnr(self, img1, img2):
        mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, index):
        # load original image
        image_name = self.image_files[index]
        image_path = os.path.join(self.root_dir, "img", image_name)
        image = Image.open(image_path).convert('RGB')

        # load transformed image for positive sample
        pos_image_names = os.listdir(os.path.join(self.transformed_dir, image_name.split('_')[0]))
        pos_image_name1 =  random.choice(pos_image_names)
        pos_image_path1 = os.path.join(self.root_dir, "img0",image_name.split('_')[0], pos_image_name1)
        pos_image1 = Image.open(pos_image_path1).convert('RGB')
        pos_image_name2 =  random.choice(pos_image_names)
        pos_image_path2 = os.path.join(self.root_dir, "img0",image_name.split('_')[0], pos_image_name2)
        pos_image2 = Image.open(pos_image_path2).convert('RGB')


        # randomly select a distorted image for negative sample
        if os.path.exists(os.path.join(self.distorted_dir, '-' + image_name.split('_')[0])):
            distorted_image_files = os.listdir(os.path.join(self.distorted_dir, '-' + image_name.split('_')[0]))
            neg_image_name1 = random.choice(distorted_image_files)
            neg_image_path1 = os.path.join(self.root_dir, "img1",'-' + image_name.split('_')[0], neg_image_name1)
            neg_image1 = Image.open(neg_image_path1).convert('RGB')
            neg_image_name2 = random.choice(distorted_image_files)
            neg_image_path2 = os.path.join(self.root_dir, "img1",'-' + image_name.split('_')[0], neg_image_name2)
            neg_image2 = Image.open(neg_image_path2).convert('RGB')
        else:
            other_image_files = os.listdir(os.path.join(self.root_dir, 'others'))
            neg_image_name1 = random.choice(other_image_files)
            neg_image_path1 = os.path.join(self.root_dir, 'others', neg_image_name1)
            neg_image1 = Image.open(neg_image_path1).convert('RGB')
            neg_image_name2 = random.choice(other_image_files)
            neg_image_path2 = os.path.join(self.root_dir, 'others', neg_image_name2)
            neg_image2 = Image.open(neg_image_path2).convert('RGB')

        psnr_p1 = self.psnr(image, pos_image1)
        psnr_p2 = self.psnr(image, pos_image2)
        psnr_n1 = self.psnr(image, neg_image1)
        psnr_n2 = self.psnr(image, neg_image2)
        if self.transform:
            image = self.transform(image)
            pos_image1 =  self.transform(pos_image1)
            neg_image1 =  self.transform(neg_image1)
            pos_image2 =  self.transform(pos_image2)
            neg_image2 =  self.transform(neg_image2)

        # return the images and their labels
        return (image, pos_image1, neg_image1, pos_image2, neg_image2), torch.tensor([index, psnr_p1, psnr_p2, psnr_n1, psnr_n2])




if __name__=='__main__':

    recset = RE_ID_PSNR('/home/sony/data/human_anno_id', transform=transforms.ToTensor())
    recset[0]
