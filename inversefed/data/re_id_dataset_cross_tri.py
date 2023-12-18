import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class RE_ID_Tri_All_Data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                    sample_path = os.path.join(cls_dir, file_name)
                    self.samples.append((sample_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, cls_idx = self.samples[idx]
        img = Image.open(sample_path).convert('RGB')


        class_dir = os.path.dirname(sample_path)
        class_name = os.path.basename(os.path.dirname(sample_path)) # '/home/sony/data/human_anno_id/train_with_ori/-100'

        class_img_names = os.listdir(class_dir)
        # class_img_names = [f for f in class_img_names if f != img_name]
        class_img_name = random.choice(class_img_names)
        positive_img_path = os.path.join(class_dir, class_img_name)
        positive_img = Image.open(positive_img_path).convert('RGB')

        if class_name.startswith('-'):
            negative_class_name= class_name.strip('-')
            negative_class_dir = os.path.join(self.root_dir, negative_class_name)
            negative_images = os.listdir(negative_class_dir)

            ori_img_name = [img for img in negative_images if img.endswith('_ori.png')]
            ori_img_path = os.path.join(negative_class_dir, ori_img_name[0])
            ori_img = Image.open(ori_img_path).convert('RGB')

            negative_images = [f for f in negative_images if f != ori_img_name]
            negative_img_name = random.choice(negative_images)
            negative_img_path = os.path.join(negative_class_dir, negative_img_name)
            negative_img = Image.open(negative_img_path).convert('RGB')
            neg_pos = -1
            # positive
        else:
            ori_img_name = [img for img in class_img_names if img.endswith('_ori.png')]
            ori_img_path = os.path.join(class_dir, ori_img_name[0])
            ori_img = Image.open(ori_img_path).convert('RGB')

            negative_class_name = '-' + class_name
            negative_class_dir = os.path.join(self.root_dir, negative_class_name)
            if os.path.exists(negative_class_dir):
                negative_images = os.listdir(negative_class_dir)
                negative_img_name = random.choice(negative_images)
                negative_img_path = os.path.join(negative_class_dir, negative_img_name)
            else:
                other_image_files = os.listdir(os.path.join(os.path.dirname(self.root_dir), 'others'))
                negative_img_name = random.choice(other_image_files)
                negative_img_path = os.path.join(os.path.dirname(self.root_dir), 'others', negative_img_name)
            negative_img = Image.open(negative_img_path).convert('RGB')
            neg_pos = 1

        if self.transform is not None:
            img = self.transform(img)
            ori_img = self.transform(ori_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return (img, ori_img, positive_img, negative_img), torch.tensor([cls_idx, neg_pos])


if __name__=='__main__':
    recset = RE_ID_Tri_All_Data('/home/sony/data/human_anno_id/train_with_ori', transform=transforms.ToTensor())
    recset[0]
    recset[5000]
    #
    # recset = CustomDataset('/home/sony/data/human_anno_id/train_with_ori', transform=transforms.ToTensor())
    # recset[0]