""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class FineGrainedDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels
    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, root, phase,resize,withSeg,sqResizing,\
                        cropRatio,brightness,saturation,withSaliency=False,\
                        randomSalCrop=False,apply_random_crop=True):

        self.image_path = {}
        self.withSeg = withSeg
        if self.withSeg:
            self.imageSeg_path = {}
        else:
            self.imageSeg_path = None
        self.image_label = {}
        self.root = "../data/"+root
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 200

        self.withSaliency = withSaliency
        self.randomSalCrop = randomSalCrop
        self.apply_random_crop = apply_random_crop

        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        instances = []
        id = 0
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        self.image_path[id] = path
                        self.image_label[id] = class_index
                        self.image_id.append(id)

                        if self.withSeg:
                            directory_seg = self.root.replace("train","seg").replace("test","seg")
                            segPath = "../data/{}/{}/{}".format(directory_seg,target_class,fname.replace(".jpg",".png"))
                            self.imageSeg_path[id] = segPath

                        id += 1

        # transform
        self.transform = get_transform(self.resize, self.phase,colorDataset=self.root.find("emb") == -1,\
                                        sqResizing=sqResizing,cropRatio=cropRatio,brightness=brightness,\
                                        saturation=saturation,salCrop=self.withSaliency)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]
        image = Image.open(self.image_path[image_id]).convert('RGB')  # (C, H, W)

        if not self.withSaliency:
            # image
            image = self.transform(image)

            if self.withSeg:
                imageSeg = Image.open(self.image_path[image_id]).convert('RGB')
                for t in self.transform.transforms:
                    if (not type(t) is transforms.Normalize) and (not type(t) is transforms.ColorJitter):
                        imageSeg = t(imageSeg)

                return image, self.image_label[image_id],imageSeg
            else:
                return image, self.image_label[image_id]
        else:

            imageSalPath = self.image_path[image_id].replace(self.root,self.root+"_sal").replace(".jpg",".png")
            imageSal = Image.open(imageSalPath).convert('RGB')

            for t in self.transform.transforms:

                if type(t) is transforms.Resize:
                    image = t(image)
                    imageSal = t(imageSal)
                else:
                    if (not type(t) is transforms.RandomCrop):
                        image = t(image)
                    else:

                        if self.randomSalCrop:
                            imageSal = np.array(imageSal).mean(axis=2)
                            imageSal = imageSal/imageSal.sum(axis=(0,1),keepdims=True)

                            center = torch.multinomial(torch.tensor(imageSal.reshape(-1)), 1, replacement=True)
                            x,y = center%imageSal.shape[1],center//imageSal.shape[1]
                        else:

                            imageSal = np.array(imageSal)[:,:,0]
                            imageSal = imageSal/imageSal.sum(axis=(0,1),keepdims=True)

                            x = int((np.arange(imageSal.shape[1])[np.newaxis]*imageSal).sum())
                            y = int((np.arange(imageSal.shape[0])[:,np.newaxis]*imageSal).sum())

                        x1,x2 = (x-imageSal.shape[1]//8),(x+imageSal.shape[1]//8)
                        y1,y2 = (y-imageSal.shape[0]//8),(y+imageSal.shape[0]//8)

                        x1,x2 = np.clip(x1,0,3*imageSal.shape[1]//4),np.clip(x2,imageSal.shape[1]//4,imageSal.shape[1])
                        y1,y2 = np.clip(y1,0,3*imageSal.shape[1]//4),np.clip(y2,imageSal.shape[1]//4,imageSal.shape[1])

                        image = Image.fromarray(np.array(image)[y1:y2,x1:x2]).convert("RGB")

                        if self.apply_random_crop:
                            image = t(image)

            return image, self.image_label[image_id]
    def __len__(self):
        return len(self.image_id)

def is_valid_file(x):
    return has_file_allowed_extension(x, IMG_EXTENSIONS)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def get_transform(resize, phase='train',colorDataset=True,sqResizing=True,\
                    cropRatio=0.875,brightness=0.126,saturation=0.5,salCrop=False):

    if sqResizing:
        kwargs={"size":(int(resize[0] / cropRatio), int(resize[1] / cropRatio))}
    else:
        kwargs={"size":int(resize[0] / cropRatio)}

    if phase == 'train':
        transf = [transforms.Resize(**kwargs),
                    transforms.RandomCrop((resize[0]//4,resize[1]//4) if salCrop else resize),
                    transforms.RandomHorizontalFlip(0.5)]

        if colorDataset:
            transf.extend([transforms.ColorJitter(brightness=brightness, saturation=saturation)])

    else:
        transf = [transforms.Resize(**kwargs),transforms.CenterCrop(resize)]

    transf.extend([transforms.ToTensor()])

    if colorDataset:
        transf.extend([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transf = transforms.Compose(transf)

    return transf
