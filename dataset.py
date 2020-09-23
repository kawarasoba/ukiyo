import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter 
# augmentation
from albumentations.augmentations.transforms import Resize,Normalize
from albumentations import Compose

import augmentations as aug
import constants as cons

minimal_class_num = 24

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class UkiyoeTrainDataset(Dataset):
    """Prediction Author of Ukiyoe Dataset.
    
    Args:
        train_images_path (str) - path of train image dataset
        train_labels_path (str) - path of train label dataset
        check_all (bool) - If True, checks all dataset
        valid (bool) - If True, provide dataset for validation
        nfold (int) - fold number in the range of 0 ~ 4 (for cross validation)
        confidence_boader - boader line for evaluate accuracy
        result_path - path of pretrained model's param
        over_sampling(bool) - If True, applies over sampling to train dataset
        seed (int) - random seed number. Default:98
        transform_aug - transforms composed of albumentations for augmentation
        mixup (bool) - If True, applies mixup to train dataset
        alpha (int) - parameter of Beta distribution. Default:1
        width (int) - Default:3
        transform - transforms composed of albumentations for output
        as_numpy (book) - If True, outputs images as numpy.ndarray
    """
    def __init__(self,train_images_path,train_labels_path,check_all=False,valid=False,
    nfold=0,confidence_boader=None,result_path='',test_images_path='',over_sampling=False,seed=98,transform_aug=None,
    mixup=False,alpha=1,augmix=False,width=3,transform=None,as_numpy=False):

        imgs = np.load('{}/ukiyoe-train-imgs.npz'.format(train_images_path))['arr_0']
        labels_value = np.load('{}/ukiyoe-train-labels.npz'.format(train_labels_path))['arr_0']
        labels = np.identity(10)[labels_value]
        fold_len = int(len(imgs)/5)
        np.random.seed(seed=seed)
        p = np.random.permutation(len(imgs))
        imgs = imgs[p]
        labels = labels[p]
        
        if check_all:
            self.data_list = imgs
            self.label_list = labels
        else:
            if valid:
                self.data_list = imgs[fold_len*(nfold) : fold_len*(nfold+1)]
                self.label_list = labels[fold_len*(nfold) : fold_len*(nfold+1)]
            else:
                self.data_list = np.delete(imgs,slice(fold_len*(nfold), fold_len*(nfold+1)), axis=0)
                self.label_list = np.delete(labels,slice(fold_len*(nfold), fold_len*(nfold+1)), axis=0)
                if confidence_boader is not None:
                    inference = pd.read_csv(result_path).values[:,1:3].T
                    test_imgs = np.load('{}/ukiyoe-test-imgs.npz'.format(test_images_path))['arr_0'][np.where(inference[1] > confidence_boader)[0]]
                    self.data_list = np.append(self.data_list,test_imgs,axis=0)
                    pseudo_labels=np.identity(10)[inference[0,np.where(inference[1] > confidence_boader)[0]].astype(np.int64)]
                    self.label_list = np.append(self.label_list,pseudo_labels,axis=0)
                    pseudo_idx = np.random.permutation(len(self.data_list))
                    self.data_list = self.data_list[pseudo_idx]
                    self.label_list = self.label_list[pseudo_idx]
                if over_sampling:
                    count_labels = np.sum(self.label_list,axis=0).astype(np.int64)
                    majority = np.max(count_labels)
                    sampled_idx = np.array([],dtype=np.int64)
                    for class_num in range(cons.NUM_CLASSES):
                        idx =  np.repeat(
                            np.where(self.label_list[:,class_num])[0],
                            majority // count_labels[class_num])
                        if count_labels[class_num] < majority:
                            idx = np.append(
                                idx,
                                np.random.choice(np.where(
                                    self.label_list[:,class_num])[0],
                                    majority % count_labels[class_num],
                                    replace=False))
                        sampled_idx = np.append(sampled_idx,idx)
                    sampled_idx = np.random.permutation(sampled_idx)
                    self.data_list = self.data_list[sampled_idx]
                    self.label_list = self.label_list[sampled_idx]
                
        self.valid = valid
        self.transform_aug = transform_aug
        self.mixup = mixup        
        self.alpha = alpha
        self.augmix = augmix
        self.width = width
        self.transform = transform
        self.as_numpy = as_numpy

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image = self.data_list[idx]
        label = self.label_list[idx]
        if not self.valid:
            if self.augmix:
                image_mixed = np.zeros_like(image)
                w = np.random.dirichlet([self.alpha]*self.width)
                m = np.random.beta(self.alpha, self.alpha)
                for i in range(self.width):
                    image_aug = image.copy()
                    depth = np.random.randint(1,4)
                    for _ in range(depth):
                        operations = np.random.choice(aug.augmentations)()
                        image_aug = operations(image=image_aug)['image']    
                    image_mixed = image_mixed + image_aug*w[i]
                image = image * m + image_mixed * (1-m)
            if self.mixup:
                if self.transform_aug is not None:
                    mix_idx = np.random.randint(0,len(self.data_list))
                    image = self.transform_aug(image=image)['image']
                    image2 = self.transform_aug(image=self.data_list[mix_idx])['image']
                    label2 = self.label_list[mix_idx]
                    l = np.random.beta(self.alpha, self.alpha)
                    image = image * l + image2 * (1-l)
                    label = label *l + label2 * (1-l)
                else:
                    mix_idx = np.random.randint(0,len(self.data_list))
                    image2 = self.data_list[mix_idx]
                    label2 = self.label_list[mix_idx]
                    l = np.random.beta(self.alpha, self.alpha)
                    image = image * l + image2 * (1-l)
                    label = label *l + label2 * (1-l)
            else:
                if self.transform_aug is not None:
                    image = self.transform_aug(image=image)['image']           
        if self.transform is not None:
            image = self.transform(image=image)['image']
        if not self.as_numpy:
            image = torch.from_numpy(image.transpose(2,0,1))
        return image, label

def load_train_data(batch_size,num_worker=4,shuffle=True,**kwargs):
    '''generatie Loader with Transformation'''

    np.random.seed(100)
    dataset = UkiyoeTrainDataset
    data_loader = DataLoader(
        dataset(**kwargs),batch_size=batch_size,shuffle = shuffle,num_workers=num_worker,worker_init_fn=worker_init_fn)
    return data_loader

class UkiyoeTestDataset(Dataset):
    """Prediction Author of Ukiyoe Dataset
    
    Args:
        data_path (str) - path of test image dataset
        transform - transforms composed of albumentations for output
    """
    def __init__(self, data_path, transform=None):
        self.data_path_imgs = '{}/ukiyoe-test-imgs.npz'.format(data_path)
        self.data_list = np.load(self.data_path_imgs)['arr_0']
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image = self.data_list[idx]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = torch.from_numpy(image.transpose(2,0,1))
        return idx+1, image
    
def load_test_data(batch_size,**kwargs):
    '''generatie Loader with Transformation'''
    dataset = UkiyoeTestDataset
    data_loader = DataLoader(
        dataset(**kwargs),batch_size=batch_size, shuffle = False,num_workers=4)
    return data_loader

def main():

    # augmentation
    transform_aug=Compose([
        aug.HueSaturationValue(),
        aug.RandomBrightnessContrast(),
        aug.CLAHE(),
        aug.JpegCompression(),
        aug.GaussNoise(),
        aug.MedianBlur(),
        aug.ElasticTransform(),
        aug.HorizontalFlip(),
        aug.Rotate(),
        aug.CoarseDropout(),
        aug.RandomSizedCrop()],
        p=1)
    # transform for output
    transform=Compose([
        Resize(cons.IMAGE_SIZE,cons.IMAGE_SIZE),
        Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0)],p=1)
    
    # Dataset
    '''
    dataset = UkiyoeTrainDataset(
        train_images_path='data',
        train_labels_path='data',
        valid=False,
        confidence_boader=0.87,
        result_path='result/model_effi_b3/efficientnet_b3_980/inference_with_c.csv',
        test_images_path='data',
        over_sampling=False,
        transform_aug=None,
        augmix=False,
        mixup=False,
        transform=transform)
    img, label = dataset[0]
    #print(img.shape)
    #plt.imshow(img)
    #plt.show()
    '''
    # train data loader
    loader = load_train_data(
        train_images_path='data',
        train_labels_path='data',
        batch_size=2,
        valid=False,
        nfold=0,
        transform_aug=None,
        augmix=True,
        mixup=False,
        transform=transform,
        as_numpy=True)
    image_batch, label_batch= next(loader.__iter__())
    print(image_batch[0].shape)
    print(label_batch[0].shape)
    '''
    # test data loader
    loader = load_test_data(
        data_path='data',
        batch_size=1)
    id_batch, image_batch = next(loader.__iter__())
    print(id_batch)
    print(image_batch.shape)
    '''

if __name__ == '__main__':
    main()