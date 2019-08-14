import h5py
import torch
import torch.utils.data as data
import glob
import os
from common import *
import numpy as np
#import nibabel as nib
#BCHW order
class H5Dataset(data.Dataset):

    def __init__(self, root_path, crop_size=crop_size, mode='train'):
        self.hdf5_list = [x for x in glob.glob(os.path.join(root_path, '*.h5'))]
        self.crop_size = crop_size
        self.mode = mode
        if (self.mode == 'train'):
            self.hdf5_list =self.hdf5_list + self.hdf5_list + self.hdf5_list + self.hdf5_list


    def __getitem__(self, index):
        h5_file = h5py.File(self.hdf5_list[index])
        self.data = h5_file.get('data')
        self.label = h5_file.get('label')      
        self.label=self.label[:,0,...]        
        _, _, C, H, W = self.data.shape
        if (self.mode=='train'):
            cx = random.randint(0, C - self.crop_size[0])
            cy = random.randint(0, H - self.crop_size[1])
            cz = random.randint(0, W - self.crop_size[2])

        elif (self.mode == 'val'):
            # -------Center crop----------
            cx = (C - self.crop_size[0])//2
            cy = (H - self.crop_size[1])//2
            cz = (W - self.crop_size[2])//2

        self.data_crop  = self.data [:, :, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], cz: cz + self.crop_size[2]]
        self.label_crop = self.label[:,  cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], cz: cz + self.crop_size[2]]
        #print(self.data_crop.shape, self.label_crop.shape)
        # ------End random crop-------------
        #h5_file.close()
        return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                torch.from_numpy(self.label_crop[0,:,:,:]).long())

    def __len__(self):
        return len(self.hdf5_list)
