import os.path

from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        print('-----------  AlignedDataset')
        BaseDataset.__init__(self, opt)
        self.dir_A=f'{opt.dataroot}/{opt.phase}/train_A'
        self.A_paths = sorted(os.listdir(self.dir_A))  # get image paths
        if opt.phase=='val':
            self.A_paths=self.A_paths[:1000]

        assert (self.opt.load_size >= self.opt.crop_size
                )  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.cache = {}

    def __getitem__(self, index):
        A_path = f'{self.dir_A}/{self.A_paths[index]}'
        B_path = f'{self.dir_A}/{self.A_paths[index]}'.replace('_A','_B')
        mask_path = f'{self.dir_A}/../mask/{self.A_paths[index]}.png' #ToDo add opt.mask

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        # mask = Image.fromarray(np.array(Image.open(mask_path))>0)
        mask = Image.open(mask_path)
        # A = mask*A
        # B = mask*B
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt,
                                    transform_params,
                                    method=Image.BILINEAR,
                                    grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt,
                                    transform_params,
                                    method=Image.BILINEAR,
                                    grayscale=(self.output_nc == 1))
        mask_transform = get_transform(self.opt,
                                    transform_params,
                                    normalized = False,
                                    method=Image.BILINEAR)
       

        A = A_transform(A)
        B = B_transform(B)
        mask = mask_transform(mask)>0 #.repeat(3,1,1) 

        # print(A.shape,B.shape,mask.shape)
        # xxx
        # for i in range(3): #toDo maybe do it not so rough and f*ck PIL
        #     # A[i][mask[0]==False]=-1
        #     B[i][mask[0]==False]=0



        return {'A': A, 'B': B, 'mask':mask,'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.max_dataset_size == -1:
            return len(self.A_paths)
        else:
            return self.opt.max_dataset_size
