import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import torchvision.transforms as transforms

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
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        if self.opt.mask:
            self.dir_mask = os.path.join(opt.dataroot, 'masks')
            self.mask_paths = sorted(make_dataset(self.dir_mask, opt.max_dataset_size)) 
            assert(len(self.mask_paths) == len(self.AB_paths)) # same number of images and masks
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        
        if self.opt.nir2cfp:
            mask = self.joint_mask(A, B)
        if self.opt.mask:
            if self.opt.nir2cfp:
                registration_mask = Image.open(self.mask_paths[index]).convert('1')
                mask = ImageChops.logical_and(mask, registration_mask)
            else:
                mask = Image.open(self.mask_paths[index]).convert('1')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        
        selective_flag = self.opt.nir2cfp
        # selective_flag = False # use if want to use the original dataset implementation
        if selective_flag:
            center_params = transform_params['center']
            edge_params = transform_params['edge']
            
            A_center_transform = get_transform(
                self.opt, center_params, grayscale=(self.input_nc == 1))
            A_edge_transform = get_transform(
                self.opt, edge_params, grayscale=(self.input_nc == 1))
            B_center_transform = get_transform(
                self.opt, center_params, grayscale=(self.output_nc == 1))
            B_edge_transform = get_transform(
                self.opt, edge_params, grayscale=(self.output_nc == 1))
            
            A_center = A_center_transform(A)
            A_edge = A_edge_transform(A)
            B_center = B_center_transform(B)
            B_edge = B_edge_transform(B)
            
            output = {'A_center': A_center, 'A_edge': A_edge, 'A_paths': AB_path,
                      'B_center': B_center, 'B_edge': B_edge, 'B`_paths': AB_path}
            
            # mask addition
            mask_center_transform = get_transform(
                self.opt, center_params, convert=False)
            mask_edge_transform = get_transform(
                self.opt, edge_params, convert=False)
            
            mask_center = transforms.functional.pil_to_tensor(
                mask_center_transform(mask))
            mask_edge = transforms.functional.pil_to_tensor(
                mask_edge_transform(mask))
            
            output['mask_center'] = mask_center
            output['mask_edge'] = mask_edge
            
        else:
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

            A = A_transform(A)
            B = B_transform(B)
            
            output = {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        
            if self.opt.nir2cfp or self.opt.mask:
                mask_transform = get_transform(self.opt, transform_params, convert=False)
                mask = transforms.functional.pil_to_tensor(mask_transform(mask))
                output['mask'] = mask
        return output

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
    
    @staticmethod
    def joint_mask(im1, im2, threshold=10):
        im1_gray = im1.convert('L').point(lambda x: 0 if x <= threshold else 255, '1')
        im2_gray = im2.convert('L').point(lambda x: 0 if x <= threshold else 255, '1')
        return ImageChops.logical_and(im1_gray, im2_gray)
