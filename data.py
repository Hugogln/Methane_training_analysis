import torch
import numpy  as np
import os
import rasterio as rio
from torchvision import transforms


class PlumeSegmentationDataset():
    """SmokePlumeSegmentation dataset class."""

    def __init__(self, datadir=None, segdir=None, band=[1,2,3,4,5,6,7,8,9,10,11,12,13], transform=None):
        
        self.datadir = datadir
        self.transform = transform
        self.band = band

        # list of image files, labels (positive or negative), segmentation
        self.imgfiles = []
        self.segfiles = []

        
        idx=0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
              if not filename.endswith('.tif'):
                print("Not ending in .tif")
                continue
              self.imgfiles.append(os.path.join(root, filename))
              segfilename = filename.replace(".tif", ".csv")
              self.segfiles.append(os.path.join(segdir, segfilename))
              idx+=1


        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.segfiles = np.array(self.segfiles)


    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)


    def __getitem__(self, idx):
        """Read in image data, preprocess, build segmentation mask, and apply
        transformations."""

        # read in image data
        imgfile = rio.open(self.imgfiles[idx], nodata = 0)
        imgdata = np.array([imgfile.read(i) for i in self.band])
        # skip band 11 (Sentinel-2 Band 10, Cirrus) as it does not contain
        # useful information in the case of Level-2A data products

        fptdata = np.loadtxt(self.segfiles[idx], delimiter=",", dtype=float)
        fptdata = np.array(fptdata)
        # fptdata.reshape(fptdata,(120,120))

        sample = {'idx': idx,
                  'band' : self.band,
                  'img': imgdata,
                  'fpt': fptdata,
                  'imgfile': self.imgfiles[idx]}

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    """Randomly crop 90x90 pixel image (from 120x120)."""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: randomized sample
        """
        imgdata = sample['img']

        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'band' : sample['band'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'fpt': sample['fpt'].copy()[y:y+90, x:x+90],
                'imgfile': sample['imgfile']}

class Crop(object):
    """Crop 90x90 pixel image (from 120x120). - TEST FUNCTION"""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: randomized sample
        """
        imgdata = sample['img']

        x, y = 0, 0

        return {'idx': sample['idx'],
                'band' : sample['band'],
                'img': imgdata.copy()[:, 0:90, 0:90],
                'fpt': sample['fpt'].copy()[0:90, 0:90],
                'imgfile': sample['imgfile']}

class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """
        imgdata = sample['img']
        fptdata = sample['fpt']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1,2))
        fptdata = np.rot90(fptdata, rot, axes=(0,1))

        return {'idx': sample['idx'],
                'band' : sample['band'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}

class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self):
        """
        :param size: edge length of quadratic output size
        """
        self.channel_means = np.array([1909.3802, 1900.5879, 2261.5823, 3164.3564, 3298.6106, 3527.9346, 3791.7458, 3604.5210, 3946.0535, 1223.0176, 27.1881, 4699.9775, 3989.9626])
        self.channel_stds = np.array([ 498.8658,  507.0728,  573.1718,  965.0130, 1014.2232, 1069.5269, 1133.6522, 1073.3431, 1146.3250,  520.9219,   28.9335, 1360.9994, 1169.5753])
    
    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        sample['img'] = (sample['img']-self.channel_means[np.array(sample['band'])-1].reshape(
            sample['img'].shape[0], 1, 1))/self.channel_stds[np.array(sample['band'])-1].reshape(
            sample['img'].shape[0], 1, 1)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'band' : sample['band'],
               'img': torch.from_numpy(sample['img'].copy()),
               'fpt': torch.from_numpy(sample['fpt'].copy()),
               'imgfile': sample['imgfile']}

        return out

def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            Randomize(),
            RandomCrop(),
            ToTensor()
           ])
    else:
        data_transforms = transforms.Compose([
            Normalize(),
            Crop(),
            ToTensor()
           ])

    data = PlumeSegmentationDataset(*args, **kwargs,
                                         transform=data_transforms)
    

    return data
