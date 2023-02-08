import torch
import numpy  as np
import os
import rasterio as rio
from torchvision import transforms


class PlumeSegmentationDataset():
    """SmokePlumeSegmentation dataset class."""

    def __init__(self, datadir=None, segdir=None, transform=None):
        
        self.datadir = datadir
        self.transform = transform

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
        imgdata = np.array([imgfile.read(i) for i in [1,2,3,4,5,6,7,8,9,10,11,12,13]])
        # skip band 11 (Sentinel-2 Band 10, Cirrus) as it does not contain
        # useful information in the case of Level-2A data products

        fptdata = np.loadtxt(self.segfiles[idx], delimiter=",", dtype=float)
        fptdata = np.array(fptdata)
        # fptdata.reshape(fptdata,(120,120))

        sample = {'idx': idx,
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
        self.channel_means = np.array(
            [809.2, 900.5, 1061.4, 1091.7, 1384.5, 1917.8,
             2105.2, 2186.3, 2224.8, 2346.8, 2346.8, 1901.2, 1460.42])
        self.channel_stds = np.array(
            [441.8, 624.7, 640.8, 718.1, 669.1, 767.5,
             843.3, 947.9, 882.4, 813.7, 813.7, 716.9, 674.8])

    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        sample['img'] = (sample['img']-self.channel_means.reshape(
            sample['img'].shape[0], 1, 1))/self.channel_stds.reshape(
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
