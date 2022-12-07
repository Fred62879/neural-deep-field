
from typing import Callable
import torch
from torch.utils.data import Dataset
from wisp.datasets.formats import load_nerf_standard_data, load_rtmv_data
from wisp.core import Rays


class Astro2DDataset(Dataset):
    """This is a static multiview image dataset class.

    This class should be used for training tasks where the task is to fit
    a astro image with 2d architectures.

    """
    def __init__(self,
                 dataset_path             : str,
                 dataset_num_workers      : int      = -1,
                 transform                : Callable = None,
                 **kwargs
    ):
        """Initializes the dataset class.

        Note that the `init` function to actually load images is separate right now, because we don't want
        to load the images unless we have to. This might change later.

        Args:
            dataset_path (str): Path to the dataset.
            multiview_dataset_format (str): The dataset format. Currently supports standard (the same format
                used for instant-ngp) and the RTMV dataset.
            mip (int): The factor at which the images will be downsampled by to save memory and such.
                       Will downscale by 2**mip.
            bg_color (str): The background color to use for images with 0 alpha.
            dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.root = dataset_path
        self.transform = transform
        self.dataset_num_workers = dataset_num_workers

    def init(self):
        """Initializes the dataset.
        """
        self.coords = None
        self.data = self.get_images()
        self.img_shape = self.data["imgs"].shape[1:3]
        self.num_imgs = self.data["imgs"].shape[0]
        self.data["imgs"] = self.data["imgs"].reshape(self.num_imgs, -1, 3)
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(self.num_imgs, -1, 1)

    def get_images(self, split='train'):
        """Will return the dictionary of image tensors.

        Args:
            split (str): The split to use from train, val, test
            mip (int): If specified, will rescale the image by 2**mip.

        Returns:
            (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
        """
        if split == 'train':
            data = load_rtmv_data(self.root, split,
                                  return_pointcloud=True, mip=mip, bg_color=self.bg_color,
                                  normalize=True, num_workers=self.dataset_num_workers)
            self.coords = data["coords"]
            self.coords_center = data["coords_center"]
            self.coords_scale = data["coords_scale"]
        else:
            if self.coords is None:
                assert False and "Initialize the dataset first with the training data!"

            data = load_rtmv_data(self.root, split,
                                  return_pointcloud=False, mip=mip, bg_color=self.bg_color,
                                  normalize=False)
        return data

    def __len__(self):
        """Length of the dataset in number of rays.
        """
        return self.data["imgs"].shape[0]

    def __getitem__(self, idx : int):
        """Returns a ray.
        """
        out = {}
        out['imgs'] = self.data["imgs"][idx]
        if self.transform is not None:
            out = self.transform(out)
        return out

    def get_img_samples(self, idx):
        """Returns a batch of samples from an image, indexed by idx.
        """
        out = {}
        out['imgs'] = self.data["imgs"][idx]
        return out
