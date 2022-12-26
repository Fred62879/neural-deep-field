
import torch

from typing import Callable
from torch.utils.data import Dataset
from wisp.utils.fits_data import FITSData
from wisp.utils.trans_data import TransData


class AstroDataset(Dataset):
    """ This is a static astronomy image dataset class.
        This class should be used for training tasks where the task is to fit
          a astro image with 2d architectures.
    """
    def __init__(self,
                 device                   : str,
                 tasks                    : list, # test/train/inferrence (incl. img recon etc.)
                 dataset_path             : str,
                 dataset_num_workers      : int      = -1,
                 transform                : Callable = None,
                 **kwargs):
        """ Initializes the dataset class.
            @Param:
              dataset_path (str): Path to the dataset.
              dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.kwargs = kwargs

        self.device = device
        self.tasks = set(tasks)
        self.root = dataset_path
        self.transform = transform
        self.dataset_num_workers = dataset_num_workers

        self.space_dim = kwargs["space_dim"]
        if self.space_dim == 3:
            self.unbatched_fields = {"wave","trans","nsmpl"}
        else: self.unbatched_fields = set()

    def init(self):
        """ Initializes the dataset.
            Load all needed data based on given tasks.
        """
        self.data = {}

        if "test" in self.tasks:
            pixels, coords = utils.load_fake_data(False, self.args)
            self.data['pixels'] = pixels[:,None,:]
            self.data['coords'] = coords[:,None,:]
            return

        self.require_full_coords = "train" in self.tasks or \
            ("recon_img" in self.tasks or "recon_flat" in self.tasks)

        self.require_pixels = "train" in self.tasks or "recon_img" in self.tasks
        self.require_weights = "train" in self.tasks and self.kwargs["weight_train"]
        self.require_masks = "train" in self.tasks and self.kwargs["inpaint_cho"] == "spectral_inpaint"

        if self.require_full_coords or self.require_pixels or self.require_weights:
            self.fits_dataset = FITSData(self.root, self.device, **self.kwargs)
            self.fits_ids = self.fits_dataset.get_fits_ids()
            self.num_rows, self.num_cols = self.fits_dataset.get_img_sizes()

            if self.require_full_coords:
                self.data['coords'] = self.fits_dataset.get_coords().to(self.device)[:,None]
                #self.coords = self.fits_dataset.get_coords().to(self.device)[:,None]
            if self.require_pixels:
                self.data['pixels'] = self.fits_dataset.get_pixels().to(self.device)
                #self.pixels = self.fits_dataset.get_pixels().to(self.device)
            if self.require_weights:
                self.data['weights'] = self.fits_dataset.get_weights()
            if self.require_masks:
                self.data['masks'] = self.fits_dataset.get_mask()

        if self.kwargs["space_dim"] == 3:
            self.trans_dataset = TransData(self.root, self.device, **self.kwargs)
            self.num_samples = self.kwargs["num_trans_samples"]
            if self.kwargs["spectra_supervision"]:
                self.data['spectra'] = self.spectra_dataset.get_spectra()

        # randomly initialize
        self.set_dataset_length(1000)

    ############
    # Setters
    ############

    def set_dataset_length(self, length):
        self.dataset_length = length

    def set_dataset_fields(self, fields):
        self.dataset_fields = fields

    ############
    # Getters
    ############

    def get_recon_cutout_gt(self, cutout_pixel_ids):
        """ Get gt cutout from loaded pixels. """
        sz = self.kwargs["recon_cutout_size"]
        return self.data["pixels"][cutout_pixel_ids].T.reshape((-1, sz, sz))

    def get_recon_cutout_pixel_ids(self):
        """ Get pixel ids of cutout to reconstruct. """
        return get_recon_cutout_pixel_ids(
            self.kwargs["recon_cutout_start_pos"],
            self.kwargs["fits_cutout_size"],
            self.kwargs["recon_cutout_size"],
            self.num_rows, self.num_cols,
            self.kwargs["recon_cutout_tile_id"],
            self.kwargs["use_full_fits"])

    def get_num_fits(self):
        return len(self.fits_ids)

    def get_fits_ids(self):
        return self.fits_ids

    def get_img_sizes (self):
        return self.num_rows, self.num_cols

    def get_num_coords(self):
        """ Get number of all coordinates. """
        return self.data["coords"].shape[0]
        #return self.coords.shape[0]

    def get_zscale_ranges(self, fits_id=None):
        return self.fits_dataset.get_zscale_ranges(fits_id)

    def get_num_spectra_coords(self):
        """ Get number of selected coords with gt spectra. """
        return self.data["spectra_coords"].shape[0]

    def __len__(self):
        """ Length of the dataset in number of pixels """
        return self.dataset_length

    def __getitem__(self, idx : list):
        """ Sample data from requried fields using given index. """

        if self.kwargs["debug"]:
            out = {"coords": self.coords[idx],
                   "pixels": self.pixels[idx]}
            return out

        out = {}
        requested_fields = set(self.dataset_fields)
        batched_fields = list(requested_fields - self.unbatched_fields)

        for field in batched_fields:
            out[field] = self.data[field][idx]

        # fields that are not batched (we do monte carlo sampling at every step)
        if len(requested_fields.intersection(self.unbatched_fields)) != 0:
            batch_size = len(idx)
            out["wave"], out["trans"], out["nsmpl"] = \
                self.trans_dataset.sample_wave_trans(batch_size, self.num_samples)
            out["wave"] = out["wave"][...,None]

            # embed ra/dec with wave together
            #out["coords"] = out["coords"][:,None].tile(1, self.num_samples,1)
            #out["coords"] = torch.cat((out["coords"], out["wave"]), dim=-1)
        else:
            out["coords"] = out["coords"][:,None]

        if self.transform is not None:
            out = self.transform(out)
        return out

    ############
    # Utilities
    ############

    def restore_evaluate_tiles(self, pixels, func=None, kwargs=None):
        return self.fits_dataset.restore_evaluate_tiles(pixels, func=func, kwargs=kwargs)
