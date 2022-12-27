
import torch

from typing import Callable
from torch.utils.data import Dataset
from wisp.datasets.fits_data import FITSData
from wisp.datasets.trans_data import TransData
from wisp.datasets.spectra_data import SpectraData


class AstroDataset(Dataset):
    """ This is a static astronomy image dataset class.
        This class should be used for training tasks where the task is to fit
          a astro image with 2d architectures.
    """
    def __init__(self,
                 device                   : str,
                 tasks                    : list,
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
            self.unbatched_fields = {"trans_data","spectra_supervision_data",
                                     "spectra_recon_data","spectra_dummy_data"}
        else: self.unbatched_fields = set()

    def init(self):
        """ Initializes the dataset.
            Load only needed data based on given tasks.
        """
        #if "test" in self.tasks:
        #    self.test_dataset = TestData(self.root, **kwargs)

        self.fits_dataset = FITSData(self.root, self.tasks, self.device, **self.kwargs)
        self.trans_dataset = TransData(self.root, self.device, **self.kwargs)
        self.spectra_dataset = SpectraData(self.fits_dataset, self.trans_dataset,
                                           self.root, self.tasks, self.device, **self.kwargs)
        # randomly initialize
        self.set_dataset_length(1000)

    ############
    # Setters
    ############

    def set_dataset_length(self, length):
        self.dataset_length = length

    def set_dataset_fields(self, fields):
        self.requested_fields = fields

    ############
    # Getters
    ############

    def get_fits_ids(self):
        return self.fits_dataset.get_fits_ids()

    def get_num_fits(self):
        fits_ids = self.get_fits_ids()
        return len(fits_ids)

    def get_img_sizes (self):
        return self.fits_dataset.get_img_sizes()

    def get_num_coords(self):
        """ Get number of all coordinates. """
        return self.get_coords().shape[0]

    def get_num_gt_spectra_coords(self):
        """ Get number of selected coords with gt spectra. """
        return self.fits_dataset.get_num_gt_spectra_coords()

    def get_zscale_ranges(self, fits_id=None):
        return self.fits_dataset.get_zscale_ranges(fits_id)

    def get_coords(self):
        return self.fits_dataset.get_coords()

    def get_pixels(self):
        return self.fits_dataset.get_pixels()

    def get_weights(self):
        return self.fits_dataset.get_weights()

    def get_masks(self):
        return self.fits_dataset.get_masks()

    def get_batched_data(self, field, idx):
        if field == "coords":
            data = self.get_coords()
        elif field == "pixels":
            data = self.get_pixels()
        elif field == "weights":
            data = self.get_weights()
        elif field == "masks":
            data = self.get_mask()
        else:
            raise ValueError("Unrecognized data field.")
        return data[idx]

    def get_trans_data(self, batch_size, out):
        """ Get transmission data (wave, trans, nsmpl etc.).
            These are not batched, we do monte carlo sampling at every step.
        """
        if "trans_data" not in self.requested_fields: return
        out["wave"], out["trans"], out["nsmpl"] = \
                self.trans_dataset.sample_wave_trans(batch_size, self.kwargs["num_trans_samples"])

    def get_spectra_data(self, out):
        if "spectra_dummy_data" in self.requested_fields: # @infer only
            out["dummy_spectra_coords"] = self.spectra_dataset.get_dummy_spectra_coords()

        if "spectra_supervision_data" in self.requested_fields:
            out["full_wave"] = self.trans_dataset.get_full_norm_wave()
            out["trusted_wave_range_id"] = self.spectra_dataset.get_trusted_wave_range_id()

            spectra_coords = self.spectra_dataset.get_spectra_coords()[:,None]
            if "coords" in out: # @train
                out["coords"] = torch.cat((out["coords"], spectra_coords), dim=0)
                out["gt_spectra"] = self.spectra_dataset.get_supervision_gt_spectra() # on GPU
            else: # @infer
                out["coords"] = spectra_coords
                out["gt_spectra"] = self.spectra_dataset.get_gt_spectra() # on CPU

        elif "spectra_recon_data" in self.requested_fields: # @infer only
            out["gt_spectra"] = self.spectra_dataset.get_gt_spectra()
            spectra_coords = self.spectra_dataset.get_spectra_coords()
            assert("coords" not in out)
            out["coords"] = spectra_coords

    def __len__(self):
        """ Length of the dataset in number of pixels """
        return self.dataset_length

    def __getitem__(self, idx : list):
        """ Sample data from requried fields using given index.
            Needed for training and all coords inferrence.
        """
        # if self.kwargs["debug"]:
        #     out = {"coords": self.coords[idx],
        #            "pixels": self.pixels[idx]}
        #     if self.space_dim == 3:
        #         out["coords"] = out["coords"][:,None] #.tile(1, self.num_samples,1)
        #         out["coords"] = torch.cat((out["coords"], out["wave"]), dim=-1)
        #     return out

        out = {}
        batched_fields = self.requested_fields - self.unbatched_fields

        for field in batched_fields:
            out[field] = self.get_batched_data(field, idx)

        self.get_trans_data(len(idx), out)
        self.get_spectra_data(out)

        if self.transform is not None:
            out = self.transform(out)
        return out

    ############
    # Utilities
    ############

    def restore_evaluate_tiles(self, pixels, func=None, kwargs=None):
        return self.fits_dataset.restore_evaluate_tiles(pixels, func=func, kwargs=kwargs)
