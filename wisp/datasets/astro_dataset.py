
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
        self.root = dataset_path
        self.transform = transform
        self.dataset_num_workers = dataset_num_workers

        self.space_dim = kwargs["space_dim"]
        self.nsmpls = kwargs["num_trans_samples"]

        if self.space_dim == 3:
            self.unbatched_fields = {"trans_data","spectra_supervision_data"}
        else:
            self.unbatched_fields = set()

    def init(self):
        """ Initializes the dataset.
            Load only needed data based on given tasks (in kwargs).
        """
        self.fits_dataset = FITSData(self.root, self.device, **self.kwargs)
        self.trans_dataset = TransData(self.root, self.device, **self.kwargs)
        self.spectra_dataset = SpectraData(self.fits_dataset, self.trans_dataset,
                                           self.root, self.device, **self.kwargs)
        # randomly initialize
        self.state = "fits"
        self.mode = "train"
        self.set_dataset_length(1000)

    ############
    # Setters
    ############

    def set_dataset_mode(self, mode):
        """ Set dataset to be in train or infer mode, which determines
              i) number of transmission samples (use all samples for inferrence)
        """
        self.mode = mode

    def set_dataset_state(self, state):
        """ Set dataset state that controls:
              i) whether load fits coords ("fits") or spectra coords ("spectra")
        """
        self.state = state

    def set_dataset_length(self, length):
        self.dataset_length = length

    def set_dataset_fields(self, fields):
        self.requested_fields = set(fields)

    ############
    # Getters
    ############

    def get_fits_ids(self):
        return self.fits_dataset.get_fits_ids()

    def get_num_fits(self):
        return len(self.get_fits_ids())

    def get_img_sizes (self):
        return self.fits_dataset.get_img_sizes()

    def get_num_coords(self):
        """ Get number of all coordinates. """
        return self.fits_dataset.get_num_coords()

    def get_zscale_ranges(self, fits_id=None):
        return self.fits_dataset.get_zscale_ranges(fits_id)

    def get_spectra_coord_ids(self):
        """ Get id of spectra pixel (in context of all coords). """
        return self.spectra_dataset.get_spectra_coord_ids()

    def get_num_spectra_coords(self):
        """ Get number of selected coords to recon spectra. """
        return self.spectra_dataset.get_num_spectra_coords()

    '''
    def get_gt_spectra(self):
        return self.spectra_dataset.get_gt_spectra()

    def get_gt_spectra_wave(self):
        return self.spectra_dataset.get_gt_spectra_wave()

    def get_recon_spectra_wave(self):
        return self.spectra_dataset.get_recon_spectra_wave()

    def get_recon_wave_bound_ids(self):
        return self.spectra_dataset.get_recon_wave_bound_ids()
    '''

    def get_num_gt_spectra(self):
        return self.spectra_dataset.get_num_gt_spectra()

    def get_batched_data(self, field, idx):
        if field == "coords":
            if self.state == "fits":
                data = self.fits_dataset.get_coords()
                #print("input coord",data.shape)
            elif self.state == "spectra":
                data = self.spectra_dataset.get_spectra_coords()
                #print("input coord",data.shape)

        elif field == "pixels":
            data = self.fits_dataset.get_pixels()
        elif field == "weights":
            data = self.fits_dataset.get_weights()
        elif field == "masks":
            data = self.fits_dataset.get_mask()
        else:
            raise ValueError("Unrecognized data field.")
        return data[idx]

    def get_trans_data(self, batch_size, out):
        """ Get transmission data (wave, trans, nsmpl etc.).
            These are not batched, we do monte carlo sampling at every step.
        """
        infer = self.mode == "infer"
        out["wave"], out["trans"], out["nsmpl"] = \
                self.trans_dataset.sample_wave_trans(batch_size, self.nsmpls, infer=infer)

    def get_spectra_data(self, out):
        """ Get unbatched spectra data (only for spectra supervision training).
        """
        # get only supervision spectra (not all gt spectra)
        out["gt_spectra"] = self.spectra_dataset.get_supervision_spectra()

        # get all coords to plot all gt spectra (not only supervision spectra)
        spectra_coords = self.spectra_dataset.get_gt_spectra_coords()
        if "coords" in out:
            out["coords"] = torch.cat((out["coords"], spectra_coords), dim=0)
        else: out["coords"] = spectra_coords
        #print(spectra_coords.shape, out["coords"].shape)

        out["full_wave"] = self.trans_dataset.get_full_norm_wave()
        out["recon_wave_bound_ids"] = self.spectra_dataset.get_recon_wave_bound_ids()

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

        if "trans_data" in self.requested_fields:
            self.get_trans_data(len(idx), out)
            #print(out["wave"].shape, out["trans"].shape)

        if "spectra_supervision_data" in self.requested_fields:
            self.get_spectra_data(out)

        if self.transform is not None:
            out = self.transform(out)

        return out

    ############
    # Utilities
    ############

    def restore_evaluate_tiles(self, recon_pixels, **re_args):
        """ Restore multiband image save locally and calculate metrics.
            @Return:
               metrics(_z): metrics of current model [n_metrics,1,ntiles,nbands]
        """
        return self.fits_dataset.restore_evaluate_tiles(recon_pixels, **re_args)

    def plot_spectrum(self, fname, recon_spectra):
        self.spectra_dataset.plot_spectrum(fname, recon_spectra)
