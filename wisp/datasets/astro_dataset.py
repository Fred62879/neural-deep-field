
import torch
import numpy as np

from typing import Callable
from torch.utils.data import Dataset
from wisp.datasets.fits_data import FITSData
from wisp.datasets.mask_data import MaskData
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
                 dataset_num_workers      : int      = -1, # not used
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
            self.unbatched_fields = {
                "trans_data","spectra_data","redshift_data"
            }
        else:
            self.unbatched_fields = set()

    def init(self):
        """ Initializes the dataset.
            Load only needed data based on given tasks (in kwargs).
        """
        self.data = {}
        self.fits_dataset = FITSData(self.root, self.device, **self.kwargs)
        self.trans_dataset = TransData(self.root, self.device, **self.kwargs)
        self.spectra_dataset = SpectraData(self.fits_dataset, self.trans_dataset,
                                           self.root, self.device, **self.kwargs)
        self.mask_dataset = MaskData(self.fits_dataset, self.root, self.device, **self.kwargs)

        # randomly initialize
        self.set_length(0)
        self.mode = "train"
        self.coords_source = "fits"
        self.use_full_wave = False
        self.perform_integration = True

    ############
    # Setters
    ############

    def set_mode(self, mode):
        self.mode = mode

    def toggle_wave_sampling(self, use_full_wave: bool):
        self.use_full_wave = use_full_wave

    def toggle_integration(self, integrate: bool):
        self.perform_integration = integrate

    def set_coords_source(self, coords_source):
        """ Set dataset source of coords that controls:
              i) whether load fits coords ("fits") or spectra coords ("spectra")
        """
        self.coords_source = coords_source

    def set_length(self, length):
        self.dataset_length = length

    def set_fields(self, fields):
        self.requested_fields = set(fields)

    def set_hardcode_data(self, field, data):
        self.data[field] = data

    ############
    # Getters
    ############

    def get_fits_uids(self):
        return self.fits_dataset.get_fits_uids()

    def get_num_fits(self):
        return len(self.get_fits_uids())

    def get_num_coords(self):
        return self.fits_dataset.get_num_coords()

    def get_zscale_ranges(self, fits_uid=None):
        return self.fits_dataset.get_zscale_ranges(fits_uid)

    def get_spectra_coord_ids(self):
        return self.spectra_dataset.get_spectra_coord_ids()

    def get_spectra_img_coords(self):
        return self.spectra_dataset.get_spectra_img_coords()

    def get_num_spectra_to_plot(self):
        return self.spectra_dataset.get_num_spectra_to_plot()

    def get_num_gt_spectra(self):
        return self.spectra_dataset.get_num_gt_spectra()

    def get_num_spectra_coords(self):
        return self.spectra_dataset.get_num_spectra_coords()

    def get_full_wave(self):
        return self.trans_dataset.get_full_wave()

    def get_full_wave_bound(self):
        return self.trans_dataset.get_full_wave_bound()

    def get_batched_data(self, field, idx):
        if field == "coords":
            if self.coords_source == "fits":
                data = self.fits_dataset.get_coords(idx)
            elif self.coords_source == "spectra":
                data = self.spectra_dataset.get_spectra_grid_coords()
            else:
                data = self.data[self.coords_source][idx]
        elif field == "pixels":
            data = self.fits_dataset.get_pixels(idx)
        elif field == "weights":
            data = self.fits_dataset.get_weights(idx)
        elif field == "redshift":
            assert 0
            data = self.fits_dataset.get_redshifts(idx)
        elif field == "masks":
            data = self.mask_dataset.get_mask(idx)
        else:
            raise ValueError("Unrecognized data field.")
        return data

    def get_trans_data(self, batch_size, out):
        """ Get transmission data (wave, trans, nsmpl etc.).
            These are not batched, we do sampling at every step.
        """
        # trans wave min and max value (used for linear normalization)
        out["full_wave_bound"] = self.get_full_wave_bound()

        if self.perform_integration:
            if self.kwargs["trans_sample_method"] == "hardcode":
                out["wave"] = self.trans_dataset.get_hdcd_wave()
                out["wave"] = out["wave"][None,:,None].tile(batch_size,1,1)
                out["trans"] = self.trans_dataset.get_hdcd_trans()
                out["nsmpl"] = self.trans_dataset.get_hdcd_nsmpl()
            else:
                out["wave"], out["trans"], out["wave_smpl_ids"], out["nsmpl"] = \
                    self.trans_dataset.sample_wave_trans(
                        batch_size, self.nsmpls, use_full_wave=self.use_full_wave)

            if self.mode == "infer" and "recon_synthetic_band" in self.kwargs["tasks"]:
                assert(self.use_full_wave) # only in inferrence
                nsmpl = out["trans"].shape[1]
                out["trans"] = torch.cat((out["trans"], torch.ones(1,nsmpl)), dim=0)
                out["nsmpl"] = torch.cat((out["nsmpl"], torch.tensor([nsmpl])), dim=0)
        else:
            out["wave"] = torch.FloatTensor(self.get_full_wave())
            out["wave"] = out["wave"][None,:,None].tile(batch_size,1,1)

    def get_spectra_data(self, out):
        """ Get unbatched spectra data
            For either spectra supervision training or codebook pretrain.
        """
        assert(self.kwargs["pretrain_codebook"] ^ self.kwargs["spectra_supervision"])

        # get only supervision spectra (not all gt spectra) for loss calculation
        out["gt_spectra"] = self.spectra_dataset.get_supervision_spectra()

        out["spectra_supervision_wave_bound_ids"] = \
            self.spectra_dataset.get_spectra_supervision_wave_bound_ids()

        if self.kwargs["pretrain_codebook"]:
            out["coords"] = self.data["spectra_latents"]
        else:
            out["full_wave"] = self.get_full_wave()

            # get all coords to plot all spectra (gt, dummy, incl. neighbours)
            spectra_coords = self.spectra_dataset.get_spectra_grid_coords()
            if "coords" in out:
                out["coords"] = torch.cat((out["coords"], spectra_coords), dim=0)
            else:
                out["coords"] = spectra_coords

            # the first #num_supervision_spectra are gt coords for supervision
            # the others are forwarded only for spectrum plotting
            out["num_spectra_coords"] = len(spectra_coords)

    def get_redshift_data(self, out):
        out["redshift"] = self.spectra_dataset.get_redshift()

    def __len__(self):
        """ Length of the dataset in number of coords.
        """
        return self.dataset_length

    def __getitem__(self, idx: list):
        """ Sample data from requried fields using given index.
            Also get unbatched data (trans, spectra etc.).
        """
        out = {}
        batch_size = len(idx)
        batched_fields = self.requested_fields - self.unbatched_fields

        for field in batched_fields:
            out[field] = self.get_batched_data(field, idx)

        if "trans_data" in self.requested_fields:
            self.get_trans_data(len(idx), out)

        if "spectra_data" in self.requested_fields:
            self.get_spectra_data(out)

        if "redshift_data" in self.requested_fields:
            self.get_redshift_data(out)

        # self.print_shape(out)
        if self.transform is not None:
            out = self.transform(out)
        return out

    ############
    # Utilities
    ############

    def restore_evaluate_tiles(self, recon_pixels, **re_args):
        """ Restore flattened image, save locally and/or calculate metrics.
            @Return:
               metrics(_z): metrics of current model [n_metrics,1,ntiles,nbands]
        """
        return self.fits_dataset.restore_evaluate_tiles(recon_pixels, **re_args)

    def plot_spectrum(self, spectra_dir, name, recon_spectra, spectra_norm_cho, save_spectra=False, clip=True, codebook=False):
        self.spectra_dataset.plot_spectrum(
            spectra_dir, name, recon_spectra, spectra_norm_cho, save_spectra=save_spectra, clip=clip, codebook=codebook)

    def log_spectra_pixel_values(self, spectra):
        return self.spectra_dataset.log_spectra_pixel_values(spectra)

    def print_shape(self, out):
        for n,p in out.items():
            # print(n, type(p))
            if type(p) == tuple or type(p) == list:
                print(n, len(p))
            else: print(n, p.shape)
