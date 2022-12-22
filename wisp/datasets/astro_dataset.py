
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

        self.tasks = set(tasks)
        self.root = dataset_path
        self.transform = transform
        self.dataset_num_workers = dataset_num_workers

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
            self.fits_dataset = FITSData(self.root, **self.kwargs)
            self.fits_ids = self.fits_dataset.get_fits_ids()
            self.num_rows, self.num_cols = self.fits_dataset.get_img_sizes()

            if self.require_full_coords:
                self.data['coords'] = self.fits_dataset.get_coords(to_tensor=False)[:,None,:]
            if self.require_pixels:
                self.data['pixels'] = self.fits_dataset.get_pixels(to_tensor=False) #[:,None,:]
            if self.require_weights:
                self.data['weights'] = self.fits_dataset.get_weights(to_tensor=False) #[:,None,:]
            if self.require_masks:
                self.data['masks'] = self.fits_dataset.get_mask()

        if self.kwargs["space_dim"] == 3:
            #self.require_full_wave = "train" not in self.tasks or self.kwargs["train_with_full_wave"]
            #self.require_trans = self.status == "train"
            # assume reconstruction always use all lambda

            self.trans_dataset = TransData(self.root, **self.kwargs)
            self.trans_dataset.get_trans()

            if self.kwargs["spectra_supervision"]:
                self.data['spectra'] = self.trans_dataset.get_spectra()

        # randomly initialize
        self.set_dataset_length(1)

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

    def sample_wave(self, batch_size, num_samples):
        """ Sample lambda randomly for each pixel at the begining of each iteration
            For mixture sampling, we also record # samples falling
              within response range of each band. each band may have
              different # of samples (bandwise has this as a hyper-para)
        """
        nsmpl_within_each_band_mixture = None
        if self.mc_cho == 'mc_hardcode':
            smpl_wave, smpl_trans = None, self.trans

        elif self.mc_cho == 'mc_bandwise':
            smpl_wave, smpl_trans, _ = trans_utils.batch_sample_trans_bandwise \
                (batch_size, self.norm_wave, self.trans, self.distrib, self.args,
                 waves=self.wave, sort=False, counts=self.counts)

        elif self.mc_cho == 'mc_mixture':
            assert(self.encd_ids is not None)
            smpl_wave, smpl_trans, _, nsmpl_within_each_band_mixture = trans_utils.batch_sample_trans \
                (batch_size, self.norm_wave, self.trans, self.distrib, self.num_trans_smpl,
                 sort=True, counts=self.counts, encd_ids=self.encd_ids,
                 use_all_wave=self.train_use_all_wave, avg_per_band=self.avg_per_band)
        else:
            raise Exception('Unsupported monte carlo choice')

        self.smpl_wave = smpl_wave   # [bsz,nsmpl,1]/[bsz,nbands,nsmpl,1]
        self.smpl_trans = smpl_trans # [bsz,nbands,nsmpl]
        self.nsmpl_within_each_band_mixture = nsmpl_within_each_band_mixture # [bsz,nbands]

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
        out = {}
        for field in self.dataset_fields:
            out[field] = self.data[field][idx]

        # if self.require_pixels:
        #     out["pixels"] = self.data["pixels"][idx]
        # if self.require_coords:
        #     out["coords"] = self.data["coords"][idx] # always sample a 3d coords [bsz,1,dim]
        # if self.require_weights:
        #     out["weights"] = self.data["weights"][idx]
        # if self.require_masks:
        #     out["masks"] = self.data["masks"][idx]
        # print('got item', (out['coords']).shape)
        # print('got item', (out['pixels']).shape)

        if self.transform is not None:
            out = self.transform(out)
        return out

    ############
    # Utilities
    ############

    def restore_evaluate_tiles(self, pixels, func=None, kwargs=None):
        return self.fits_dataset.restore_evaluate_tiles(pixels, func=func, kwargs=kwargs)
