

import torch

from typing import Callable
from torch.utils.data import Dataset
from wisp.utils.data import FITSData, \
    generate_cutout_pixl_ids


class AstroDataset(Dataset):
    """ This is a static astronomy image dataset class.
        This class should be used for training tasks where the task is to fit
          a astro image with 2d architectures.
    """
    def __init__(self,
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
        self.root = dataset_path
        self.transform = transform
        self.dataset_num_workers = dataset_num_workers

    def init(self):
        """ Initializes the dataset """
        self.data = {}
        if self.kwargs["is_test"]:
            pixels, coords = utils.load_fake_data(False, self.args)
            self.data['pixels'] = pixels
            self.data['coords'] = coords
        else:
            self.fits_dataset = FITSData(self.root, **self.kwargs)
            num_rows, num_cols = self.fits_dataset.get_img_sz()
            self.data['coords'] = self.fits_dataset.get_coords(to_tensor=False)
            self.data['pixels'] = self.fits_dataset.get_pixels(to_tensor=False)
            #self.data['mask'] = self.fits_dataset.get_mask()
            self.data['weights'] = self.fits_dataset.get_weights(to_tensor=False)

            if self.kwargs["space_dim"] == 3:
                self.data['trans'] = self.trans_dataset.get_trans()

                if self.kwargs["spectra_supervision"]:
                    self.data['spectra'] = self.trans_dataset.get_spectra()

        # get ids of pixels within specified cutout to save
        self.cutout_pixels_ids = generate_cutout_pixl_ids(pos, cutout_sz, num_cols[0])

    ############
    # Sample data
    ############

    def sample_wave(self, batch_sz):
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
                (bsz, self.norm_wave, self.trans, self.distrib, self.args,
                 waves=self.wave, sort=False, counts=self.counts)

        elif self.mc_cho == 'mc_mixture':
            assert(self.encd_ids is not None)
            smpl_wave, smpl_trans, _, nsmpl_within_each_band_mixture = trans_utils.batch_sample_trans \
                (bsz, self.norm_wave, self.trans, self.distrib, self.num_trans_smpl,
                 sort=True, counts=self.counts, encd_ids=self.encd_ids,
                 use_all_wave=self.train_use_all_wave, avg_per_band=self.avg_per_band)
        else:
            raise Exception('Unsupported monte carlo choice')

        self.smpl_wave = smpl_wave   # [bsz,nsmpl,1]/[bsz,nbands,nsmpl,1]
        self.smpl_trans = smpl_trans # [bsz,nbands,nsmpl]
        self.nsmpl_within_each_band_mixture = nsmpl_within_each_band_mixture # [bsz,nbands]

    def resample(self):
        """ Resamples a new working set of SDFs """

    def get_cutotu_pixel_ids(self):
        return self.cutout_pixel_ids

    def __len__(self):
        """ Length of the dataset in number of pixels """
        return self.data["pixels"].shape[0]

    def __getitem__(self, idx : int):
        """ Returns coord and pixel value for pixel.
            Together with sampled lambda values if doing 3d training.
        """
        out = {
            'pixels': self.data['pixels'][idx],
            'coords': self.data['coords'][idx:idx+1] # always sample a 3d coords [bsz,1,dim]
        }

        if self.kwargs["weight_train"]:
            out["weights"] = self.data['weights'][idx]

        if self.kwargs["space_dim"] == 3:
            out["wave"] = self.data['wave'][idx:idx+1]
            out["trans"] = self.data['trans'][idx]

        if self.transform is not None:
            out = self.transform(out)
        return out
