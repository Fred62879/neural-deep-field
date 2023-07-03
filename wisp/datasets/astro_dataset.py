
import torch
import numpy as np

from typing import Callable
from torch.utils.data import Dataset
from wisp.utils.common import print_shape
from wisp.datasets.fits_data import FitsData
from wisp.datasets.mask_data import MaskData
from wisp.datasets.trans_data import TransData
from wisp.datasets.spectra_data import SpectraData
from wisp.datasets.data_utils import clip_data_to_ref_wave_range


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
        self.trans_dataset = TransData(self.root, self.device, **self.kwargs)
        self.spectra_dataset = SpectraData(self.trans_dataset,
                                           self.root, self.device, **self.kwargs)
        self.fits_dataset = FitsData(self.root, self.device, self.spectra_dataset, **self.kwargs)
        self.mask_dataset = MaskData(self.fits_dataset, self.root, self.device, **self.kwargs)

        # randomly initialize
        self.set_length(0)
        self.mode = "main_train"
        self.coords_source = "fits"
        self.use_full_wave = False
        self.perform_integration = True
        self.use_predefined_wave_range = False

    ############
    # Setters
    ############

    def set_mode(self, mode):
        """ Possible modes: ["pre_train","pretrain_infer","infer","main_train"]
        """
        self.mode = mode

    def toggle_wave_sampling(self, use_full_wave: bool):
        self.use_full_wave = use_full_wave

    def toggle_within_wave_range(self, use_wave_range: bool):
        self.use_predefined_wave_range = use_wave_range

    def set_wave_range(self, lo, hi):
        self.wave_range = (lo, hi)

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

    def get_num_patches(self):
        return len(self.get_patch_uids())


    def get_patch_uids(self):
        return self.fits_dataset.get_patch_uids()

    def get_num_coords(self):
        return self.fits_dataset.get_num_coords()

    def get_zscale_ranges(self, patch_uid=None):
        return self.fits_dataset.get_zscale_ranges(patch_uid)


    def get_spectra_coord_ids(self):
        return self.spectra_dataset.get_spectra_coord_ids()

    def get_spectra_img_coords(self):
        return self.spectra_dataset.get_spectra_img_coords()

    def get_validation_spectra_ids(self, patch_uid=None):
        return self.spectra_dataset.get_validation_spectra_ids(patch_uid)

    def get_validation_spectra_coords(self, idx=None):
        return self.spectra_dataset.get_validation_coords(idx)

    def get_validation_spectra_fluxes(self, idx=None):
        return self.spectra_dataset.get_validation_fluxes(idx)

    def get_validation_spectra_pixels(self, idx=None):
        return self.spectra_dataset.get_validation_pixels(idx)

    def get_supervision_spectra_pixels(self):
        return self.spectra_dataset.get_supervision_pixels()

    def get_supervision_spectra_redshift(self):
        return self.spectra_dataset.get_supervision_redshift()

    def get_num_gt_spectra(self):
        return self.spectra_dataset.get_num_gt_spectra()

    def get_num_spectra_coords(self):
        return self.spectra_dataset.get_num_spectra_coords()

    def get_num_spectra_to_plot(self):
        return self.spectra_dataset.get_num_spectra_to_plot()

    def get_num_supervision_spectra(self):
        return self.spectra_dataset.get_num_supervision_spectra()

    def get_num_validation_spectra(self):
        return self.spectra_dataset.get_num_validation_spectra()


    def get_full_wave(self):
        return self.trans_dataset.get_full_wave()

    def get_full_wave_bound(self):
        return self.trans_dataset.get_full_wave_bound()


    def get_batched_data(self, field, idx):
        if field == "coords":
            if self.coords_source == "fits":
                data = self.fits_dataset.get_coords(idx)
            else:
                data = self.data[self.coords_source][idx]
        elif field == "pixels":
            data = self.fits_dataset.get_pixels(idx)
        elif field == "weights":
            data = self.fits_dataset.get_weights(idx)
        elif field == "spectra_id_map":
            data = self.fits_dataset.get_spectra_id_map(idx)
        elif field == "spectra_bin_map":
            data = self.fits_dataset.get_spectra_bin_map(idx)
        elif field == "spectra_sup_fluxes":
            data = self.spectra_dataset.get_supervision_fluxes(idx)
        elif field == "spectra_sup_pixels":
            data = self.spectra_dataset.get_supervision_pixels(idx)
        elif field == "spectra_sup_redshift":
            #print(idx[:10])
            data = self.spectra_dataset.get_supervision_redshift(idx)
            #print(data[:10])
        elif field == "spectra_sup_wave_bound_ids":
            data = self.spectra_dataset.get_supervision_wave_bound_ids()
        elif field == "masks":
            data = self.mask_dataset.get_mask(idx)
        else:
            raise ValueError("Unrecognized data field.")
        return data

    def get_trans_data(self, batch_size, out):
        """ Get transmission data (wave, trans, nsmpl etc.).
            These are not batched, we do sampling at every step.
        """
        assert not (self.use_full_wave and self.use_predefined_wave_range)

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

        elif self.use_predefined_wave_range:
            full_wave = self.get_full_wave()
            clipped_wave, _ = clip_data_to_ref_wave_range(
                full_wave, full_wave, wave_range=self.wave_range)
            out["wave"] = torch.FloatTensor(clipped_wave)[None,:,None].tile(batch_size,1,1)
        else:
            out["wave"] = torch.FloatTensor(self.get_full_wave())
            out["wave"] = out["wave"][None,:,None].tile(batch_size,1,1)

    def get_spectra_data(self, out):
        """ Get unbatched spectra data during
              i)   codebook pretrain or
              ii)  main train after pretrain or
              iii) spectra supervision training (without pretrain)
            Used only with very small amount of spectra.
            If we train on large amount of spectra, use batched data instead.
        """
        assert(self.kwargs["pretrain_codebook"] ^ self.kwargs["spectra_supervision"])

        if self.kwargs["pretrain_codebook"]:
            if self.mode == "pre_train":
                n = self.spectra_dataset.get_num_gt_spectra()
                out["coords"] = self.data["spectra_latents"][:n]
                if self.kwargs["codebook_pretrain_pixel_supervision"]:
                    out["spectra_sup_pixels"] = self.spectra_dataset.get_supervision_pixels()

                # get only supervision spectra (not all gt spectra) for loss calculation
                out["spectra_sup_fluxes"] = \
                    self.spectra_dataset.get_supervision_fluxes()
                out["spectra_sup_redshift"] = \
                    self.spectra_dataset.get_supervision_redshift()
                out["spectra_sup_wave_bound_ids"] = \
                    self.spectra_dataset.get_supervision_wave_bound_ids()

            elif self.mode == "pretrain_infer":
                out["spectra_sup_fluxes"] = \
                    self.spectra_dataset.get_supervision_fluxes()
                out["spectra_sup_redshift"] = \
                    self.spectra_dataset.get_supervision_redshift()

                if self.kwargs["infer_selected"]:
                    out["spectra_sup_fluxes"] = out["spectra_sup_fluxes"][
                        self.data["selected_ids"]]
                    out["spectra_sup_redshift"] = out["spectra_sup_redshift"][
                        self.data["selected_ids"]]

                #print('infer', out["spectra_sup_redshift"])

            elif self.mode == "main_train": # or self.mode == "infer":
                bin_map = out["spectra_bin_map"]
                ids = out["spectra_id_map"][bin_map]
                out["spectra_val_ids"] = ids
                # out["spectra_val_fluxes"] = self.fits_dataset.get_spectra_pixel_fluxes(ids)
                out["spectra_sup_redshift"] = self.fits_dataset.get_spectra_pixel_redshift(ids)
                del out["spectra_id_map"]

        elif self.kwargs["spectra_supervision"]:
            assert self.mode == "main_train"
            out["full_wave"] = self.get_full_wave()

            # get all coords to plot all spectra (gt, dummy, incl. neighbours)
            spectra_coords = self.spectra_dataset.get_spectra_grid_coords()
            if "coords" in out:
                out["coords"] = torch.cat((out["coords"], spectra_coords), dim=0)
            else: out["coords"] = spectra_coords

            # the first #num_supervision_spectra are gt coords for supervision
            # the others are forwarded only for spectrum plotting
            out["num_spectra_coords"] = len(spectra_coords)

            if self.kwargs["redshift_semi_supervision"]:
                out["spectra_sup_redshift"] = self.spectra_dataset.get_supervision_redshift()

    def get_redshift_data(self, out):
        """ Get validation redshift values (only when apply gt redshift directly).
        """
        # out["spectra_valid_redshift"] = self.spectra_dataset.get_validation_redshift()
        out["spectra_valid_redshift"] = self.fits_dataset.get_spectra_pixel_redshift()

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
        # print(batched_fields, self.requested_fields)

        for field in batched_fields:
            out[field] = self.get_batched_data(field, idx)

        if "trans_data" in self.requested_fields:
            self.get_trans_data(len(idx), out)

        if "spectra_data" in self.requested_fields:
            self.get_spectra_data(out)

        if "redshift_data" in self.requested_fields:
           self.get_redshift_data(out)

        print_shape(out)
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

    def plot_spectrum(self, spectra_dir, name, recon_fluxes, flux_norm_cho,
                      clip=True, spectra_clipped=False, is_codebook=False,
                      save_spectra=False, save_spectra_together=False,
                      mode="pretrain_infer", ids=None
    ):
        self.spectra_dataset.plot_spectrum(
            spectra_dir, name, recon_fluxes, flux_norm_cho,
            clip=clip,
            spectra_clipped=spectra_clipped,
            is_codebook=is_codebook,
            save_spectra=save_spectra,
            save_spectra_together=save_spectra_together,
            mode=mode, ids=ids)

    def log_spectra_pixel_values(self, spectra):
        return self.spectra_dataset.log_spectra_pixel_values(spectra)
