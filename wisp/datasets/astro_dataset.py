
import torch
import numpy as np

from os.path import exists

from typing import Callable
from torch.utils.data import Dataset
from wisp.utils.common import print_shape
from wisp.datasets.fits_data import FitsData
from wisp.datasets.mask_data import MaskData
from wisp.datasets.trans_data import TransData
from wisp.datasets.spectra_data import SpectraData
from wisp.datasets.data_utils import clip_data_to_ref_wave_range, \
    get_wave_range_fname, batch_sample_torch, get_bound_id


class AstroDataset(Dataset):
    """ Wrapper dataset class for astronomy data.
    """
    def __init__(self,
                 device                   : str,
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
        self.transform = transform
        self.dataset_num_workers = dataset_num_workers

        self.root = kwargs["dataset_path"]
        self.space_dim = kwargs["space_dim"]
        self.num_bands = kwargs["num_bands"]

        if self.space_dim == 3:
            self.unbatched_fields = {
                "wave_data","spectra_data","redshift_data"
            }
        else:
            self.unbatched_fields = set()

    def init(self):
        """ Initializes the dataset.
            Load only needed data based on given tasks (in kwargs).
        """
        self.data = {}
        self.trans_dataset = TransData(self.device, **self.kwargs)
        self.spectra_dataset = SpectraData(self.trans_dataset,
                                           self.device, **self.kwargs)
        self.fits_dataset = FitsData(self.device, self.spectra_dataset, **self.kwargs)
        self.mask_dataset = MaskData(self.fits_dataset, self.device, **self.kwargs)
        self.spectra_dataset.finalize_spectra()

        wave_range_fname = get_wave_range_fname(**self.kwargs)
        self.data["wave_range"] = np.load(wave_range_fname)

        # randomly initialize
        self.set_length(0)
        self.mode = "main_train"
        self.wave_source = "trans"
        self.coords_source = "fits"
        self.sample_wave = False
        self.use_all_wave = True
        self.infer_selected = False
        self.perform_integration = True
        self.use_predefined_wave_range = False

    ############
    # Setters
    ############

    def set_mode(self, mode):
        """ Possible modes: ["codebook_pretrain","main_train","pretrain_infer","main_infer","test"]
        """
        self.mode = mode

    def set_length(self, length):
        self.dataset_length = length

    def set_fields(self, fields):
        self.requested_fields = set(fields)

    def set_wave_range(self, lo, hi):
        self.wave_range = (lo, hi)

    def set_wave_source(self, wave_source):
        """ Set dataset source of wave that controls:
              whether load transmission wave ("trans") or spectra wave ("spectra")
        """
        self.wave_source = wave_source

    def set_coords_source(self, coords_source):
        """ Set dataset source of coords that controls:
              whether load fits coords ("fits") or spectra coords ("spectra")
        """
        self.coords_source = coords_source

    def set_hardcode_data(self, field, data):
        self.data[field] = data

    def toggle_integration(self, integrate: bool):
        self.perform_integration = integrate

    def toggle_wave_sampling(self, sample_wave: bool):
        self.sample_wave = sample_wave
        self.use_all_wave = not sample_wave

    def toggle_selected_inferrence(self, infer_selected: bool):
        self.infer_selected = infer_selected

    ############
    # Getters
    ############

    def get_trans_data_obj(self):
        return self.trans_dataset

    def get_trans_wave_range(self):
        return self.trans_dataset.get_trans_wave_range()

    def get_transmission_interpolation_function(self):
        return self.trans_dataset.get_transmission_interpolation_function()

    def get_spectra_data_obj(self):
        return self.spectra_dataset

    def get_num_patches(self):
        return len(self.get_patch_uids())


    def get_patch_uids(self):
        return self.fits_dataset.get_patch_uids()

    def get_num_coords(self):
        return self.fits_dataset.get_num_coords()

    def get_zscale_ranges(self, patch_uid=None):
        return self.fits_dataset.get_zscale_ranges(patch_uid)

    def get_coords(self):
        return self.fits_dataset.get_coords()

    def get_full_spectra_wave_mask(self):
        return self.spectra_dataset.get_full_wave_mask()

    def get_full_spectra_wave_coverage(self):
        return self.spectra_dataset.get_full_wave_coverage()


    def get_test_spectra(self, idx=None):
        return self.spectra_dataset.get_test_spectra(idx)

    def get_test_spectra_masks(self):
        return self.spectra_dataset.get_test_masks()

    def get_test_spectra_coords(self, idx=None):
        return self.spectra_dataset.get_test_coords(idx)

    # def get_validation_spectra_ids(self, patch_uid=None):
    #     return self.spectra_dataset.get_validation_spectra_ids(patch_uid)

    def get_validation_spectra(self, idx=None):
        return self.spectra_dataset.get_validation_spectra(idx)

    def get_validation_spectra_masks(self):
        return self.spectra_dataset.get_validation_masks()

    def get_validation_spectra_coords(self, idx=None):
        return self.spectra_dataset.get_validation_coords(idx)

    def get_validation_spectra_pixels(self, idx=None):
        return self.spectra_dataset.get_validation_pixels(idx)

    def get_supervision_spectra(self):
        return self.spectra_dataset.get_supervision_spectra()

    def get_supervision_spectra_masks(self):
        return self.spectra_dataset.get_supervision_masks()

    def get_supervision_spectra_pixels(self):
        return self.spectra_dataset.get_supervision_pixels()

    def get_supervision_spectra_redshift(self):
        return self.spectra_dataset.get_supervision_redshift()

    def get_semi_supervision_spectra_redshift(self):
        return self.spectra_dataset.get_semi_supervision_redshift()


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

    def get_num_test_spectra(self):
        return self.spectra_dataset.get_num_test_spectra()

    def get_selected_ids(self):
        assert "selected_ids" in self.data
        return self.data["selected_ids"]


    def get_full_wave(self):
        return self.trans_dataset.get_full_wave()

    def get_full_wave_masks(self):
        return self.trans_dataset.get_full_wave_masks()

    def get_wave_range(self):
        """ Get wave min and max value (used for linear normalization)
        """
        # return self.trans_dataset.get_full_wave_bound()
        return self.data["wave_range"]

    def index_selected_data(self, data, idx):
        """ Index data with both selected_ids and given idx
              (for selective spectra inferrence only)
            @Param
               selected_ids: select from source data (filter index)
               idx: dataset index (batch index)
        """
        if self.mode == "pretrain_infer" and self.infer_selected:
            assert "selected_ids" in self.data
            data = data[self.data["selected_ids"]]
        return data[idx]

    def get_batched_data(self, field, idx):
        if field == "coords":
            if self.coords_source == "fits":
                data = self.fits_dataset.get_coords()
            else:
                data = self.data[self.coords_source]

        elif field == "pixels":
            data = self.fits_dataset.get_pixels()
        elif field == "weights":
            data = self.fits_dataset.get_weights()
        elif field == "spectra_id_map":
            data = self.fits_dataset.get_spectra_id_map()
        elif field == "spectra_bin_map":
            data = self.fits_dataset.get_spectra_bin_map()

        elif field == "spectra_val_pixels":
            data = self.get_validation_spectra_pixels()

        elif field == "spectra_sup_data":
            data = self.get_supervision_spectra()
        elif field == "spectra_sup_plot_mask":
            data = self.get_supervision_spectra_masks()
        elif field == "spectra_sup_pixels":
            data = self.get_supervision_spectra_pixels()
        elif field == "spectra_sup_redshift":
            data = self.get_supervision_spectra_redshift()
        elif field == "spectra_semi_sup_redshift":
            data = self.get_semi_supervision_spectra_redshift()

        elif field == "masks":
            data = self.mask_dataset.get_mask(idx)
        else:
            raise ValueError(f"Unrecognized data field: {field}.")

        data = self.index_selected_data(data, idx)
        return data

    def get_wave_data(self, batch_size, out):
        """ Get wave (lambda and transmission) data depending on data source.
        """
        out["wave_range"] = self.get_wave_range()

        if self.wave_source == "full_spectra":
            # for codebook spectra recon
            mask = self.get_full_spectra_wave_mask()
            full_wave = self.get_full_spectra_wave_coverage()

            bsz = out["coords"].shape[0]
            out["wave"] = full_wave[None,:,None].tile(bsz,1,1)
            out["spectra_sup_plot_mask"] = mask[None,:].tile(bsz,1)

        elif self.wave_source == "spectra":
            # spectra_sup_data: [bsz,4+2*nbands,nsmpl]
            #  (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))

            if self.sample_wave:
                # sample from spectra data (wave, flux, ivar, and interpolated trans)
                # sample_ids [bsz,nsmpl,2]
                assert self.kwargs["uniform_sample_wave"]

                out["spectra_sup_data"], sample_ids = batch_sample_torch(
                    out["spectra_sup_data"], self.kwargs["pretrain_num_wave_samples"],
                    keep_sample_ids=True)

                out["spectra_sup_plot_mask"] = batch_sample_torch(
                    out["spectra_sup_plot_mask"], self.kwargs["pretrain_num_wave_samples"],
                    sample_ids=sample_ids)

            if self.perform_integration:
                # out["trans_mask"] = out["spectra_sup_data"][:,3]              # [bsz,nsmpl]
                out["trans"] = out["spectra_sup_data"][:,4:4+self.num_bands]    # [bsz,nbands,nsmpl]
                out["band_mask"] = out["spectra_sup_data"][:,4+self.num_bands:] # [bsz,nbands,nsmpl]
                # num of sample within each band (replace 0 with 1 to avoid division by 0)
                nsmpl = torch.sum(out["band_mask"], dim=-1)
                nsmpl[nsmpl == 0] = 1
                out["nsmpl"] = nsmpl

            out["wave"] = out["spectra_sup_data"][:,0][...,None] # [bsz,nsmpl,1]

        elif self.wave_source == "trans":
            # trans wave are not batched, we sample at every step
            if self.perform_integration:
                if self.kwargs["trans_sample_method"] == "hardcode":
                    out["wave"] = self.trans_dataset.get_hdcd_wave()
                    out["wave"] = out["wave"][None,:,None].tile(batch_size,1,1)
                    out["trans"] = self.trans_dataset.get_hdcd_trans()
                    out["nsmpl"] = self.trans_dataset.get_hdcd_nsmpl()
                else:
                    out["wave"], out["trans"], out["wave_smpl_ids"], out["nsmpl"] = \
                        self.trans_dataset.sample_wave(
                            batch_size,
                            self.kwargs["train_num_wave_samples"],
                            use_all_wave=self.use_all_wave
                        )

                if self.mode == "main_infer" and "recon_synthetic_band" in self.kwargs["tasks"]:
                    assert(self.use_all_wave) # only in inferrence
                    nsmpl = out["trans"].shape[1]
                    out["trans"] = torch.cat((out["trans"], torch.ones(1,nsmpl)), dim=0)
                    out["nsmpl"] = torch.cat((out["nsmpl"], torch.tensor([nsmpl])), dim=0)
            else:
                out["wave"] = torch.FloatTensor(self.get_full_wave())
                out["wave"] = out["wave"][None,:,None].tile(batch_size,1,1)
        else:
            raise ValueError("Unsupported wave data source.")

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
            assert self.mode == "codebook_pretrain"
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
        """ Get redshift values for main train redshift semi-supervision.
        """
        ids = out["spectra_id_map"]
        if not self.kwargs["train_spectra_pixels_only"]:
            bin_map = out["spectra_bin_map"]
            ids = ids[bin_map]
        out["spectra_semi_sup_redshift"] = self.fits_dataset.get_spectra_pixel_redshift(ids)
        del out["spectra_id_map"]

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

        if "wave_data" in self.requested_fields:
            self.get_wave_data(len(idx), out)

        if "spectra_data" in self.requested_fields:
            self.get_spectra_data(out)

        if "redshift_data" in self.requested_fields:
           self.get_redshift_data(out)

        # print_shape(out)
        if self.transform is not None:
            out = self.transform(out)
        return out

    ############
    # Utilities
    ############

    def get_pixel_ids_one_patch(self, r, c, neighbour_size=1):
        return self.fits_dataset.get_pixel_ids_one_patch(r, c, neighbour_size)

    def interpolate_spectra(self, f, spectra, spectra_masks):
        return self.spectra_dataset.interpolate_spectra(f, spectra, spectra_masks)

    def integrate_spectra_over_transmission(
            self, spectra, spectra_masks=None, all_wave=True, interpolate=True
    ):
        return self.trans_dataset.integrate(
            spectra, spectra_masks=spectra_masks, all_wave=all_wave, interpolate=interpolate
        )

    def restore_evaluate_tiles(self, recon_pixels, **re_args):
        """ Restore flattened image, save locally and/or calculate metrics.
            @Return:
               metrics(_z): metrics of current model [n_metrics,1,ntiles,nbands]
        """
        return self.fits_dataset.restore_evaluate_tiles(recon_pixels, **re_args)

    def plot_spectrum(self, spectra_dir, name, flux_norm_cho,
                      gt_wave, gt_fluxes, recon_wave, recon_fluxes,
                      is_codebook=False, spectra_ids=None,
                      save_spectra=False, save_spectra_together=False,
                      gt_masks=None, recon_masks=None,
                      clip=False, spectra_clipped=False
    ):
        self.spectra_dataset.plot_spectrum(
            spectra_dir, name, flux_norm_cho,
            gt_wave, gt_fluxes, recon_wave, recon_fluxes,
            is_codebook=is_codebook,
            save_spectra=save_spectra,
            save_spectra_together=save_spectra_together,
            spectra_ids=spectra_ids,
            gt_masks=gt_masks, recon_masks=recon_masks,
            clip=clip, spectra_clipped=spectra_clipped,
        )
