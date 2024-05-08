
import torch
import numpy as np

from os.path import exists

from typing import Callable
from itertools import product
from torch.utils.data import Dataset

from wisp.datasets.fits_data import FitsData
from wisp.datasets.mask_data import MaskData
from wisp.datasets.trans_data import TransData
from wisp.datasets.spectra_data import SpectraData
from wisp.utils.common import print_shape, get_bin_ids, \
    create_gt_redshift_bin_masks, get_redshift_range
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
                "idx","selected_ids","wave_data","spectra_data","redshift_data","model_data",
                "gt_redshift_bin_ids","gt_redshift_bin_masks","global_restframe_spectra_loss"
            }
        else:
            self.unbatched_fields = set()

    def init(self):
        """ Initializes the dataset.
            Load only needed data based on given tasks (in kwargs).
        """
        self.data = {}
        self.trans_dataset = TransData(self.device, **self.kwargs)
        self.spectra_dataset = SpectraData(self.trans_dataset, self.device, **self.kwargs)
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
        self.spectra_source = "sup"
        self.wave_sample_method = "NA"

        self.sample_wave = False
        self.use_all_wave = True
        self.infer_selected = False
        self.perform_integration = True
        self.use_predefined_wave_range = False

    ############
    # Setters
    ############

    def set_mode(self, mode):
        """
        Possible modes: ["codebook_pretrain","sanity_check","generalization",
                         "main_train","codebook_pretrain_infer","sanity_check_infer",
                         "generalization_infer","main_infer","test"]
        """
        self.mode = mode

    def set_patch(self, patch_obj):
        """ Set current patch to process.
        """
        self.patch_obj = patch_obj

    def set_length(self, length):
        self.dataset_length = length

    def set_fields(self, fields):
        self.requested_fields = set(fields)

    # def set_wave_range(self, lo, hi):
    #     self.wave_range = (lo, hi)

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

    def set_spectra_source(self, spectra_source):
        self.spectra_source = spectra_source

    def set_hardcode_data(self, field, data):
        self.data[field] = data

    def set_num_wave_samples(self, num_samples):
        self.num_wave_samples = num_samples

    def set_wave_sample_method(self, method="uniform"):
        """ Set lambda sampling method.
            @Choices
              uniform: uniformly at random (most cases)
              uniform_non_random: get every several wave (for pretrain infer)
        """
        self.wave_sample_method = method

    def set_num_redshift_bins(self, num_bins):
        self.num_redshift_bins = num_bins

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

    def get_fields(self):
        return list(self.requested_fields)

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

    def get_emitted_wave(self):
        return self.spectra_dataset.get_emitted_wave()

    def get_emitted_wave_mask(self):
        return self.spectra_dataset.get_emitted_wave_mask()


    def get_spectra_source(self):
        return self.spectra_source

    def get_spectra_masks(self, idx=None):
        if self.spectra_source == "sup":
            return self.spectra_dataset.get_supervision_mask(idx)
        if self.spectra_source == "val":
            return self.spectra_dataset.get_validation_mask(idx)
        if self.spectra_source == "test":
            return self.spectra_dataset.get_test_mask(idx)

    def get_spectra_ivar_reliable(self, idx=None):
        if self.spectra_source == "sup":
            return self.spectra_dataset.get_supervision_ivar_reliable(idx)
        if self.spectra_source == "val":
            return self.spectra_dataset.get_validation_ivar_reliable(idx)
        if self.spectra_source == "test":
            return self.spectra_dataset.get_test_ivar_reliable(idx)

    def get_spectra_sup_bounds(self, idx=None):
        if self.spectra_source == "sup":
            return self.spectra_dataset.get_supervision_sup_bound(idx)
        if self.spectra_source == "val":
            return self.spectra_dataset.get_validation_sup_bound(idx)
        if self.spectra_source == "test":
            return self.spectra_dataset.get_test_sup_bound(idx)

    def get_spectra_coords(self, idx=None):
        if self.spectra_source == "sup":
            return self.spectra_dataset.get_supervision_coords(idx)
        if self.spectra_source == "val":
            return self.spectra_dataset.get_validation_coords(idx)
        if self.spectra_source == "test":
            return self.spectra_dataset.get_test_coords(idx)

    def get_spectra_pixels(self, idx=None):
        if self.spectra_source == "sup":
            return self.spectra_dataset.get_supervision_pixels(idx)
        if self.spectra_source == "val":
            return self.spectra_dataset.get_validation_pixels(idx)
        if self.spectra_source == "test":
            return self.spectra_dataset.get_test_pixels(idx)

    def get_spectra_redshift(self, idx=None):
        if self.spectra_source == "sup":
            return self.spectra_dataset.get_supervision_redshift(idx)
        if self.spectra_source == "val":
            a = self.spectra_dataset.get_validation_redshift(idx)
            print('getter', a.shape, torch.min(a), torch.max(a))
            return a
        if self.spectra_source == "test":
            return self.spectra_dataset.get_test_redshift(idx)

    def get_spectra_source_data(self, idx=None):
        if self.spectra_source == "sup":
            return self.spectra_dataset.get_supervision_spectra(idx)
        if self.spectra_source == "val":
            return self.spectra_dataset.get_validation_spectra(idx)
        if self.spectra_source == "test":
            return self.spectra_dataset.get_test_spectra(idx)

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

    def get_supervision_spectra_ids(self):
        return self.spectra_dataset.get_supervision_spectra_ids()

    def get_validation_spectra_ids(self):
        return self.spectra_dataset.get_validation_spectra_ids()

    def get_sanity_check_spectra_ids(self):
        return self.spectra_dataset.get_sanity_check_spectra_ids()

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
        self.get_unbatched_data(idx, out)

        # print_shape(out)
        if self.transform is not None:
            out = self.transform(out)
        return out

    ############
    # Helpers
    ############

    def get_batched_data(self, field, idx):
        if field == "coords":
            if self.coords_source == "fits":
                data = self.fits_dataset.get_coords()
            else: data = self.data[self.coords_source]

        elif field == "pixels":
            data = self.fits_dataset.get_pixels()
        elif field == "weights":
            data = self.fits_dataset.get_weights()
        elif field == "spectra_id_map":
            data = self.fits_dataset.get_spectra_id_map()
        elif field == "spectra_bin_map":
            data = self.fits_dataset.get_spectra_bin_map()

        elif field == "spectra_masks":
            data = self.get_spectra_masks()
        elif field == "spectra_pixels":
            data = self.get_spectra_pixels()
        elif field == "spectra_redshift":
            data = self.get_spectra_redshift()
        elif field == "spectra_sup_bounds":
            data = self.get_spectra_sup_bounds()
        elif field == "spectra_ivar_reliable":
            data = self.get_spectra_ivar_reliable()
        elif field == "spectra_source_data":
            data = self.get_spectra_source_data()

        # elif field == "masks":
        #     data = self.mask_dataset.get_mask()
        elif field in self.data:
            data = self.data[field]
        else:
            raise ValueError(f"Unrecognized data field: {field}.")

        # print('*', field, data.shape, idx)
        data = self.index_selected_data(data, idx)
        return data

    def get_unbatched_data(self, idx, out):
        if "idx" in self.requested_fields:
            out["idx"] = idx
        if "selected_ids" in self.requested_fields:
            out["selected_ids"] = self.data["selected_ids"]
        # self.get_debug_data(out)
        # if "model_data" in self.requested_fields:
        #     self.get_model_data(out)
        # if self.kwargs["plot_logits_for_gt_bin"]:
        #     self.get_gt_redshift_bin_ids(out)
        if self.kwargs["add_redshift_logit_bias"]:
            self.get_init_redshift_logit_bias(out)
        if "wave_data" in self.requested_fields:
            self.get_wave_data(len(idx), out)
        if "spectra_data" in self.requested_fields:
            self.get_spectra_data(out)
        if "redshift_data" in self.requested_fields:
            self.get_redshift_data(out)
        if "gt_redshift_bin_ids" in self.requested_fields:
            self.get_gt_redshift_bin_ids(out)
        if "gt_redshift_bin_masks" in self.requested_fields:
            self.get_gt_redshift_bin_masks(out)
        if "global_restframe_spectra_loss" in self.requested_fields:
            out["global_restframe_spectra_loss"] = \
                self.data["global_restframe_spectra_loss"]

    def index_selected_data(self, data, idx):
        """ Index data with both selected_ids and given idx
              (for selective spectra inferrence only)
            @Param
               selected_ids: select from source data (filter index)
               idx: dataset index (batch index)
        """
        if self.infer_selected:
            assert self.mode == "codebook_pretrain_infer" or \
                self.mode == "sanity_check_infer" or \
                self.mode == "generalization_infer"
            assert "selected_ids" in self.data
            data = data[self.data["selected_ids"]]
        return data[idx]

    # def get_debug_data(self, out):
    #     if self.kwargs["plot_logits_for_gt_bin"]:
    #         self.get_gt_redshift_bin_ids(out)
    #     if self.kwargs["add_redshift_logit_bias"]:
    #         self.get_init_redshift_logit_bias(out)

    # def get_model_data(self, out):
    #     if "scaler_latents" in self.data:
    #         out["scaler_latents"] = self.data["scaler_latents"]
    #     if "redshift_latents" in self.data:
    #         out["redshift_latents"] = self.data["redshift_latents"]

    def get_wave_data(self, batch_size, out):
        """ Get wave (lambda and transmission) data depending on data source.
        """
        out["wave_range"] = self.get_wave_range()

        if self.wave_source == "full_spectra":
            # for codebook spectra recon (according to emitted wave)
            # doesn't consider effect of redshift on each spectra
            bsz = out["coords"].shape[0]
            wave = self.get_emitted_wave()
            masks = self.get_emitted_wave_mask()
            out["wave"] = wave[None,:,None].tile(bsz,1,1)
            out["spectra_masks"] = masks[None,:].tile(bsz,1)

        elif self.wave_source == "spectra":
            # spectra_sup_data: [bsz,4+2*nbands,nsmpl]
            #  (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))

            if self.sample_wave:
                # sample from spectra data (wave, flux, ivar, and interpolated trans)
                # sample_ids [bsz,nsmpl,2]
                out["spectra_source_data"], sample_ids = batch_sample_torch(
                    out["spectra_source_data"], self.num_wave_samples,
                    sample_method=self.wave_sample_method,
                    sup_bounds=out["spectra_sup_bounds"],
                    keep_sample_ids=True)

                wave = out["spectra_source_data"][:,0]

                out["spectra_masks"] = batch_sample_torch(
                    out["spectra_masks"], self.num_wave_samples,
                    sample_method=self.wave_sample_method,
                    sup_bounds=out["spectra_sup_bounds"],
                    sample_ids=sample_ids)

            if self.perform_integration:
                # out["trans_mask"] = out["spectra_source_data"][:,3]              # [bsz,nsmpl]
                out["trans"] = out["spectra_source_data"][:,4:4+self.num_bands]    # [bsz,nbands,nsmpl]
                out["band_mask"] = out["spectra_source_data"][:,4+self.num_bands:] # [bsz,nbands,nsmpl]
                # num of sample within each band (replace 0 with 1 to avoid division by 0)
                nsmpl = torch.sum(out["band_mask"], dim=-1)
                nsmpl[nsmpl == 0] = 1
                out["nsmpl"] = nsmpl

            out["wave"] = out["spectra_source_data"][:,0][...,None] # [bsz,nsmpl,1]

            if self.mode == "codebook_pretrain" and (
                    self.kwargs["regularize_within_codebook_spectra"] or \
                    self.kwargs["regularize_across_codebook_spectra"]
            ):
                out["emitted_wave"] = self.get_emitted_wave()
                out["emitted_wave_masks"] = self.get_emitted_wave_mask()

        elif self.wave_source == "trans":
            # trans wave are not batched, we sample at every step
            if self.perform_integration:
                if self.kwargs["trans_sample_method"] == "hardcode":
                    out["wave"] = self.trans_dataset.get_hdcd_wave()
                    out["wave"] = out["wave"][None,:,None].tile(batch_size,1,1)
                    out["trans"] = self.trans_dataset.get_hdcd_trans()
                    out["nsmpl"] = self.trans_dataset.get_hdcd_nsmpl()
                else:
                    nsmpls = -1 if self.use_all_wave else self.num_wave_samples
                    out["wave"], out["trans"], out["wave_smpl_ids"], out["nsmpl"] = \
                        self.trans_dataset.sample_wave(
                            batch_size, nsmpls, use_all_wave=self.use_all_wave)

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
        """ Get unbatched spectra data during main train for spectra supervision only.
            If we train on large amount of spectra, use batched data instead.
        """
        assert self.mode == "main_train" and self.kwargs["spectra_supervision"]

        if self.kwargs["main_train_with_pretrained_latents"]:
            spectra_coords = self.data["pretrained_latents"]
        else:
            if self.kwargs["normalize_coords"]:
                spectra_coords = self.patch_obj.get_spectra_normed_img_coords()
            else: spectra_coords = self.patch_obj.get_spectra_img_coords()
            spectra_coords = torch.FloatTensor(spectra_coords)[:,None]

        if "coords" in out:
            out["coords"] = torch.cat((out["coords"], spectra_coords), dim=0)
        else: out["coords"] = spectra_coords

        out["sup_spectra_data"] = torch.FloatTensor(self.patch_obj.get_spectra_data())
        out["sup_spectra_masks"] = torch.BoolTensor(self.patch_obj.get_spectra_pixel_masks())

        if not self.kwargs["spectra_supervision_use_all_wave"]:
            assert self.kwargs["pretrain_wave_sample_method"] == "uniform"
            out["sup_spectra_data"], sample_ids = batch_sample_torch(
                out["sup_spectra_data"],
                self.kwargs["spectra_supervision_num_wave_samples"],
                wave_sample_method=self.kwargs["pretrain_wave_sample_method"],
                keep_sample_ids=True
            )
            out["sup_spectra_masks"] = batch_sample_torch(
                out["sup_spectra_masks"],
                self.kwargs["spectra_supervision_num_wave_samples"],
                wave_sample_method=self.kwargs["pretrain_wave_sample_method"],
                sample_ids=sample_ids
            )

        out["sup_spectra_wave"] = out["sup_spectra_data"][:,0][...,None] # [bsz,nsmpl,1]

        # the first #num_supervision_spectra are gt coords for supervision
        # the others are forwarded only for spectrum plotting
        out["num_sup_spectra"] = len(spectra_coords)

    def get_redshift_data(self, out):
        """ From currently sampled pixels, pick spectra pixels and get corresponding redshift.
            Called during main train redshift semi-supervision only.
            @Param
              spectra_id_map: label each spectra pixel with corresponding global spectra id.
              spectra_bin_map: binary map that masks (sets as 0) all non-spectra pixels.
        """
        assert not self.kwargs["train_spectra_pixels_only"]
        ids = out["spectra_id_map"]
        bin_map = out["spectra_bin_map"]
        ids = ids[bin_map]
        out["spectra_semi_sup_redshift"] = self.fits_dataset.get_spectra_pixel_redshift(ids)
        del out["spectra_id_map"]

    def get_gt_redshift_bin_ids(self, out):
        print(out["spectra_redshift"])
        out["gt_redshift_bin_ids"] = self.create_gt_redshift_bin_ids(
            spectra_redshift=out["spectra_redshift"])

    def get_gt_redshift_bin_masks(self, out):
        _, out["gt_redshift_bin_masks"] = \
            self.create_gt_redshift_bin_masks(
                self.num_redshift_bins,
                spectra_redshift=out["spectra_redshift"])

    ############
    # Debug data
    ############

    def get_init_redshift_logit_bias(self, out):
        bsz = out["spectra_redshift"].shape[0]
        (lo, hi) = get_redshift_range(**self.kwargs)
        n_bins = int( np.rint((hi - lo) / self.kwargs["redshift_bin_width"]))
        # ids = np.array(
        #     [get_bin_id(self.kwargs["redshift_lo"], self.kwargs["redshift_bin_width"], val)
        #      for val in out["spectra_redshift"]])
        # ids = np.rint(ids).astype(int)
        # init_probs = np.zeros((bsz, n_bins)).astype(np.float32)
        # pos = np.arange(bsz)
        # ids = np.concatenate((pos[None,:],ids[None,:]),axis=0)
        # init_probs[ ids[0,:], ids[1:,] ] = 1
        ids = get_bin_ids(
            lo, self.kwargs["redshift_bin_width"],
            out["spectra_redshift"], add_batched_dim=True
        )
        init_probs[ids[0], ids[1]] = 1
        out["init_redshift_prob"] = init_probs

    ############
    # Utilities
    ############

    def create_gt_redshift_bin_ids(self, spectra_redshift=None):
        if spectra_redshift is None:
            spectra_redshift = self.get_spectra_redshift()
        (lo, hi) = get_redshift_range(**self.kwargs)
        print(spectra_redshift.shape, torch.min(spectra_redshift), torch.max(spectra_redshift))
        print(lo, hi)
        gt_bin_ids = get_bin_ids(
            lo, self.kwargs["redshift_bin_width"],
            spectra_redshift.numpy(), add_batched_dim=True)
        print(gt_bin_ids)
        assert 0
        return torch.tensor(gt_bin_ids)

    def create_wrong_redshift_bin_ids(self, gt_bin_masks):
        """
        @Param: gt bin masks [bsz,nbins]
        @Return: wrong bin ids [2,bsz,nbins-1]
        """
        bsz, nbins = gt_bin_masks.shape
        wrong_bin_ids = torch.tensor(
            list(product(range(bsz), range(nbins))), dtype=gt_bin_masks.dtype
        ).view(bsz,nbins,2)
        gt_bin_masks = gt_bin_masks[...,None].tile(1,1,2)
        wrong_bin_ids = wrong_bin_ids[~gt_bin_masks].view(bsz,nbins-1,2).permute(2,0,1)
        return wrong_bin_ids

    def create_gt_redshift_bin_masks(self, num_bins, spectra_redshift=None, to_bool=True):
        """ Get mask with 0 in indices of wrong bins
        """
        gt_bin_ids = self.create_gt_redshift_bin_ids(spectra_redshift=spectra_redshift)
        gt_bin_masks = create_gt_redshift_bin_masks(gt_bin_ids, num_bins)
        if to_bool: gt_bin_masks = gt_bin_masks.astype(bool)
        else: gt_bin_masks = gt_bin_masks.astype(np.long)
        gt_bin_masks = torch.tensor(gt_bin_masks)
        return gt_bin_ids, gt_bin_masks

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

    def plot_spectrum(self, spectra_dir, name, flux_norm_cho, redshift,
                      gt_wave, ivar, gt_fluxes, recon_wave, recon_fluxes,
                      recon_fluxes2=None,
                      recon_losses2=None,
                      recon_fluxes3=None,
                      recon_losses3=None,
                      lambdawise_losses=None,
                      lambdawise_weights=None,
                      colors=None,
                      titles=None,
                      is_codebook=False,
                      save_spectra=False,
                      calculate_metrics=True,
                      save_spectra_together=False,
                      gt_masks=None, recon_masks=None,
                      clip=False, spectra_clipped=False
    ):
        return self.spectra_dataset.plot_spectrum(
            spectra_dir, name, flux_norm_cho, redshift,
            gt_wave, ivar, gt_fluxes, recon_wave, recon_fluxes,
            recon_fluxes2=recon_fluxes2,
            recon_losses2=recon_losses2,
            recon_fluxes3=recon_fluxes3,
            recon_losses3=recon_losses3,
            lambdawise_losses=lambdawise_losses,
            lambdawise_weights=lambdawise_weights,
            colors=colors,
            titles=titles,
            is_codebook=is_codebook,
            save_spectra=save_spectra,
            calculate_metrics=calculate_metrics,
            save_spectra_together=save_spectra_together,
            gt_masks=gt_masks, recon_masks=recon_masks,
            clip=clip, spectra_clipped=spectra_clipped)
