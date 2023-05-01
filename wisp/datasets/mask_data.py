
import torch
import random
import numpy as np
import logging as log

from pathlib import Path
from os.path import exists, join
from collections import defaultdict
from wisp.datasets.data_utils import create_uid


class MaskData:
    """ Data class for masks.
        Only used when doing inpainting
    """
    def __init__(self, fits_obj, dataset_path, device, **kwargs):

        self.spatial_inpaint = kwargs["inpaint_cho"] == "spatial_inpaint"
        self.spectral_inpaint = kwargs["inpaint_cho"] == "spectral_inpaint"
        assert not (self.spatial_inpaint and self.spectral_inpaint)

        if not "train" in kwargs["tasks"] or \
           not (self.spatial_inpaint or self.spectral_inpaint):
            return

        self.kwargs = kwargs
        self.fits_obj = fits_obj
        self.verbose = kwargs["verbose"]
        self.plot_masked_gt = kwargs["plot_masked_gt"]

        self.num_bands = kwargs["num_bands"]
        self.mask_mode = kwargs["mask_mode"]
        self.inpaint_cho = kwargs["inpaint_cho"]
        self.inpaint_bands = kwargs["inpaint_bands"]
        self.inpaint_sample_ratio = kwargs["inpaint_sample_ratio"]

        self.set_path(dataset_path)
        self.data = defaultdict(lambda: [])
        self.load_mask()

    def set_path(self, dataset_path):
        input_path = join(dataset_path, "input")
        mask_path = join(input_path, "masks")

        # mask path creation
        if self.mask_mode == "region":
            mask_str = "region_" + str(self.kwargs["m_start_r"]) + "_" \
                + str(self.kwargs["m_start_c"]) + "_" \
                + str(self.kwargs["mask_size"])
        else: mask_str = "_" + str(float(100 * self.inpaint_sample_ratio))

        suffix = create_uid(self.fits_obj, **self.kwargs)
        if self.inpaint_cho == "spectral_inpaint":
            self.mask_fname = join(mask_path, suffix + mask_str + ".npy")
            self.masked_pixel_ids_fname = join(mask_path, suffix + mask_str + "_masked_ids.npy")
        else:
            self.mask_fname, self.masked_pixel_id_fname = None, None

        # create path
        for path in [mask_path]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_mask(self, flat=True, to_bool=True, to_tensor=True):
        """ Load (or generate) mask dependeing on config for spectral inpainting.
            If train bands and inpaint bands form a smaller set of
              the band of the current mask file, then we load only and
              slice the corresponding dimension from the larger mask.
        """
        print(self.mask_fname, self.masked_pixel_ids_fname)
        if exists(self.mask_fname) and exists(self.masked_pixel_ids_fname):
            if self.verbose:
                log.info(f"loading spectral mask from {self.mask_fname}")
            mask = np.load(self.mask_fname)
            masked_pixel_ids = np.load(self.masked_pixel_ids_fname)
        else:
            assert(len(self.kwargs["filters"]) ==
                   len(self.kwargs["train_bands"]) + len(self.kwargs["inpaint_bands"]))
            mask, masked_pixel_ids = self.create_mask_all_patches()
            mask = np.concatenate(mask) # [npixels,nbands]
            masked_pixel_ids = np.concatenate(masked_pixel_ids) # [n_masked_pixels]

            # print(mask.shape, masked_pixel_ids.shape)
            np.save(self.mask_fname, mask)
            np.save(self.masked_pixel_ids_fname, masked_pixel_ids)

        num_smpl_pixls = [np.count_nonzero(mask[:,i]) for i in range(mask.shape[1])]
        log.info(f"Sampled pixels for each band of spectral mask: {num_smpl_pixls}")

        # slice mask, leave inpaint bands only
        mask = mask[:,self.inpaint_bands] # [npixels,num_inpaint_bands]
        if to_bool:   mask = (mask == 1) # convert to boolean array
        if not flat:  mask = mask.reshape((args.img_size, args.img_size, -1))
        if to_tensor: mask = torch.tensor(mask)

        if self.spatial_inpaint:
            # dont support spatial inpainting currently
            assert 0
            self.spatial_masking()

        self.data["mask"] = mask # [npixels,nbands]
        self.data["masked_pixel_ids"] = masked_pixel_ids # [n_masked_pixels]

    #############
    # Mask creation
    #############

    def create_mask_one_band(self, n, ratio, seed):
        """ Generate mask for one image band.
            First 100*ratio% pixels has value 1 (unmasked).
            @Return
              mask: binary mask (0-mask, 1-unmask) [n,]
              masked_pixel_ids: local id of masked pixels [n,]
                (local id is pixel id within current patch,
                 later on we convert local id to global id,
                 which is pixel id within all selected patches)
        """
        ids = np.arange(n)
        random.seed(seed)
        random.shuffle(ids)
        offset = int(ratio*n)
        mask = np.zeros(n)
        mask[ids[:offset]] = 1
        return mask, ids[-offset:]

    def create_mask_one_patch(self, npixels, npixels_acc):
        """ Generate mask for one multi-band patch.
            Mask only pixels in inpaint_bands.
        """
        if self.mask_mode == "rand_diff" or self.mask_mode == "rand_same":
            # mask different pixels in different bands
            if self.verbose: log.info("mask diff pixels in diff bands")

            ratio = self.inpaint_sample_ratio
            mask, masked_pixel_ids = np.ones((npixels, self.num_bands)), []

            for i in self.inpaint_bands:
                seed = self.kwargs["mask_seed"]
                if self.mask_mode == "rand_diff": seed += i

                mask[:,i], cur_band_masked_pixel_ids = self.create_mask_one_band(
                    npixels, ratio, seed)
                masked_pixel_ids.append(cur_band_masked_pixel_ids)

            # [npixels,nbands], [nbands,num_masked_pixels]
            masked_pixel_ids = np.array(masked_pixel_ids)

        elif self.mask_mode == "region":
            # mask a rectangular area
            assert(False) # NOT TESTED
            if self.verbose: log.info("= mask region")

            rs = np.arange(self.kwargs["m_start_r"],
                           self.kwargs["m_start_r"] + self.kwargs["msize"])
            cs = np.arange(self.kwargs["m_start_c"],
                           self.kwargs["m_start_c"] + self.kwarsg["msize"])

            grid = np.stack(np.meshgrid(*tuple([rs,cs]), indexing="ij"), axis=-1).reshape((-1,2))
            m_ids = np.array(list(map(lambda p: p[0]*nr + p[1], grid)))
            nm_ids = np.array(list(set(ids)-set(m_ids))) # id of pixels used for training

        else:
            raise ValueError("Unsupported mask mode.")

        masked_pixel_ids += npixels_acc
        return mask, masked_pixel_ids

    def create_mask_all_patches(self):
        mask, masked_pixel_ids = [], []

        acc_npixels = 0
        num_rows = self.fits_obj.get_num_rows()
        num_cols = self.fits_obj.get_num_cols()
        fits_uids = self.fits_obj.get_fits_uids()

        if self.plot_masked_gt:
            gt_img_fnames = self.fits_obj.get_gt_img_fnames()

        for id, fits_uid in enumerate(fits_uids):
            num_rows, num_cols = num_rows[fits_uid], num_cols[fits_uid]
            npixels = num_rows * num_cols
            _mask, _masked_pixel_ids = self.create_mask_one_patch(npixels, acc_npixels)

            if self.plot_masked_gt:
                self.plot_masked_gt_img(_mask, gt_img_fnames[fits_uid])

            mask.append(_mask)
            masked_pixel_ids.append(_masked_pixel_ids)
            acc_npixels += npixels

        return mask, masked_pixel_ids

    def spatial_masking(self):
        mask = load_mask(args)[...,0]
        pixls, coords = pixls[mask], coords[mask]
        if self.load_weights:
            assert(weights is not None)
            weights = weights[mask]
        log.info("Spatial mask: total num train pixls: {}, with {} per epoch".
                 format(len(mask), int(len(mask)*args.train_ratio)))
        return pixls, coords, weights

    #############
    # Getters
    #############

    def get_mask(self, idx=None):
        assert self.kwargs["inpaint_cho"] == "spectral_inpaint"
        if idx is None:
            return self.data["mask"][idx]
        return self.data["mask"]

        # self.relative_train_bands = self.kwargs["relative_train_bands"]
        # self.relative_inpaint_bands = self.kwargs["relative_inpaint_bands"]

        # self.num_masked_pixls = self.masked_pixl_ids.shape[0]

        # # iv) get ids of cutout pixels
        # if self.save_cutout:
        #     self.cutout_pixl_ids = utils.generate_cutout_pixl_ids\
        #         (self.cutout_pos, self.fits_cutout_size, self.img_size)
        # else: self.cutout_pixl_ids = None
        # return

    ############
    # Utilities
    ############

    def plot_masked_gt_img(self, mask, in_fname):
        out_fname = in_fname + "masked"
        gt_img = np.load(in_fname + ".npy")
        img_shape = gt_img.shape
        print(gt_img.shape, mask.shape)
        masked_gt = gt_img * (mask.T.reshape(img_shape))
        np.save(out_fname, masked_gt)
