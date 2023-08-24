# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import re
import torch
import collections
import numpy as np

from wisp.core import Rays
from functools import lru_cache
from os.path import join, exists
from torch._six import string_classes
from torch.utils.data._utils.collate import default_convert

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def get_neighbourhood_center_pixel_id(neighbour_size):
    """ Get id of center pixel within a neighbourhood (defined in PatchData).
    """
    offset = neighbour_size // 2
    return offset * (neighbour_size + 1)

@lru_cache
def patch_exists(path, tract, patch):
    """ Check whether the given patch file exists.
        @Param
          tract: image tract, e.g. `9812`
          patch: image patch, e.g. `1,2`
    """
    fname = create_patch_fname(tract, patch, "HSC-G") # a random band
    fname = join(path, fname)
    return exists(fname)

def get_wave_range_fname(**kwargs):
    fname = join(
        kwargs["dataset_path"], "input/wave", kwargs["wave_range_fname"])
    return fname

def get_coords_range_fname(**kwargs):
    _, img_data_path = set_input_path(
        kwargs["dataset_path"], kwargs["sensor_collection_name"]
    )
    if kwargs["use_full_patch"]: patch = "full_patch"
    else: patch = kwargs["patch_selection_cho"]
    coords_cho = kwargs["train_coords_cho"]
    norm_cho = "_normed" if kwargs["normalize_coords"] else ""
    fname = join(
        img_data_path, f"coords_range_{patch}{norm_cho}_{coords_cho}.npy")
    return fname

def create_patch_fname(tract, patch, band, megau=False, weights=False):
    """ Create image patch file name.
        @Param
          tract: image tract, e.g. `9812`
          patch: image patch, e.g. `1,3` (converted to `1%2C3` or 1c3)
          band:  filter name, e.g. `HSC-G`, `NB111`
    """
    if megau:
        patch = patch.replace(",","c")
        if weights:
            return "Mega-" + band + "_" + self.tract + "_" + upatch + ".weight.fits"
        return "Mega-" + band + "_" + tract + "_" + upatch + ".fits"

    patch = patch.replace(",","%2C")
    return "calexp-" + band + "-" + tract + "-" + patch + ".fits"

def set_input_path(dataset_path, sensor_name):
    input_path = join(dataset_path, "input")
    input_patch_path = join(input_path, "input_fits")
    img_data_path = join(input_path, sensor_name, "img_data")
    return input_patch_path, img_data_path

def create_selected_patches_uid(fits_obj, **kwargs):
    """ Form suffix that uniquely identifies the currently selected group of
        patches with the corresponding cropping parameters, if any.
    """
    suffix = ""
    if kwargs["use_full_patch"]:
        for patch_uid in fits_obj.patch_uids:
            suffix += f"_{patch_uid}"
    else:
        for (patch_uid, num_rows, num_cols, (r,c)) in zip(
                fits_obj.patch_uids, fits_obj.patch_cutout_num_rows,
                fits_obj.patch_cutout_num_cols, fits_obj.patch_cutout_start_pos):
            suffix += f"_{patch_uid}_{num_rows}_{num_cols}_{r}_{c}"

    return suffix

def clip_data_to_ref_wave_range(input_data, ref_wave, wave_range=None, wave_range_id=None):
    if wave_range_id is None:
        assert wave_range is not None
        id_lo, id_hi = get_bound_id(wave_range, ref_wave, within_bound=True)
        wave_range_id = (id_lo, id_hi)
    else: id_lo, id_hi = wave_range_id
    clipped_data = input_data[id_lo:id_hi+1]
    return clipped_data, wave_range_id

def batch_sample_torch(data, nsmpl, sample_method="uniform", distrib=None,
                       sample_ids=None, keep_sample_ids=False, ordered=True
):
    """ Batched sampling from given data.
        We sample the same nsmpl out of n samples for all dims in the middle.
        @Param
          data: [bsz,...,n]
          sample_ids: pre-defined ids to sample
        @Return
          ret:  [bsz,...,nsmpl]
          sample_ids: [nsmpl,2]
    """
    sp = data.shape
    if sample_ids is not None:
        assert list(sample_ids.shape) == [sp[0], nsmpl, 2]
    else:
        if sample_method == "uniform":
            sample_ids = torch.zeros(sp[0], nsmpl).uniform_(0, sp[-1]).to(torch.long)
        elif sample_method == "importance":
            sample_ids = torch.multinomial(distrib, nsmpl, replacement=True)
        else:
            raise ValueError("Unsupported sampling method.")

        if ordered:
            sample_ids, _ = torch.sort(sample_ids, dim=-1)
        row_ids = torch.repeat_interleave(torch.arange(sp[0]), nsmpl).view(sp[0],nsmpl)
        sample_ids = torch.cat((row_ids[...,None],sample_ids[...,None]), dim=-1)

    # print(sample_ids[0], sample_ids.shape)
    ret = data[sample_ids[...,0],...,sample_ids[...,1]]
    mid_axis = list(np.arange(2,len(sp)))
    reorded_axis = [0] + mid_axis + [1]
    ret = ret.permute(reorded_axis)

    if keep_sample_ids: return ret, sample_ids
    return ret

def get_bound_id(wave_bound, source_wave, within_bound=True):
    """ Get id of lambda values in source wave that bounds or is bounded by given wave_bound
        if `within_bound`
            source_wave[id_lo] >= wave_lo
            source_wave[id_hi] <= wave_hi
        else
            source_wave[id_lo] <= wave_lo
            source_wave[id_hi] >= wave_hi
    """
    if type(source_wave).__module__ == "torch":
        source_wave = source_wave.numpy()

    wave_lo, wave_hi = wave_bound
    # wave_hi = int(min(wave_hi, int(max(source_wave))))

    if within_bound:
        if wave_lo <= min(source_wave): id_lo = 0
        else: id_lo = np.argmax((source_wave >= wave_lo))

        if wave_hi >= max(source_wave): id_hi = len(source_wave) - 1
        else: id_hi = np.argmax((source_wave > wave_hi)) - 1

        assert(source_wave[id_lo] >= wave_lo and source_wave[id_hi] <= wave_hi)
    else:
        if wave_lo <= min(source_wave): id_lo = 0
        else: id_lo = np.argmax((source_wave > wave_lo)) - 1

        if wave_hi >= max(source_wave): id_hi = len(source_wave) - 1
        else: id_hi = np.argmax((source_wave >= wave_hi))

        assert(source_wave[id_lo] <= wave_lo and source_wave[id_hi] >= wave_hi)

    return [id_lo, id_hi]

def get_mgrid_np(num_rows, num_cols, rlo=-1, rhi=1, clo=-1, chi=1,
                 dim=2, indexing='ij', flat=True
):
    """ Generates a mesh grid of coords (numpy version).
    """
    x = np.linspace(clo, chi, num=num_cols)
    y = np.linspace(rlo, rhi, num=num_rows)
    mgrid = np.stack(np.meshgrid(y, x, indexing=indexing), axis=-1)
    if flat: mgrid = mgrid.reshape(-1,dim) # [sidelen**2,dim]
    return mgrid

# def get_mgrid_np(num_rows, num_cols, lo=-1, hi=1, dim=2, indexing='ij', flat=True):
#     """ Generates a flattened grid of (x,y,...) coords in [-1,1] (numpy version).
#     """
#     x = np.linspace(lo, hi, num=num_cols)
#     y = np.linspace(lo, hi, num=num_rows)
#     mgrid = np.stack(np.meshgrid(x, y, indexing=indexing), axis=-1)
#     if flat: mgrid = mgrid.reshape(-1,dim) # [sidelen**2,dim]
#     return mgrid

def get_mgrid_tensor(self, sidelen, lo=-1, hi=1, dim=2, flat=True):
    """ Generates a flattened grid of (x,y,...) coords in [-1,1] (Tensor version).
    """
    tensors = tuple(dim * [torch.linspace(lo, hi, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    if flat: mgrid = mgrid.reshape(-1, dim)
    return mgrid

def add_dummy_dim(coords, **kwargs):
    if kwargs["coords_encode_method"] == "grid" and kwargs["grid_dim"] == 3:
        sp = list(coords.shape[:-1])
        sp.append(3)
        class_name = coords.__class__.__name__
        # print(coords.shape, class_name)
        # if type(coords).__module__ == "torch":
        coords_2d = coords

        if class_name == "Tensor":
            coords = torch.zeros(sp)
        elif class_name == "ndarray":
            coords = np.zeros(sp)
        else:
            raise ValueError("Unknown collection class")
        coords[...,:2] = coords_2d
    coords = torch.FloatTensor(coords)
    return coords

def default_collate(batch):
    r"""
        Function that extends torch.utils.data._utils.collate.default_collate
        to support Rays.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]
    elif isinstance(elem, Rays):
        return Rays.cat(batch)

    raise TypeError(default_collate_err_msg_format.format(elem_type))
