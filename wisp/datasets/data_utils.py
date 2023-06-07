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

def create_patch_uid(tract, patch):
    patch = patch.replace(",", "")
    return f"{tract}{patch}"

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

def get_mgrid_np(num_rows, num_cols, lo=-1, hi=1, dim=2, indexing='ij', flat=True):
    #def get_mgrid_np(self, sidelen, lo=-1, hi=1, dim=2, indexing='ij', flat=True):
    """ Generates a flattened grid of (x,y,...) coords in [-1,1] (numpy version).
    """
    x = np.linspace(lo, hi, num=num_cols)
    y = np.linspace(lo, hi, num=num_rows)
    mgrid = np.stack(np.meshgrid(x, y, indexing=indexing), axis=-1)

    if flat: mgrid = mgrid.reshape(-1,dim) # [sidelen**2,dim]
    return mgrid

def get_mgrid_tensor(self, sidelen, lo=-1, hi=1, dim=2, flat=True):
    """ Generates a flattened grid of (x,y,...) coords in [-1,1] (Tensor version).
    """
    tensors = tuple(dim * [torch.linspace(lo, hi, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    if flat: mgrid = mgrid.reshape(-1, dim)
    return mgrid

def add_dummy_dim(coords, **kwargs):
    if kwargs["coords_encode_method"] == "grid" and kwargs["grid_dim"] == 3:
        num_coords = coords.shape[0]
        class_name = coords.__class__.__name__
        # print(coords.shape, class_name)
        # if type(coords).__module__ == "torch":
        coords_2d = coords

        if class_name == "Tensor":
            coords = torch.zeros((num_coords, 3))
        elif class_name == "ndarray":
            coords = np.zeros((num_coords, 3))
        else:
            raise ValueError("Unknown collection class")
        coords[...,:2] = coords_2d
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
