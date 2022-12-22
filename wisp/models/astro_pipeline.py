# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch.nn as nn

from wisp.models.hypers import LatentQuantizer, Integrator
from wisp.models.nefs import BaseNeuralField


class AstroPipeline(nn.Module):
    """Base class for implementing neural field pipelines.

    Pipelines consist of several components:

        - Neural fields (``self.nef``) which take coordinates as input and outputs signals.
          These usually consist of several optional components:

            - A feature grid (``self.nef.grid``)
              Sometimes also known as 'hybrid representations'.
            - An acceleration structure (``self.nef.grid.blas``) which can be used to accelerate spatial queries.
            - A decoder (``self.net.decoder``) which can take the features (or coordinates, or embeddings) and covert it to signals.

        - A forward map (``self.tracer``) which is a function which will invoke the pipeline in
          some outer loop. Usually this consists of renderers which will output a RenderBuffer object.

    The 'Pipeline' classes are responsible for holding and orchestrating these components.
    """

    def __init__(self, nef: BaseNeuralField, quantz: LatentQuantizer, inte: Integrator):
        """ Initialize the Pipeline.

        Args:
            nef (nn.Module): Neural fields module.
        """
        super().__init__()

        self.nef: BaseNeuralField = nef
        self.quantz: LatentQuantizer = qantz
        self.inte: Integrator = inte

    def forward(self, *args, **kwargs):
        """ The forward function will use the tracer (the forward model) if one is available.
            Otherwise, it'll execute the neural field.
        """
        ret = self.nef(*args, **kwargs)
        if self.quantz is not None:
            ret = self.quantz(ret)
        if self.inte is not None:
            ret = self.inte(ret)
        return ret
