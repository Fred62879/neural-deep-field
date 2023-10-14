# (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch.nn as nn

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

    def __init__(self, nef: BaseNeuralField):

        super().__init__()

        self.nef: BaseNeuralField = nef

    def set_batch_reduction_order(self, order="qtz_first"):
        self.nef.set_batch_reduction_order(order=order)

    def set_bayesian_redshift_logits_calculation(self, loss, mask, gt_spectra):
        self.nef.set_bayesian_redshift_logits_calculation(loss, mask, gt_spectra)

    def forward(self, *args, **kwargs):
        return self.nef(*args, **kwargs)
