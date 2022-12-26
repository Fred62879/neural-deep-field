
import torch
import torch.nn as nn

from torch.nn.functional import one_hot
from wisp.utils.numerical import find_closest_tensor


class LatentQuantizer(nn.Module):
    ''' Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
        Input:  latent [...,latent_dim]
        Output: quantized [...,latent_dim]
    '''
    def __init__(self, qtz_latent_dim, qtz_num_embed, qtz_beta, qtz_calculate_loss, qtz_seed, **kwargs):
        super(LatentQuantizer, self).__init__()

        self.beta = qtz_beta
        self.num_embed = qtz_num_embed
        self.latent_dim = qtz_latent_dim
        self.calculate_loss = qtz_calculate_loss

        self.init_codebook(qtz_seed)

    def init_codebook(self, seed):
        torch.manual_seed(seed)
        self.codebook = nn.Embedding(self.latent_dim, self.num_embed)
        self.codebook.weight.data.uniform_(
            -1.0 / self.latent_dim, 1.0 / self.latent_dim)
        #self.codebook.weight.data *= 50
        #self.codebook = torch.zeros(self.latent_dim, self.num_embed).to('cuda:0')
        #self.codebook.uniform_(-1/2,1/2)

    def quantize(self, z):
        # flatten input [...,]
        assert(z.shape[-1] == self.latent_dim)
        z_shape = z.shape
        z_f = z.view(-1,self.latent_dim)

        min_embed_ids = find_closest_tensor(z_f, self.codebook.weight) # [bsz]

        # replace each z with closest embedding
        encodings = one_hot(min_embed_ids, self.num_embed) # [n,num_embed]
        encodings = encodings.type(z.dtype)
        z_q = torch.matmul(encodings, self.codebook.weight.T).view(z_shape)
        return z_q, min_embed_ids

    def partial_loss(self, z, z_q):
        codebook_loss = torch.mean((z_q.detach() - z)**2) + \
            torch.mean((z_q - z.detach())**2) * self.beta
        return codebook_loss

    def forward(self, input):
        z, scaler, redshift = input["latents"], input["scaler"], input["redshift"]

        z_q, min_embed_ids = self.quantize(z)
        if self.calculate_loss:
            loss = self.partial_loss(z, z_q)
        else: loss = None

        # straight-through estimator
        z_q = z + (z_q - z).detach()
        return dict(latents=z_q, scaler=scaler, redshift=redshift,
                    codebook_loss=loss, min_embed_ids=min_embed_ids)

"""
class LatentQuantizer(nn.Module):
    ''' Encoder with vector quantization
          dim_in is by default 2 (2 dim coordinate)
          dim_out is the latent dimension
          if we generate scaler, dim_out will be incremented by 1
    '''
    def __init__(self, latent_dim, num_embed, mlp_cho, mlp_args, output_scaler,
                 output_redshift, calculate_loss, quantize, beta, cdbk_seed):

        super(LatentQuantizer, self).__init__()

        self.quantize = quantize
        self.output_scaler = output_scaler
        self.output_redshift = output_redshift

        self.encoder = MLP_All(mlp_cho, mlp_args)
        if self.quantize:
            self.quantization = Vector_Quantizer \
                (latent_dim, num_embed, beta, calculate_loss, cdbk_seed)

    ''' @Param
          coords    [...,in_dim]
        @Return
          latents_q [...,latent_dim]: quantized latents
          redshift  [...,1]
          scaler    [...,1]
    '''
    def forward(self, input):
        (coords, _) = input
        latents = self.encoder([coords, None])

        if self.output_scaler:
            if self.output_redshift:
                scaler = latents[...,-2:-1]
                redshift = latents[...,-1:]
                latents = latents[...,:-2]
            else:
                redshift = None
                scaler = latents[...,-1:]
                latents = latents[...,:-1]
        else: redshift, scaler = None, None

        if self.quantize:
            latents_q, cdbk_loss, embed_ids = self.quantization([latents, None])
        else: latents_q, cdbk_loss, embed_ids = latents, None, None
        return (latents_q, latents, scaler, redshift, cdbk_loss, embed_ids)
"""
