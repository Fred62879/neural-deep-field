
import torch
import torch.nn as nn

from torch.nn.functional import one_hot

from wisp.utils.common import get_input_latents_dim
from wisp.utils.numerical import find_closest_tensor

from wisp.models.decoders import BasicDecoder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class


class QuantizedDecoder(nn.Module):
    """ Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
    """
    def __init__(self, calculate_loss, **kwargs):
        super(QuantizedDecoder, self).__init__()

        self.kwargs = kwargs
        self.beta = kwargs["qtz_beta"]
        self.num_embed = kwargs["qtz_num_embed"]
        self.latent_dim = kwargs["qtz_latent_dim"]
        self.input_dim = get_input_latents_dim(**kwargs)

        self.calculate_loss = calculate_loss
        self.output_scaler = kwargs["generate_scaler"]
        self.output_redshift = kwargs["generate_redshift"]

        self.init_decoder()
        if self.output_scaler:
            self.init_scaler_decoder()
        if self.output_redshift:
            self.init_redshift_decoder()
            self.redshift_adjust = nn.ReLU(inplace=True)
        self.init_codebook(kwargs["qtz_seed"])

    def init_scaler_decoder(self):
        self.scaler_decoder = BasicDecoder(
            self.input_dim, 1,
            get_activation_class(self.kwargs["scaler_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["scaler_decod_layer_type"]),
            num_layers=self.kwargs["scaler_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["scaler_decod_hidden_dim"], skip=[])

    def init_redshift_decoder(self):
        self.redshift_decoder = BasicDecoder(
            self.input_dim, 1,
            get_activation_class(self.kwargs["redshift_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["redshift_decod_layer_type"]),
            num_layers=self.kwargs["redshift_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["redshift_decod_hidden_dim"], skip=[])

    def init_decoder(self):
        self.decoder = BasicDecoder(
            self.input_dim, self.latent_dim,
            get_activation_class(self.kwargs["qtz_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["qtz_decod_layer_type"]),
            num_layers=self.kwargs["qtz_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["qtz_decod_hidden_dim"], skip=[])

    def init_codebook(self, seed):
        torch.manual_seed(seed)
        self.codebook = nn.Embedding(self.latent_dim, self.num_embed)
        self.codebook.weight.data.uniform_(
            -1.0 / self.latent_dim, 1.0 / self.latent_dim)
        self.codebook.weight.data /= 10
        # self.codebook = torch.zeros(self.latent_dim, self.num_embed).to('cuda:0')
        # self.codebook.uniform_(-1/2,1/2)
        # print(torch.min(self.codebook.weight.data), torch.max(self.codebook.weight.data))

    def quantize(self, z):
        # flatten input [...,]
        # assert(z.shape[-1] == self.latent_dim)
        z_shape = z.shape
        z_f = z.view(-1,self.latent_dim)

        min_embed_ids = find_closest_tensor(z_f, self.codebook.weight) # [bsz]

        # replace each z with closest embedding
        encodings = one_hot(min_embed_ids, self.num_embed) # [n,num_embed]
        encodings = encodings.type(z.dtype)
        z_q = torch.matmul(encodings, self.codebook.weight.T).view(z_shape)
        return z_q, min_embed_ids

    def partial_loss(self, z, z_q):
        #codebook_loss = torch.mean((z_q.detach() - z)**2) + \
        #    torch.mean((z_q - z.detach())**2) * self.beta
        codebook_loss = torch.mean((z_q.detach() - z)**2)
        return codebook_loss

    def forward(self, z, ret):
        """ Quantize latent variables
            @Param
              z: raw 2D coordinate or embedding of 2D coordinate [batch_size,1,dim]
        """
        #timer = PerfTimer(activate=self.kwargs["activate_timer"], show_memory=False)

        if self.kwargs["print_shape"]: print('qtz ', z.shape)

        # decode high-dim features into low dim latents
        #timer.check("quantization decode")

        if self.output_scaler:
            scaler = self.scaler_decoder(z[:,0])[...,0]
        else: scaler = None

        if self.output_redshift:
            redshift = self.redshift_decoder(z[:,0])[...,0]
            redshift = self.redshift_adjust(redshift)
        else: redshift = None

        z = self.decoder(z)

        ret["scaler"] = scaler
        ret["redshift"] = redshift

        # quantize latents
        #timer.check("quantization quantize")
        z_q, min_embed_ids = self.quantize(z)

        if self.calculate_loss:
            #timer.check("quantization calculate loss")
            ret["codebook_loss"] = self.partial_loss(z, z_q)

        # straight-through estimator
        z_q = z + (z_q - z).detach()
        if self.kwargs["print_shape"]: print('qtz, z_q ', z_q.shape)

        ret["latents"] = z
        ret["min_embed_ids"] = min_embed_ids
        return z_q
