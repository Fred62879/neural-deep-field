
from wisp.models.nefs import BaseNeuralField

class HardcodeNef(BaseNeuralField):
    """ Model for returning hardcoded value.
    """
    def init_embedder(self):
        pass

    def init_grid(self):
        pass

    def init_decoder(self):
        pass

    def get_nef_type(self):
        return 'hardcode'

    def register_forward_functions(self):
        channels = ["latents"]
        self._register_forward_function( self.hardcode, channels )

    def hardcode(self, coords, pidx=None, lod_idx=None):
        """ Output given latents without any modifications.
            @Params:
              coords (torch.FloatTensor): tensor of shape [1, batch, num_samples, 2/3]
              pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                       Unused in the current implementation.
              lod_idx (int): index into active_lods. If None, will use the maximum LOD.
                             Currently interpolation doesn't use this.
            @Return
              {"indensity": torch.FloatTensor }:
                - Output intensity tensor of shape [batch, num_samples, 3]
        """
        return dict(latents=coords)
