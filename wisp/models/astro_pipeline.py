
import torch.nn as nn

class AstroPipeline(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_addup_latents(self):
        return self.model.get_addup_latents()

    def set_latents(self, latents):
        self.model.set_latents(latents)

    def set_base_latents(self, latents):
        self.model.set_base_latents(latents)

    def set_addup_latents(self, latents):
        self.model.set_addup_latents(latents)

    def set_gt_bin_latents(self, latents):
        self.model.set_gt_bin_latents(latents)

    def set_wrong_bin_latents(self, latents):
        self.model.set_wrong_bin_latents(latents)

    def set_redshift_latents(self, redshift_latents):
        self.model.set_redshift_latents(redshift_latents)

    def set_batch_reduction_order(self, order="qtz_first"):
        self.model.set_batch_reduction_order(order=order)

    def set_bayesian_redshift_logits_calculation(self, loss, mask, gt_spectra):
        self.model.set_bayesian_redshift_logits_calculation(loss, mask, gt_spectra)

    def toggle_sample_bins(self, sample: bool):
        self.model.toggle_sample_bins(sample)

    def add_latents(self):
        self.model.add_latents()

    def combine_latents_all_bins(self, gt_bin_ids, wrong_bin_ids, redshift_bins_mask):
        self.model.combine_latents_all_bins(gt_bin_ids, wrong_bin_ids, redshift_bins_mask)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
