
    def get_spectra_data_infer(self, out, idx):
        """ Load spectra data. Since we only do one inferrence task at a time,
              we load exactly one of the three possible sets of data.
            Note that gt spectra data with or without supervision are mutually exclusive.
        """
        if "gt_spectra_data_w_supervision" in self.requested_fields:
            out["full_wave"] = self.trans_dataset.get_full_norm_wave()
            spectra_coords = self.spectra_dataset.get_gt_spectra_coords()[:,None]

            out["coords"] = spectra_coords
            out["spectra"] = self.spectra_dataset.get_gt_spectra() # CPU np
            out["gt_spectra_wave"] = self.spectra_dataset.get_gt_spectra_wave()
            out["recon_spectra_wave"] = self.spectra_dataset.get_gt_spectra_wave()

        elif "gt_spectra_data" in self.requested_fields: # @infer only
            assert("coords" not in out)
            spectra_coords = self.spectra_dataset.get_spectra_coords()
            out["coords"] = spectra_coords
            out["spectra"] = self.spectra_dataset.get_gt_spectra()
            out["gt_spectra_wave"] = self.spectra_dataset.get_gt_spectra_wave()
            out["recon_spectra_wave"] = self.spectra_dataset.get_recon_spectra_wave()

        elif "dummy_spectra_data" in self.requested_fields: # @infer only
            out["coords"] = self.spectra_dataset.get_dummy_spectra_coords()
            out["recon_spectra_wave"] = self.spectra_dataset.get_dummy_spectra_wave()

        #print(out["spectra_coords"].shape, out["spectra"].shape, out["spectra_wave"].shape)
