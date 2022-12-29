
import os
import torch
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from pathlib import Path
from os.path import exists, join
from wisp.inferrers import BaseInferrer
from wisp.utils.common import forward, load_model_weights


class AstroInferrer(BaseInferrer):
    """ Inferrer class for astro dataset.

        Inferrence using all/a few selected saved models.
            Possible inferrence tasks:                          _
              Reconstruct multiband observations.                | infer w/ full model
              Reconstruct observations under flat transmission.  |          all coords
              Plot pixel embedding map.                          | (all pixels need to
              Plot embedding latent distribution (up to 3 dims)._|  be inferred)
              Reconstruct spectra.                              _| partial model / selected coords
              Reconstruct spectra using codebook.               _| modified model / all coords

            The first four tasks are based on the original pipeline
              and needs to evaluate all coordinates.
            Spectra reconstruction doesn't need integration
              and only evaluate certain selected coordinates.
            Codebook spectra reconstruction omits the scaler generation
              part and evalute all coordinates.

        If infer with hyperspectral net, assume using all
          available lambda values without sampling.
    """

    def __init__(self, pipelines, dataset, device, extra_args, info=None):

        super().__init__(pipelines, dataset, device, extra_args, info=info)

        if "full" in pipelines:
            self.full_pipeline = pipelines["full"]
        if "partial" in pipelines:
            self.partial_pipeline = pipelines["partial"]
        if "modified" in pipelines:
            self.modified_pipeline = pipelines["modified"]

        self.set_log_path()
        self.select_models()
        self.summarize_inferrence_tasks()
        self.generate_inferrence_funcs()

    #############
    # Initializations
    #############

    def set_log_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose: log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["model_dir","recon_dir","metric_dir", "spectrum_dir",
                 "cdbk_spectrum_dir", "embd_map_dir","latent_dir",
                 "latent_embd_dir"],
                ["models","recons","metrics","spectrum","cdbk_spectrum",
                 "embd_map","latent","latent_embd_dir"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

    def select_models(self):
        self.selected_model_fnames = os.listdir(self.model_dir)
        self.num_models = len(self.selected_model_fnames)
        self.selected_model_fnames.sort()
        if self.verbose: log.info(f"selected {self.num_models} models")

    def summarize_inferrence_tasks(self):
        """ Group similar inferrence tasks (tasks using same dataset and same model) together.
        """
        tasks = set(self.extra_args["tasks"])

        self.quantize_latent = "quantize_latent" in self.tasks

        # infer all coords using original model
        self.recon_img = "recon_img" in tasks
        self.recon_HSI = "recon_HSI" in self.tasks
        self.recon_flat_trans = "recon_flat_trans" in tasks
        self.plot_embd_map = "plot_embd_map_during_recon" in tasks \
            and self.space_dim == 3 and self.quantize_latent
        self.plot_embd_latent_distrib = "plot_embd_latent_distrib" in tasks \
            and self.space_dim == 3 and self.quantize_latent

        # infer all coords using modified model
        self.recon_cdbk_spectra = "recon_cdbk_spectra" in tasks \
            and self.space_dim == 3 and self.quantize_latent

        # infer selected coords using partial model
        self.recon_gt_spectra = "recon_gt_spectra" in tasks and self.space_dim == 3
        self.recon_dummy_spectra = "recon_dummy_spectra" in tasks and self.space_dim == 3
        self.recon_gt_spectra_w_supervision = "recon_gt_spectra_w_supervision" in tasks and self.space_dim == 3
        assert(not (self.recon_gt_spectra and self.recon_gt_spectra_w_supervision) )

        # keep only tasks required to perform
        self.group_tasks = []
        if self.recon_img or self.recon_flat_trans or self.recon_cdbk_spectra \
           or self.plot_embd_map or self.plot_embd_latent_distrib:
            self.group_tasks.append("infer_all_coords_full_model")
        if self.recon_cdbk_spectra:
            self.group_tasks.append("infer_all_coords_modified_model")
        if self.recon_dummy_spectra or self.recon_gt_spectra or self.recon_gt_spectra_w_supervision:
            self.group_tasks.append("infer_selected_coords_partial_model")

        # set all grouped tasks to False, only required tasks will be toggled afterwards
        self.infer_all_coords_full_model = False
        self.infer_all_coords_modified_model = False
        self.infer_selected_coords_partial_model = False

        log.info(f"inferrence group tasks: {self.group_tasks}.")

    def generate_inferrence_funcs(self):
        self.infer_funcs = {}
        for group_task in self.group_tasks:
            if group_task == "infer_all_coords_full_model":
                self.infer_funcs[group_task] = [
                    self.pre_inferrence_all_coords_full_model,
                    self.post_inferrence_all_coords_full_model,
                    self.pre_checkpoint_all_coords_full_model,
                    self.run_checkpoint_all_coords_full_model,
                    self.post_checkpoint_all_coords_full_model ]

            elif group_task == "infer_selected_coords_partial_model":
                self.infer_funcs[group_task] = [
                    self.pre_inferrence_selected_coords_partial_model,
                    self.post_inferrence_selected_coords_partial_model,
                    self.pre_checkpoint_selected_coords_partial_model,
                    self.run_checkpoint_selected_coords_partial_model,
                    self.post_checkpoint_selected_coords_partial_model ]

            elif group_task == "infer_all_coords_modified_model":
                self.infer_funcs[group_task] = [
                    self.pre_inferrence_all_coords_modified_model,
                    self.post_inferrence_all_coords_modified_model,
                    self.pre_checkpoint_all_coords_modified_model,
                    self.run_checkpoint_all_coords_modified_model,
                    self.post_checkpoint_all_coords_modified_model ]

            else: raise Exception("Unrecgonized group inferrence task.")

    #############
    # Inferrence
    #############

    def pre_inferrence_all_coords_full_model(self):
        num_fits = self.dataset.get_num_fits()
        num_coords = self.dataset.get_num_coords()
        self.num_batches = int(np.ceil(num_coords / self.batch_size))

        if self.recon_img:
            self.metric_options = self.extra_args["metric_options"]
            self.num_metrics = len(self.metric_options)
            self.calculate_metrics = self.recon_img and \
                not self.recon_flat_trans and self.metric_options is not None

            if self.calculate_metrics:
                self.metrics = np.zeros((self.num_metrics, 0, num_fits, self.num_bands))
                self.metrics_zscale = np.zeros((self.num_metrics, 0, num_fits, self.num_bands))
                self.metric_fnames = [ join(self.metric_dir, f"{option}.npy")
                                       for option in self.metric_options ]
                self.metric_fnames_z = [ join(self.metric_dir, f"{option}_zscale.npy")
                                         for option in self.metric_options ]

    def post_inferrence_all_coords_full_model(self):
        if self.calculate_metrics:
            [ np.save(self.metric_fnames[i], self.metrics[i])
              for i in range(self.num_metrics) ]
            [ np.save(self.metric_fnames_z[i], self.metrics_zscale[i])
              for i in range(self.num_metrics) ]
            log.info(f"metrics: {np.round(self.metrics[:,-1,0], 3)}")
            log.info(f"zscale metrics: {np.round(self.metrics_zscale[:,-1,0], 3)}")

    def pre_inferrence_selected_coords_partial_model(self):
        if self.recon_gt_spectra or self.recon_gt_spectra_w_supervision:
            self.spectra_task = "gt"
        else: self.spectra_task = "dummy"
        num_coords = self.dataset.get_num_spectra_coords()

        if self.extra_args["infer_spectra_individually"]:
            self.num_batches = num_coords
        else: self.num_batches = int(np.ceil(num_coords / self.batch_size))

    def post_inferrence_selected_coords_partial_model(self):
        pass

    def pre_inferrence_all_coords_modified_model(self):
        pass

    def post_inferrence_all_coords_modified_model(self):
        pass

    #############
    # Infer with checkpoint
    #############

    def pre_checkpoint_all_coords_full_model(self, model_id):
        self.reset_dataloader()

        if self.recon_img:
            self.to_HDU_now = self.extra_args["to_HDU"] and model_id == self.num_models
            self.recon_HSI_now = self.recon_HSI and model_id == self.num_models
            self.recon_flat_trans_now = self.recon_flat_trans and model_id == self.num_models
            #if self.recon_flat_trans_now: self.num_bands = 1
            self.recon_pixels = []

        if self.plot_embd_map:
            self.embd_ids = []

    def run_checkpoint_all_coords_full_model(self, model_id, checkpoint):
        self.infer_all_coords(model_id, checkpoint)

    def post_checkpoint_all_coords_full_model(self, model_id):
        if self.recon_img:
            re_args = {
                "fname": str(model_id),
                "dir": self.recon_dir,
                "metric_options": self.metric_options,
                "verbose": self.verbose,
                "to_HDU": self.to_HDU_now,
                "recon_HSI": self.recon_HSI_now,
                "calculate_metrics": self.calculate_metrics,
                "recon_norm": self.extra_args["recon_norm"],
                "recon_flat_trans": self.recon_flat_trans_now
            }
            cur_metrics, cur_metrics_zscale = self.dataset.restore_evaluate_tiles(
                self.recon_pixels, **re_args)

            if self.calculate_metrics:
                # add metrics for current checkpoint
                self.metrics = np.concatenate((self.metrics, cur_metrics), axis=1)
                self.metrics_zscale = np.concatenate((self.metrics_zscale, cur_metrics_zscale), axis=1)

        if self.plot_embd_map:
            self.plot_embd_map()

    def pre_checkpoint_selected_coords_partial_model(self, model_id):
        self.reset_dataloader()
        self.recon_spectra = []
        self.gt_spectra = []
        self.gt_spectra_wave = []
        self.recon_spectra_wave = []

    def run_checkpoint_selected_coords_partial_model(self, model_id, checkpoint):
        self.infer_spectra(model_id, checkpoint)

    def post_checkpoint_selected_coords_partial_model(self, model_id):
        if self.extra_args["plot_spectrum_with_trans"]:
            self.colors = self.extra_args["plot_colors"]
            self.labels = self.extra_args["plot_labels"]
            self.styles = self.extra_args["plot_styles"]
        self.plot_spectrum(model_id)
        #self.calculate_recon_spectra_pixel_values()

    def pre_checkpoint_all_coords_modified_model(self, model_id):
        self.reset_dataloader()
        self.recon_cdbk_spectra(model_id, checkpoint)

    def run_checkpoint_all_coords_modified_model(self, model_id, checkpoint):
        self.recon_cdbk_spectra(model_id, checkpoint)

    def post_checkpoint_all_coords_modified_model(self, model_id):
        pass

    #############
    # Infer all coords
    #############

    def infer_all_coords(self, model_id, checkpoint):
        """ Using given checkpoint, reconstruct, if specified:
              multi-band image - np, to_HDU (FITS), recon_HSI (hyperspectral)
              flat-trans image,
              pixel embedding map
        """
        # load model checkpoint into model
        load_model_weights(self.full_pipeline, checkpoint)
        self.full_pipeline.eval()

        # run one epoch for inferrence
        for i in range(self.num_batches):
            data = self.next_batch()
            with torch.no_grad():
                ret = forward(self, self.full_pipeline, data, self.quantize_latent,
                              self.plot_embd_map, self.recon_gt_spectra_w_supervision)
            if self.recon_img: self.recon_pixels.extend(ret["intensity"])
            if self.plot_embd_map: self.embd_ids.extend(ret["embd_ids"])

    # Plot embd map
    def plot_embd_map(self):
        if self.plot_embd_map: embd_map_fname = self.embd_map_fname

        for fits_id in self.fits_ids:
            metrics = np.zeros((len(self.metric_options), 0, self.num_bands))
            metrics_zscale = np.zeros((len(self.metric_options), 0, self.num_bands))

            # plot residue map between recon and gt
            if self.plot_residue_heatmap:
                gt_fname = class_obj.gt_imgs_fnames[id]

                resid = class_obj.gt_imgs[fits_id] - recon
                fname = join(recon_dir, f"resid_{model_id}.png")
                heat_all(resid, fn=fname)

        #if self.plot_embd_map:
        sz = int(np.sqrt(num_pixels))
        embd_ids = np.array(embd_ids).reshape((sz, sz))
        plt.imshow(embd_ids, cmap="gray")
        plt.savefig(embd_map_fn)
        plt.close()
        np.save(embd_map_fn, embd_ids)
        plot_embd_map(embd_ids, embd_map_fn)

    #############
    # Infer spectra
    #############

    def infer_spectra(self, model_id, checkpoint):
        load_model_weights(self.partial_pipeline, checkpoint)
        self.partial_pipeline.eval()

        for i in range(self.num_batches):
            data = self.next_batch()

            with torch.no_grad():
                spectra = forward(
                    self, self.partial_pipeline, data, self.quantize_latent,
                    self.plot_embd_map, self.recon_gt_spectra_w_supervision)["spectra"]

            if spectra.ndim == 3: # bandwise
                spectra = spectra.flatten(1,2) # [bsz,nsmpl]

            (lo, hi) = data["trusted_wave_bound_id"]
            spectra = spectra[...,lo:hi]

            self.recon_spectra.extend(spectra)
            self.recon_wave.extend(data["gt_recon_wave"][0])

            if spectra_task == "gt":
                print(data["gt_spectra_wave"][0].shape)
                self.gt_spectra.extend(data["spectra"][0])
                self.gt_spectra_wave.extend(data["gt_spectra_wave"][0])

        print('****', len(self.recon_spectra_wave))

    def plot_spectrum(self, model_id):
        recon_spectra = torch.stack(self.recon_spectra).detach().cpu().numpy()

        for i, cur_spectra in enumerate(recon_spectra):
            if self.extra_args["plot_spectrum_average"]:
                cur_spectra = np.mean(cur_spectra, axis=0)
            cur_spectra /= np.max(cur_spectra)

            if self.extra_args["plot_spectrum_with_trans"]:
                for j, cur_trans in enumerate(self.trans):
                    plt.plot(self.full_wave, cur_trans, color=self.colors[j],
                             label=self.labels[j], linestyle=self.styles[j])

            plt.plot(self.recon_spectra_wave[i], cur_spectra,
                     color="black", label="spectrum")

            if self.recon_gt_spectra or self.recon_gt_spectra_w_supervision:
                cur_gt_spectra = self.gt_spectra[i]
                cur_gt_spectra_wave = self.gt_spectra_wave[i]
                cur_gt_spectra /= np.max(cur_gt_spectra)
                plt.plot(cur_gt_spectra_wave, cur_gt_spectra, color="blue", label="gt")

            fname = join(self.spectra_dir, f"model_{model_id}_spectra_{i}")
            plt.savefig(fname);plt.close()

    def calculate_recon_spectra_pixel_values(self):
        for fits_id in self.fits_ids:
            # calculate spectrum pixel recon value
            if args.plot_spectrum:
                print("recon spectrum pixel", recon[args.spectrum_pos])

    #############
    # Recon cdbk spectra
    #############

    def recon_cdbk_spectra(self, checkpoint):
        pass

    #############
    # Helpers
    #############

    def _configure_dataset(self):
        """ Configure dataset (batched fields and len) for inferrence.
        """
        fields = []

        if self.infer_all_coords_full_model or self.infer_all_coords_modified_model:
            state = "fits"
            fields = ['coords']
            if self.recon_img: fields.append('pixels')
            length = self.dataset.get_num_coords()

        elif self.infer_selected_coords_partial_model:
            state = "spectra"

            if self.recon_gt_spectra_w_supervision or self.recon_gt_spectra:
                fields= ["coords","spectra","gt_spectra_wave","gt_recon_wave"]
                length = self.dataset.get_num_gt_spectra_coords()

            elif self.recon_dummy_spectra:
                fields = ["coords","dummy_recon_wave"]
                length = self.dataset.get_num_dummy_spectra_coords()

        else: raise Exception("Unrecgonized group inferrence task.")

        if self.space_dim == 3: fields.extend(['trans_data'])

        self.dataset.set_dataset_state(state)
        self.dataset.set_dataset_length(length)
        self.dataset.set_dataset_fields(fields)
