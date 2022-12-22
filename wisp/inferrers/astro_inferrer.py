
import os
import torch
import numpy as np
import logging as log

from pathlib import Path
from os.path import exists, join
from wisp.inferrers import BaseInferrer
from wisp.utils.plot import plot_horizontally
from wisp.utils.fits_data import recon_img_and_evaluate
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

        self.full_pipeline = pipelines[0]

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
        self.spectra_supervision = self.space_dim == 3 \
            and self.extra_args['spectra_supervision']
        self.quantize_latent = self.extra_args["quantize_latent"] and \
            (self.extra_args["use_ngp"] or self.extra_args["encode"])

        # infer all coords using original model
        self.recon_img = "recon_img" in self.tasks
        self.recon_flat_trans = "recon_flat_trans" in self.tasks
        self.plot_embd_map = "plot_embd_map_during_recon" in self.tasks \
            and self.space_dim == 3 and self.quantize_latent
        self.plot_embd_latent_distrib = "plot_embd_latent_distrib" in self.tasks \
            and self.space_dim == 3 and self.quantize_latent

        # infer all coords using modified model
        self.recon_cdbk_spectra = "recon_cdbk_spectra" in self.tasks \
            and self.space_dim == 3 and self.quantize_latent

        # infer selected coords using partial model
        self.recon_spectra = "recon_spectra" in self.tasks and self.space_dim == 3

        # keep only tasks required to perform
        self.group_tasks = []
        if self.recon_img or self.recon_flat_trans or self.recon_cdbk_spectra \
           or self.plot_embd_map or self.plot_embd_latent_distrib:
            self.group_tasks.append("infer_all_coords_full_model")
        if self.recon_cdbk_spectra:
            self.group_tasks.append("infer_all_coords_modified_model")
        if self.recon_spectra:
            self.group_tasks.append("infer_selected_coords_partial_model")

        # set all grouped tasks to False, only required tasks will be toggled afterwards
        self.infer_all_coords_full_model = False
        self.infer_all_coords_modified_model = False
        self.infer_selected_coords_partial_model = False

        log.info(f"inferrence group tasks: {self.group_tasks}.")

    def configure_dataset(self):
        """ Configure dataset for inferrence. """
        if self.infer_all_coords_full_model or self.infer_all_coords_modified_model:
            fields = ['coords']
            if self.recon_img: fields.append('pixels')
            length = self.dataset.get_num_coords()

        elif self.infer_selected_coords_partial_model:
            fields = ['spectra_coords','gt_spectra']
            length = len(self.dataset.get_num_spectra_coords())

        else: raise Exception("Unrecgonized group inferrence task.")

        if self.space_dim == 3: fields.extend(['wave','trans'])
        self.dataset.set_dataset_length(length)
        self.dataset.set_dataset_fields(fields)

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

    def pre_inferrence_selected_coords_partial_model(self):
        pass

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
            self.recon_HSI_now = "recon_HSI" in self.tasks and model_id == self.num_models
            self.recon_flat_trans_now = self.recon_flat_trans and model_id == self.num_models
            if self.recon_flat_trans_now: self.num_bands = 1
            self.recon_pixels = []

        if self.plot_embd_map:
            self.embd_ids = []

    def run_checkpoint_all_coords_full_model(self, model_id, checkpoint):
        self.infer_all_coords(model_id, checkpoint)

    def post_checkpoint_all_coords_full_model(self, model_id):
        if self.recon_img:
            kwargs = {
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
            cur_metrics, cur_metrics_zscale = recon_img_and_evaluate(
                self.recon_pixels, self.dataset, **kwargs)

            if self.calculate_metrics:
                # add metrics for current checkpoint
                self.metrics = np.concatenate((self.metrics, cur_metrics), axis=1)
                self.metrics_zscale = np.concatenate((self.metrics_zscale, cur_metrics_zscale), axis=1)

        if self.plot_embd_map:
            self.plot_embd_map()

    def pre_checkpoint_selected_coords_partial_model(self, model_id):
        self.reset_dataloader()
        self.recon_spectra(model_id, checkpoint)
        self.calculate_recon_spectra_pixel_values()

    def run_checkpoint_selected_coords_partial_model(self, model_id, checkpoint):
        self.recon_spectra(model_id, checkpoint)
        self.calculate_recon_spectra_pixel_values()

    def post_checkpoint_selected_coords_partial_model(self, model_id):
        pass

    def pre_checkpoint_all_coords_modified_model(self, model_id):
        self.reset_dataloader()
        self.recon_cdbk_spectra(model_id, checkpoint)

    def run_checkpoint_all_coords_modified_model(self, model_id, checkpoint):
        self.recon_cdbk_spectra(model_id, checkpoint)

    def post_checkpoint_all_coords_modified_model(self, model_id):
        pass

    #############
    # Task I:Infer all coords
    #############

    def infer_all_coords(self, model_id, checkpoint):
        """ Using given checkpoint, reconstruct, if specified:
              multi-band image - np, to_HDU (FITS), recon_HSI (hyperspectral)
              flat-trans image,
              pixel embedding map
        """
        # load model checkpoint into model
        load_model_weights(self.full_pipeline, checkpoint)
        # if model_id == self.num_models - 1:
        #     for k,v in self.full_pipeline.state_dict().items():
        #         if "embedder" in k and "bias" not in k:
        #             print(v.T)

        self.full_pipeline.eval()

        # run one epoch for inferrence
        for i in range(self.num_batches):
            data = self.next_batch()
            with torch.no_grad():
                ret = forward(self, self.full_pipeline, data, self.quantize_latent,
                              self.plot_embd_map, self.spectra_supervision)

            if self.recon_img: self.recon_pixels.extend( ret["intensity"] )
            if self.plot_embd_map: self.embd_ids.extend( ret["embd_ids"] )

    # Plot embd map
    def plot_embd_map(self):
        if self.plot_embd_map: embd_map_fname = self.embd_map_fname

        for fits_id in self.fits_ids:
            metrics = np.zeros((len(self.metric_options), 0, self.num_bands))
            metrics_zscale = np.zeros((len(self.metric_options), 0, self.num_bands))

            # plot residue map between recon and gt
            if self.plot_residue_heatmap:
                gt_fname = class_obj.gt_imgs_fnames[id]

                #recon_fname =
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
    # Task II:Recon spectra
    #############

    def recon_spectra(self, checkpoint):
        pass

    def calculate_recon_spectra_pixel_values(self):
        for fits_id in self.fits_ids:
            # calculate spectrum pixel recon value
            if args.plot_spectrum:
                print("recon spectrum pixel", recon[args.spectrum_pos])

    #############
    # Task III:Recon cdbk spectra
    #############

    def recon_cdbk_spectra(self, checkpoint):
        pass
