
import os
import torch
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from pathlib import Path
from os.path import exists, join
from wisp.inferrers import BaseInferrer
from wisp.utils.plot import plot_horizontally, plot_embed_map, plot_latent_embed
from wisp.utils.common import add_to_device, forward, load_model_weights, load_layer_weights


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
            Spectra reconstruction doesn"t need integration
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
        if "spectra_infer" in pipelines:
            self.spectra_infer_pipeline = pipelines["spectra_infer"]
        if "codebook" in pipelines:
            self.codebook_pipeline = pipelines["codebook"]

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
                ["model_dir","recon_dir","metric_dir", "spectra_dir",
                 "codebook_spectra_dir", "embed_map_dir","latent_dir",
                 "latent_embed_dir"],
                ["models","recons","metrics","spectra","codebook_spectra",
                 "embed_map","latents","latent_embed"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

    def select_models(self):
        self.selected_model_fnames = os.listdir(self.model_dir)
        self.selected_model_fnames.sort()
        if self.infer_last_model_only:
            self.selected_model_fnames = self.selected_model_fnames[-1:]
        self.num_models = len(self.selected_model_fnames)
        if self.verbose: log.info(f"selected {self.num_models} models")

    def summarize_inferrence_tasks(self):
        """ Group similar inferrence tasks (tasks using same dataset and same model) together.
        """
        tasks = set(self.extra_args["tasks"])
        self.quantize_latent = self.extra_args["quantize_latent"]

        # infer all coords using original model
        self.recon_img = "recon_img" in tasks
        self.recon_HSI = "recon_HSI" in tasks
        self.recon_flat_trans = "recon_flat_trans" in tasks

        self.plot_embed_map = "plot_embed_map" in tasks \
            and self.quantize_latent and self.space_dim == 3
        self.plot_latent_embed = "plot_latent_embed" in tasks \
            and self.quantize_latent and self.space_dim == 3

        # infer all coords using modified model
        self.recon_codebook_spectra = "recon_codebook_spectra" in tasks \
            and self.quantize_latent and self.space_dim == 3

        # infer selected coords using partial model
        self.recon_gt_spectra = "recon_gt_spectra" in tasks and self.space_dim == 3
        self.recon_dummy_spectra = "recon_dummy_spectra" in tasks and self.space_dim == 3

        # keep only tasks required to perform
        self.group_tasks = []

        if self.recon_img or self.recon_flat_trans or \
           self.plot_embed_map or self.plot_latent_embed:
            self.group_tasks.append("infer_all_coords_full_model")

        if self.recon_codebook_spectra:
            self.group_tasks.append("infer_hardcode_coords_modified_model")

        if self.recon_dummy_spectra or self.recon_gt_spectra:
            self.group_tasks.append("infer_selected_coords_partial_model")

        # set all grouped tasks to False, only required tasks will be toggled afterwards
        self.infer_all_coords_full_model = False
        self.infer_hardcode_coords_modified_model = False
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

            elif group_task == "infer_hardcode_coords_modified_model":
                self.infer_funcs[group_task] = [
                    self.pre_inferrence_hardcode_coords_modified_model,
                    self.post_inferrence_hardcode_coords_modified_model,
                    self.pre_checkpoint_hardcode_coords_modified_model,
                    self.run_checkpoint_hardcode_coords_modified_model,
                    self.post_checkpoint_hardcode_coords_modified_model ]

            else: raise Exception("Unrecgonized group inferrence task.")

    #############
    # Inferrence
    #############

    def pre_inferrence_all_coords_full_model(self):
        self.fits_ids = self.dataset.get_fits_ids()
        self.coords_source = "fits"
        self.batched_fields = ["coords"]
        if self.recon_img: self.batched_fields.append("pixels")
        self.dataset_length = self.dataset.get_num_coords()

        self.batch_size = self.extra_args["infer_batch_size"]
        self.reset_dataloader()

        num_fits = self.dataset.get_num_fits()
        # num_coords = self.dataset.get_num_coords()
        # self.num_batches = int(np.ceil(num_coords / self.batch_size))
        # if self.drop_last: self.num_batches -= 1

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
        self.coords_source = "spectra"
        self.batched_fields = ["coords"]
        self.dataset_length = self.dataset.get_num_spectra_coords()

        #self.num_spectra = self.dataset.get_num_spectra_coords()
        if not self.extra_args["infer_spectra_individually"]:
            # self.num_batches = int(np.ceil(num_coords / self.batch_size))
            self.batch_size = self.extra_args["infer_batch_size"]
        else:
            # self.num_batches = num_coords
            self.batch_size = self.extra_args["spectra_neighbour_size"]**2

        self.reset_dataloader()

    def post_inferrence_selected_coords_partial_model(self):
        pass

    def pre_inferrence_hardcode_coords_modified_model(self):
        self.coords_source = "codebook_latents"
        self.batched_fields = ["coords"]
        self.dataset_length = self.extra_args["qtz_num_embed"]

        self.batch_size = min(self.extra_args["infer_batch_size"],
                              self.extra_args["qtz_num_embed"])
        self.reset_dataloader()

    def post_inferrence_hardcode_coords_modified_model(self):
        pass

    #############
    # Infer with checkpoint
    #############

    def pre_checkpoint_all_coords_full_model(self, model_id):
        self.reset_data_iterator()

        if self.recon_img:
            self.to_HDU_now = self.extra_args["to_HDU"] and model_id == self.num_models
            self.recon_HSI_now = self.recon_HSI and model_id == self.num_models
            self.recon_flat_trans_now = self.recon_flat_trans and model_id == self.num_models
            #if self.recon_flat_trans_now: self.num_bands = 1
            self.recon_pixels = []

        if self.plot_embed_map:
            self.embed_ids = []

    def run_checkpoint_all_coords_full_model(self, model_id, checkpoint):
        epoch = checkpoint["epoch_trained"]
        model_state = checkpoint["model_state_dict"]
        self.infer_all_coords(model_id, model_state)
        if self.plot_latent_embed:
            plot_latent_embed(epoch, self.latent_dir, self.latent_embed_dir, model_state)

    def post_checkpoint_all_coords_full_model(self, model_id):
        if self.recon_img:
            re_args = {
                "fname": model_id,
                "dir": self.recon_dir,
                "metric_options": self.metric_options,
                "verbose": self.verbose,
                "num_bands": self.extra_args["num_bands"],
                "log_max": True,
                "save_locally": True,
                "plot_func": plot_horizontally,
                "zscale": True,
                "to_HDU": self.to_HDU_now,
                "calculate_metrics": self.calculate_metrics,
                "recon_flat_trans": self.recon_flat_trans_now
            }
            cur_metrics, cur_metrics_zscale = self.dataset.restore_evaluate_tiles(
                self.recon_pixels, **re_args)

            if self.calculate_metrics:
                # add metrics for current checkpoint
                self.metrics = np.concatenate((self.metrics, cur_metrics[:,None]), axis=1)
                self.metrics_zscale = np.concatenate((
                    self.metrics_zscale, cur_metrics_zscale[:,None]), axis=1)

        if self.plot_embed_map:
            re_args = {
                "fname": model_id,
                "dir": self.embed_map_dir,
                "verbose": self.verbose,
                "num_bands": 1,
                "log_max": False,
                "save_locally": True,
                "plot_func": plot_embed_map,
                "zscale": False,
                "to_HDU": False,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.embed_ids, **re_args)

    def pre_checkpoint_selected_coords_partial_model(self, model_id):
        self.reset_data_iterator()
        self.recon_spectra = []

    def run_checkpoint_selected_coords_partial_model(self, model_id, checkpoint):
        self.infer_spectra(model_id, checkpoint["model_state_dict"])

    def post_checkpoint_selected_coords_partial_model(self, model_id):
        self.recon_spectra = torch.stack(self.recon_spectra).view(
            self.dataset.get_num_gt_spectra(),
            self.extra_args["spectra_neighbour_size"]**2, -1
        ).detach().cpu().numpy()

        self.dataset.plot_spectrum(self.spectra_dir, model_id, self.recon_spectra,
                                   save_spectra=True)
        #self.calculate_recon_spectra_pixel_values()

    def pre_checkpoint_hardcode_coords_modified_model(self, model_id):
        self.reset_data_iterator()
        self.codebook_spectra = []

    def run_checkpoint_hardcode_coords_modified_model(self, model_id, checkpoint):
        self.infer_codebook_spectra(model_id, checkpoint["model_state_dict"])

    def post_checkpoint_hardcode_coords_modified_model(self, model_id):
        self.codebook_spectra = torch.stack(self.codebook_spectra).detach().cpu().numpy()
        self.dataset.plot_spectrum(
            self.codebook_spectra_dir, model_id, self.codebook_spectra,
            save_spectra=True, codebook=True)

    #############
    # Helpers
    #############

    def infer_all_coords(self, model_id, checkpoint):
        """ Using given checkpoint, reconstruct, if specified:
              multi-band image - np, to_HDU (FITS), recon_HSI (hyperspectral)
              flat-trans image,
              pixel embedding map
        """
        load_model_weights(self.full_pipeline, checkpoint)
        self.full_pipeline.eval()

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                with torch.no_grad():
                    ret = forward(
                        self, self.full_pipeline, data,
                        pixel_supervision_train=False,
                        spectra_supervision_train=False,
                        quantize_latent=self.quantize_latent,
                        calculate_codebook_loss=False,
                        infer=True,
                        save_spectra=False,
                        save_latents=False,
                        save_embed_ids=self.plot_embed_map)

                if self.recon_img: self.recon_pixels.extend(ret["intensity"])
                if self.plot_embed_map: self.embed_ids.extend(ret["min_embed_ids"])

            except StopIteration:
                log.info("all coords inferrence done")
                break

    def infer_spectra(self, model_id, checkpoint):
        load_model_weights(self.spectra_infer_pipeline, checkpoint)
        self.spectra_infer_pipeline.eval()

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                with torch.no_grad():
                    spectra = forward(
                        self, self.spectra_infer_pipeline, data,
                        pixel_supervision_train=False,
                        spectra_supervision_train=False,
                        quantize_latent=self.quantize_latent,
                        calculate_codebook_loss=False,
                        infer=True,
                        save_spectra=False,
                        save_latents=False,
                        save_embed_ids=False)["intensity"]

                if spectra.ndim == 3: # bandwise
                    spectra = spectra.flatten(1,2) # [bsz,nsmpl]
                self.recon_spectra.extend(spectra)

            except StopIteration:
                log.info("spectra inferrence done")
                break

    def calculate_recon_spectra_pixel_values(self):
        for fits_id in self.fits_ids:
            # calculate spectrum pixel recon value
            if args.plot_spectrum:
                print("recon spectrum pixel", recon[args.spectrum_pos])

    def infer_codebook_spectra(self, model_id, checkpoint):
        load_model_weights(self.codebook_pipeline, checkpoint)
        self.codebook_pipeline.eval()
        #print(self.codebook_pipeline)

        codebook_latents = load_layer_weights(
            checkpoint, lambda n: "grid" not in n and "codebook" in n)
        codebook_latents = codebook_latents.T[:,None] # [num_embd, 1, latent_dim]
        self.dataset.set_hardcode_data(self.coords_source, codebook_latents)

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                with torch.no_grad():
                    spectra = forward(
                        self, self.codebook_pipeline, data,
                        pixel_supervision_train=False,
                        spectra_supervision_train=False,
                        quantize_latent=False,
                        calculate_codebook_loss=False,
                        infer=True,
                        save_spectra=False,
                        save_latents=False,
                        save_embed_ids=False)["intensity"]

                self.codebook_spectra.extend(spectra)

            except StopIteration:
                log.info("codebook spectra inferrence done")
                break

    def _configure_dataset(self):
        """ Configure dataset (batched fields and len) for inferrence.
        """
        if self.space_dim == 3: self.batched_fields.extend(["trans_data"])

        # self.dataset.set_dataset_mode("infer")
        self.dataset.set_wave_sample_mode(use_full_wave=True)
        self.dataset.set_dataset_length(self.dataset_length)
        self.dataset.set_dataset_fields(self.batched_fields)
        self.dataset.set_dataset_coords_source(self.coords_source)
