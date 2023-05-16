
import os
import torch
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from os.path import exists, join

from wisp.inferrers import BaseInferrer
from wisp.utils.plot import plot_horizontally, plot_embed_map, plot_latent_embed, annotated_heat, plot_simple
from wisp.utils.common import add_to_device, forward, load_model_weights, load_layer_weights, load_embed, sort_alphanumeric


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
    def __init__(self, pipelines, dataset, device, mode="infer", **extra_args):
        """ @Param
               pipelines: a dictionary of pipelines for different inferrence tasks.
               dataset: use the same dataset object for all tasks.
               mode: toggle between `pretrain_infer` and `infer`.
        """
        super().__init__(pipelines, dataset, device, mode, **extra_args)

        self.set_log_path()
        self.select_models()
        self.init_model(pipelines)
        self.summarize_inferrence_tasks()
        self.set_inferrence_funcs()

    #############
    # Initializations
    #############

    def set_log_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose: log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["model_dir","recon_dir","recon_synthetic_dir","metric_dir", "spectra_dir",
                 "codebook_spectra_dir", "embed_map_dir","latent_dir",
                 "redshift_dir","latent_embed_dir","zoomed_recon_dir",
                 "scaler_dir","pixel_distrib_dir","soft_qtz_weights_dir"],
                ["models","recons","recon_synthetic","metrics","spectra","codebook_spectra",
                 "embed_map","latents","redshift","latent_embed","zoomed_recon",
                 "scaler","pixel_distrib","soft_qtz_weights"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

    def select_models(self):
        self.selected_model_fnames = os.listdir(self.model_dir)
        self.selected_model_fnames = sort_alphanumeric(self.selected_model_fnames)
        if self.infer_last_model_only:
            self.selected_model_fnames = self.selected_model_fnames[-1:]
        self.num_models = len(self.selected_model_fnames)
        if self.verbose: log.info(f"selected {self.num_models} models")

    def init_model(self, pipelines):
        if self.mode == "pretrain_infer":
            assert("pretrain_infer" in pipelines)
            self.full_pipeline = pipelines["pretrain_infer"]
            self.spectra_infer_pipeline = pipelines["pretrain_infer"]
            if self.recon_codebook_spectra:
                self.codebook_pipeline = pipelines["codebook"]
            elif self.recon_codebook_spectra_individ:
                self.codebook_pipeline = pipelines["codebook_individ"]
        else:
            if "full" in pipelines:
                self.full_pipeline = pipelines["full"]
            if "spectra_infer" in pipelines:
                self.spectra_infer_pipeline = pipelines["spectra_infer"]
            if "codebook" in pipelines:
                self.codebook_pipeline = pipelines["codebook"]

    def summarize_inferrence_tasks(self):
        """ Group similar inferrence tasks (tasks using same dataset and same model) together.
        """
        tasks = set(self.extra_args["tasks"])
        self.quantize_latent = self.extra_args["quantize_latent"]
        self.quantize_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        assert not (self.quantize_latent and self.quantize_spectra)

        # infer all coords using original model
        self.pretrain_infer = self.mode == "pretrain_infer"
        self.recon_img = "recon_img" in tasks
        self.recon_HSI = "recon_HSI" in tasks
        self.plot_pixel_distrib = "plot_pixel_distrib" in tasks
        self.recon_synthetic_band = "recon_synthetic_band" in tasks

        self.plot_embed_map = "plot_embed_map" in tasks \
            and (self.quantize_latent or self.quantize_spectra) \
            and self.space_dim == 3
        self.plot_latent_embed = "plot_latent_embed" in tasks \
            and (self.quantize_latent or self.quantize_spectra) \
            and self.space_dim == 3
        self.plot_redshift = "plot_redshift" in tasks \
            and self.extra_args["generate_redshift"] \
            and (self.quantize_latent or self.quantize_spectra) \
            and self.space_dim == 3
        self.log_redshift = "log_redshift" in tasks \
            and self.extra_args["generate_redshift"] \
            and (self.quantize_latent or self.quantize_spectra) \
            and self.space_dim == 3
        self.plot_scaler =  "plot_save_scaler" in tasks \
            and self.extra_args["generate_scaler"] \
            and (self.quantize_latent or self.quantize_spectra) \
            and self.space_dim == 3

        self.save_soft_qtz_weights = "save_soft_qtz_weights" in tasks \
            and ((self.quantize_latent and self.extra_args["quantization_strategy"] == "soft") \
                 or self.quantize_spectra) \
            and self.space_dim == 3

        self.log_soft_qtz_weights = "log_soft_qtz_weights" in tasks \
            and ((self.quantize_latent and self.extra_args["quantization_strategy"] == "soft") \
                 or self.quantize_spectra) \
            and self.space_dim == 3

        # infer all coords using modified model (recon codebook spectra)
        # either we have the codebook spectra for all coords
        # or we recon codebook spectra individually for each coord (when generate redshift)
        self.recon_codebook_spectra = "recon_codebook_spectra" in tasks \
            and (self.quantize_latent or self.quantize_spectra) \
            and self.space_dim == 3
        self.recon_codebook_spectra_individ = "recon_codebook_spectra_individ" in tasks \
            and (self.quantize_latent or self.quantize_spectra) \
            and self.space_dim == 3
        assert not (self.recon_codebook_spectra and self.recon_codebook_spectra_individ)

        # infer selected coords using partial model
        self.log_pixel_value = "log_pixel_value" in tasks # log spectra pixel value
        self.recon_gt_spectra = "recon_gt_spectra" in tasks and self.space_dim == 3
        self.recon_dummy_spectra = "recon_dummy_spectra" in tasks and self.space_dim == 3

        # keep only tasks required to perform
        self.group_tasks = []

        if self.recon_img or self.recon_synthetic_band or \
           self.plot_embed_map or self.plot_latent_embed or \
           self.plot_redshift or self.log_redshift  or self.plot_scaler or \
           self.save_soft_qtz_weights or self.log_soft_qtz_weights:
            self.group_tasks.append("infer_all_coords_full_model")

        if self.recon_dummy_spectra or self.recon_gt_spectra:
            self.group_tasks.append("infer_selected_coords_partial_model")

        if self.recon_codebook_spectra or self.recon_codebook_spectra_individ:
            self.group_tasks.append("infer_hardcode_coords_modified_model")

        if self.plot_pixel_distrib:
            self.group_tasks.append("infer_no_model_run")

        # set all grouped tasks to False, only required tasks will be toggled afterwards
        self.infer_all_coords_full_model = False
        self.infer_hardcode_coords_modified_model = False
        self.infer_selected_coords_partial_model = False
        self.infer_without_model_run = False

        if self.verbose:
            log.info(f"inferrence group tasks: {self.group_tasks}.")

    def set_inferrence_funcs(self):
        self.infer_funcs = {}

        for group_task in self.group_tasks:
            if group_task == "infer_all_coords_full_model":
                self.run_model = True
                self.infer_funcs[group_task] = [
                    self.pre_inferrence_all_coords_full_model,
                    self.post_inferrence_all_coords_full_model,
                    self.pre_checkpoint_all_coords_full_model,
                    self.run_checkpoint_all_coords_full_model,
                    self.post_checkpoint_all_coords_full_model ]

            elif group_task == "infer_selected_coords_partial_model":
                self.run_model = True
                self.infer_funcs[group_task] = [
                    self.pre_inferrence_selected_coords_partial_model,
                    self.post_inferrence_selected_coords_partial_model,
                    self.pre_checkpoint_selected_coords_partial_model,
                    self.run_checkpoint_selected_coords_partial_model,
                    self.post_checkpoint_selected_coords_partial_model ]

            elif group_task == "infer_hardcode_coords_modified_model":
                self.run_model = True
                self.infer_funcs[group_task] = [
                    self.pre_inferrence_hardcode_coords_modified_model,
                    self.post_inferrence_hardcode_coords_modified_model,
                    self.pre_checkpoint_hardcode_coords_modified_model,
                    self.run_checkpoint_hardcode_coords_modified_model,
                    self.post_checkpoint_hardcode_coords_modified_model ]

            elif group_task == "infer_no_model_run":
                self.run_model = False
                self.infer_funcs[group_task] = [None]*5

            else: raise Exception("Unrecgonized group inferrence task.")

    #############
    # Inferrence
    #############

    def pre_inferrence_all_coords_full_model(self):
        self.fits_uids = self.dataset.get_fits_uids()
        self.use_full_wave = True
        self.coords_source = "fits"
        self.perform_integration = self.recon_img

        self.requested_fields = ["coords"]
        if self.recon_img:
            self.requested_fields.append("pixels")
        if self.pretrain_infer:
            self.requested_fields.append("spectra_data")

        if self.pretrain_infer:
            self.dataset_length = self.extra_args["num_supervision_spectra"]
        else:
            self.dataset_length = self.dataset.get_num_coords()
        self.batch_size = min(self.extra_args["infer_batch_size"], self.dataset_length)

        self.reset_dataloader(drop_last=False)

        if self.recon_img:
            self.metric_options = self.extra_args["metric_options"]
            self.num_metrics = len(self.metric_options)
            self.calculate_metrics = self.recon_img and self.metric_options is not None

            if self.calculate_metrics:
                self.metrics = np.zeros((self.num_metrics, 0, num_fits, self.num_bands))
                self.metrics_zscale = np.zeros((self.num_metrics, 0, num_fits, self.num_bands))
                self.metric_fnames = [ join(self.metric_dir, f"{option}.npy")
                                       for option in self.metric_options ]
                self.metric_fnames_z = [ join(self.metric_dir, f"{option}_zscale.npy")
                                         for option in self.metric_options ]
        else:
            self.calculate_metrics = False

    def post_inferrence_all_coords_full_model(self):
        if self.calculate_metrics:
            [ np.save(self.metric_fnames[i], self.metrics[i])
              for i in range(self.num_metrics) ]
            [ np.save(self.metric_fnames_z[i], self.metrics_zscale[i])
              for i in range(self.num_metrics) ]
            log.info(f"metrics: {np.round(self.metrics[:,-1,0], 3)}")
            log.info(f"zscale metrics: {np.round(self.metrics_zscale[:,-1,0], 3)}")

    def pre_inferrence_selected_coords_partial_model(self):
        """ Spectra reconstruction.
        """
        self.use_full_wave = True
        self.perform_integration = False
        self.coords_source = "spectra"
        self.requested_fields = ["coords"]
        if self.pretrain_infer:
            self.dataset_length = self.extra_args["num_supervision_spectra"]
        else:
            self.dataset_length = self.dataset.get_num_spectra_coords()

        #self.num_spectra = self.dataset.get_num_spectra_coords()
        if not self.extra_args["infer_spectra_individually"]:
            # self.num_batches = int(np.ceil(num_coords / self.batch_size))
            self.batch_size = min(
                self.dataset_length * self.extra_args["spectra_neighbour_size"]**2,
                self.extra_args["infer_batch_size"])
        else:
            # self.num_batches = num_coords
            self.batch_size = self.extra_args["spectra_neighbour_size"]**2

        self.reset_dataloader()

    def post_inferrence_selected_coords_partial_model(self):
        pass

    def pre_inferrence_hardcode_coords_modified_model(self):
        """ Codebook spectra reconstruction.
        """
        self.use_full_wave = True
        self.perform_integration = False
        self.coords_source = "codebook_latents"
        self.requested_fields = ["coords"]

        if self.recon_codebook_spectra:
            self.dataset_length = self.extra_args["qtz_num_embed"]
        elif self.recon_codebook_spectra_individ:
            self.dataset_length = self.extra_args["num_supervision_spectra"]

        self.batch_size = min(self.extra_args["infer_batch_size"], self.dataset_length)
        self.reset_dataloader()

    def post_inferrence_hardcode_coords_modified_model(self):
        pass

    def inferrence_no_model_run(self):
        if self.plot_pixel_distrib:
            assert(exists(self.recon_dir))
            for fname in os.listdir(self.recon_dir):
                if not "npy" in fname: continue
                in_fname = join(self.recon_dir, fname)
                out_fname = join(self.pixel_distrib_dir, fname[:-4] + ".png")
                pixels = np.load(in_fname)
                plot_horizontally(pixels, out_fname, plot_option="plot_distrib")

    #############
    # Infer with checkpoint
    #############

    def pre_checkpoint_all_coords_full_model(self, model_id):
        self.reset_data_iterator()

        if self.recon_img:
            self.to_HDU_now = self.extra_args["to_HDU"] and model_id == self.num_models
            self.recon_HSI_now = self.recon_HSI and model_id == self.num_models
            # self.recon_flat_trans_now = self.recon_flat_trans and model_id == self.num_models
            self.recon_pixels = []

        if self.recon_synthetic_band:
            self.recon_synthetic_pixels = []

        if self.plot_scaler:
            self.scalers = []

        if self.plot_latent_embed:
            self.latents = []

        if self.plot_embed_map:
            self.embed_ids = []

        if self.plot_redshift or self.log_redshift:
            self.redshifts = []

        if self.save_soft_qtz_weights or self.log_soft_qtz_weights:
            self.soft_qtz_weights = []

    def run_checkpoint_all_coords_full_model(self, model_id, checkpoint):
        if self.pretrain_infer:
            self._set_dataset_coords_pretrain(checkpoint)
        self.infer_all_coords(model_id, checkpoint)

        if self.plot_latent_embed:
            self.embed = load_embed(checkpoint["model_state_dict"])

    def post_checkpoint_all_coords_full_model(self, model_id):
        if self.recon_img:
            re_args = {
                "fname": model_id,
                "dir": self.recon_dir,
                "metric_options": self.metric_options,
                "verbose": self.verbose,
                "num_bands": self.extra_args["num_bands"],
                "plot_func": plot_horizontally,
                "zscale": True,
                "log_max": True,
                "save_locally": True,
                "to_HDU": self.to_HDU_now,
                "calculate_metrics": self.calculate_metrics,
                "recon_synthetic_band": False,
                "zoom": self.extra_args["recon_zoomed"],
                "cutout_fits_uids": self.extra_args["recon_cutout_fits_uids"],
                "cutout_sizes": self.extra_args["recon_cutout_sizes"],
                "cutout_start_pos": self.extra_args["recon_cutout_start_pos"],
                "zoomed_recon_dir": self.zoomed_recon_dir,
                "zoomed_recon_fname": model_id,
            }
            cur_metrics, cur_metrics_zscale = self.dataset.restore_evaluate_tiles(
                self.recon_pixels, **re_args)

            if self.calculate_metrics:
                # add metrics for current checkpoint
                self.metrics = np.concatenate((self.metrics, cur_metrics[:,None]), axis=1)
                self.metrics_zscale = np.concatenate((
                    self.metrics_zscale, cur_metrics_zscale[:,None]), axis=1)

        if self.recon_synthetic_band:
            re_args = {
                "fname": model_id,
                "dir": self.recon_synthetic_dir,
                "metric_options": None,
                "verbose": self.verbose,
                "num_bands": 1,
                "plot_func": plot_horizontally,
                "zscale": True,
                "log_max": True,
                "save_locally": True,
                "to_HDU": False,
                "calculate_metrics": False,
                "recon_synthetic_band": True,
                "zoom": self.extra_args["recon_zoomed"],
                "cutout_fits_uids": self.extra_args["recon_cutout_fits_uids"],
                "cutout_sizes": self.extra_args["recon_cutout_sizes"],
                "cutout_start_pos": self.extra_args["recon_cutout_start_pos"],
                "zoomed_recon_dir": self.zoomed_recon_dir,
                "zoomed_recon_fname": model_id,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.recon_synthetic_pixels, **re_args)

        if self.plot_latent_embed:
            plot_latent_embed(self.latents, self.embed, model_id, self.latent_embed_dir)

        if self.plot_embed_map:
            if self.extra_args["mark_spectra"]:
                coords = self.dataset.get_spectra_img_coords()
            else: coords = []
            plot_embed_map_log = partial(plot_embed_map, coords)

            re_args = {
                "fname": model_id,
                "dir": self.embed_map_dir,
                "verbose": self.verbose,
                "num_bands": 1,
                "log_max": False,
                "save_locally": False,
                "plot_func": plot_embed_map_log,
                "zscale": False,
                "to_HDU": False,
                "match_fits": True,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.embed_ids, **re_args)

        if self.plot_redshift:
            if self.extra_args["mark_spectra"]:
                positions = self.dataset.get_spectra_img_coords() # [n,3] r/c/fits_id
                markers = np.array(self.extra_args["spectra_markers"])
            else:
                positions, markers = [], []
            plot_annotated_heat_map = partial(annotated_heat, positions, markers)

            re_args = {
                "fname": model_id,
                "dir": self.redshift_dir,
                "verbose": self.verbose,
                "num_bands": 1,
                "log_max": False,
                "to_HDU": False,
                "save_locally": False,
                "plot_func": plot_annotated_heat_map,
                "match_fits": True,
                "zscale": False,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.redshifts, **re_args)

        if self.log_redshift:
            assert(self.mode == "pretrain_infer")
            redshifts = torch.stack(self.redshifts).detach().cpu().numpy()
            fname = join(self.redshift_dir, f"{model_id}.pth")
            np.save(fname, redshifts)
            np.set_printoptions(precision=3)
            log.info(f"Est. redshift {redshifts}")

        if self.plot_scaler:
            re_args = {
                "fname": f'infer_{model_id}',
                "dir": self.scaler_dir,
                "verbose": self.verbose,
                "num_bands": 1,
                "log_max": False,
                "to_HDU": False,
                "save_locally": False,
                "plot_func": plot_horizontally,
                "match_fits": False,
                "zscale": True,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.scalers, **re_args)

        if self.save_soft_qtz_weights:
            re_args = {
                "fname": f'{model_id}',
                "dir": self.soft_qtz_weights_dir,
                "verbose": self.verbose,
                "num_bands": self.extra_args["qtz_num_embed"],
                "log_max": False,
                "to_HDU": False,
                "save_locally": True,
                "match_fits": False,
                "zscale": False,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.soft_qtz_weights, **re_args)

        if self.log_soft_qtz_weights:
            assert(self.mode == "pretrain_infer")
            weights = torch.stack(self.soft_qtz_weights).detach().cpu().numpy()
            fname = join(self.soft_qtz_weights_dir, f"{model_id}.pth")
            np.save(fname, weights)
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            log.info(f"Qtz weights {weights[:,0]}")

    def pre_checkpoint_selected_coords_partial_model(self, model_id):
        self.reset_data_iterator()
        self.recon_spectra = []

    def run_checkpoint_selected_coords_partial_model(self, model_id, checkpoint):
        if self.pretrain_infer:
            self._set_dataset_coords_pretrain(checkpoint)
        self.infer_spectra(model_id, checkpoint)

    def post_checkpoint_selected_coords_partial_model(self, model_id):
        self.recon_spectra = torch.stack(self.recon_spectra).view(
            self.dataset.get_num_spectra_to_plot(),
            self.extra_args["spectra_neighbour_size"]**2, -1
        ).detach().cpu().numpy()

        self.dataset.plot_spectrum(
            self.spectra_dir, model_id, self.recon_spectra,
            spectra_norm_cho=self.extra_args["spectra_norm_cho"],
            save_spectra=True, clip=self.extra_args["plot_clipped_spectrum"])

        if self.log_pixel_value:
            self.dataset.log_spectra_pixel_values(self.recon_spectra)

    def pre_checkpoint_hardcode_coords_modified_model(self, model_id):
        self.reset_data_iterator()
        self.codebook_spectra = []

    def run_checkpoint_hardcode_coords_modified_model(self, model_id, checkpoint):
        if self.recon_codebook_spectra:
            self._set_dataset_coords_codebook(checkpoint)
        elif self.recon_codebook_spectra_individ:
            self._set_dataset_coords_pretrain(checkpoint)
        self.infer_codebook_spectra(model_id, checkpoint)

    def post_checkpoint_hardcode_coords_modified_model(self, model_id):
        # [(num_supervision_spectra,)num_embeds,nsmpl]
        self.codebook_spectra = torch.stack(self.codebook_spectra).detach().cpu().numpy()
        # np.save('code.npy',self.codebook_spectra)

        # if spectra is 2d, add dummy 1st dim to simplify code
        if self.recon_codebook_spectra:
            self.codebook_spectra = [self.codebook_spectra]

        for i, codebook_spectra in enumerate(self.codebook_spectra):
            fname = f"{i}_{model_id}"
            self.dataset.plot_spectrum(
                self.codebook_spectra_dir, fname, codebook_spectra,
                spectra_norm_cho=self.extra_args["spectra_norm_cho"],
                save_spectra=True, codebook=True,
                clip=self.extra_args["plot_clipped_spectrum"]
            )

    #############
    # Infer logic
    #############

    def infer_all_coords(self, model_id, checkpoint):
        """ Using given checkpoint, reconstruct, if specified:
              multi-band image - np, to_HDU (FITS), recon_HSI (hyperspectral)
              flat-trans image,
              pixel embedding map
        """
        iterations = checkpoint["epoch_trained"]
        model_state = checkpoint["model_state_dict"]
        load_model_weights(self.full_pipeline, model_state)
        self.full_pipeline.eval()

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                with torch.no_grad():
                    ret = forward(
                        data,
                        self.full_pipeline,
                        iterations,
                        self.space_dim,
                        self.extra_args["trans_sample_method"],
                        pretrain_infer=self.pretrain_infer,
                        quantize_latent=self.quantize_latent,
                        quantize_spectra=self.quantize_spectra,
                        quantization_strategy=self.extra_args["quantization_strategy"],
                        save_soft_qtz_weights=self.save_soft_qtz_weights or self.log_soft_qtz_weights,
                        recon_img=self.recon_img,
                        save_scaler=self.plot_scaler,
                        save_embed_ids=self.plot_embed_map,
                        save_latents=self.plot_latent_embed,
                        save_redshift=self.plot_redshift or self.log_redshift,
                    )

                if self.recon_img:
                    # artifically generated transmission function (last channel)
                    if self.recon_synthetic_band:
                        self.recon_synthetic_pixels.extend(ret["intensity"][...,-1:])
                        ret["intensity"] = ret["intensity"][...,:-1]
                    self.recon_pixels.extend(ret["intensity"])

                if self.plot_scaler:
                    self.scalers.extend(ret["scaler"])
                if self.plot_latent_embed:
                    self.latents.extend(ret["latents"])
                if self.plot_embed_map:
                    self.embed_ids.extend(ret["min_embed_ids"])
                if self.plot_redshift or self.log_redshift:
                    self.redshifts.extend(ret["redshift"])
                if self.save_soft_qtz_weights or self.log_soft_qtz_weights:
                    self.soft_qtz_weights.extend(ret["soft_qtz_weights"])

            except StopIteration:
                if self.verbose: log.info("all coords inferrence done")
                break

    def infer_spectra(self, model_id, checkpoint):
        iterations = checkpoint["epoch_trained"]
        model_state = checkpoint["model_state_dict"]
        load_model_weights(self.spectra_infer_pipeline, model_state)
        self.spectra_infer_pipeline.eval()

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                with torch.no_grad():
                    spectra = forward(
                        data,
                        self.spectra_infer_pipeline,
                        iterations,
                        # self.extra_args["num_epochs"],
                        self.space_dim,
                        self.extra_args["trans_sample_method"],
                        pretrain_infer=self.pretrain_infer,
                        quantize_latent=self.quantize_latent,
                        quantize_spectra=self.quantize_spectra,
                        quantization_strategy=self.extra_args["quantization_strategy"],
                        recon_spectra=True)["intensity"]

                if spectra.ndim == 3: # bandwise
                    spectra = spectra.flatten(1,2) # [bsz,nsmpl]
                self.recon_spectra.extend(spectra)

            except StopIteration:
                break

    def infer_codebook_spectra(self, model_id, checkpoint):
        """ Reconstruct codebook spectra.
            The logic is identical between normal inferrence and pretrain inferrence.
        """
        iterations = checkpoint["epoch_trained"]
        model_state = checkpoint["model_state_dict"]
        load_model_weights(self.codebook_pipeline, model_state)
        self.codebook_pipeline.eval()

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                with torch.no_grad():
                    ret = forward(
                        data,
                        self.codebook_pipeline,
                        iterations,
                        self.space_dim,
                        self.extra_args["trans_sample_method"],
                        quantize_latent=self.quantize_latent,
                        quantize_spectra=self.quantize_spectra,
                        quantization_strategy=self.extra_args["quantization_strategy"],
                        recon_codebook_spectra=True,
                        save_codebook=self.recon_codebook_spectra_individ
                    )

                    if self.recon_codebook_spectra:
                        spectra = ret["intensity"]
                    else:
                        spectra = ret["codebook"]

                # np.save('code.npy',spectra.detach().cpu().numpy())
                self.codebook_spectra.extend(spectra)

            except StopIteration:
                if self.verbose: log.info("codebook spectra inferrence done")
                break

    #############
    # Helpers
    #############

    def _configure_dataset(self):
        """ Configure dataset (batched fields and len) for inferrence.
        """
        if self.space_dim == 3:
            self.requested_fields.extend(["trans_data"])

        self.dataset.set_mode(self.mode)
        self.dataset.set_length(self.dataset_length)
        self.dataset.set_fields(self.requested_fields)
        self.dataset.set_coords_source(self.coords_source)
        self.dataset.toggle_wave_sampling(self.use_full_wave)
        self.dataset.toggle_integration(self.perform_integration)

    def _set_dataset_coords_pretrain(self, checkpoint):
        """ Set spectra latent vars as input coords (for pretrain infer only).
        """
        self.dataset.set_coords_source("spectra_latents")
        self.dataset.set_hardcode_data("spectra_latents", checkpoint["latents"].weight)

    def _set_dataset_coords_codebook(self, checkpoint):
        """ Set codebook weights as input coords (for codebook spectra recon only).
        """
        codebook_latents = load_layer_weights(
            checkpoint['model_state_dict'], lambda n: "grid" not in n and "codebook" in n)
        codebook_latents = codebook_latents[:,None] # [num_embd, 1, latent_dim]
        codebook_latents = codebook_latents.detach().cpu().numpy()
        self.dataset.set_coords_source("codebook_latents")
        self.dataset.set_hardcode_data("codebook_latents", codebook_latents)

    # def calculate_recon_spectra_pixel_values(self):
    #     for fits_uid in self.fits_uids:
    #         # calculate spectrum pixel recon value
    #         if args.plot_spectrum:
    #             print("recon spectrum pixel", recon[args.spectrum_pos])
