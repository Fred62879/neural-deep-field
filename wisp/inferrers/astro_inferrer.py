
import os
import torch
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from os.path import exists, join

from wisp.inferrers import BaseInferrer
from wisp.utils.plot import plot_horizontally, plot_embed_map, \
    plot_latent_embed, annotated_heat, plot_simple
from wisp.utils.common import add_to_device, forward, select_inferrence_ids, \
    load_model_weights, load_layer_weights, load_embed, sort_alphanumeric


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

        if mode == "pretrain_infer":
            self.num_sup_spectra = dataset.get_num_supervision_spectra()
        else: self.num_val_spectra = dataset.get_num_validation_spectra()

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
                 "scaler_dir","pixel_distrib_dir","qtz_weights_dir"],
                ["models","recons","recon_synthetic","metrics","spectra","codebook_spectra",
                 "embed_map","latents","redshift","latent_embed","zoomed_recon",
                 "scaler","pixel_distrib","qtz_weights"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

    def select_models(self):
        self.model_fnames = os.listdir(self.model_dir)
        self.selected_model_fnames = sort_alphanumeric(self.model_fnames)
        if self.infer_last_model_only:
            self.selected_model_fnames = self.selected_model_fnames[-1:]
        self.num_models = len(self.selected_model_fnames)
        if self.verbose: log.info(f"selected {self.num_models} models")

    def init_model(self, pipelines):
        self.full_pipeline = pipelines["full"]
        if self.recon_gt_spectra:
            self.spectra_infer_pipeline = pipelines["spectra_infer"]
        if self.recon_codebook_spectra or self.recon_codebook_spectra_individ:
            self.codebook_spectra_infer_pipeline = pipelines["codebook_spectra_infer"]

    def summarize_inferrence_tasks(self):
        """ Group similar inferrence tasks (tasks using same dataset and same model) together.
        """
        tasks = set(self.extra_args["tasks"])

        self.main_infer = self.mode == "infer"
        self.pretrain_infer = self.mode == "pretrain_infer"
        self.infer_selected = self.extra_args["infer_selected"]

        # quantization setups
        self.qtz_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.qtz_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        assert not (self.qtz_latent and self.qtz_spectra)
        self.qtz = self.qtz_latent or self.qtz_spectra
        self.qtz_strategy = self.extra_args["quantization_strategy"]
        self.generate_scaler = self.qtz and self.extra_args["generate_scaler"]

        # redshift setups
        self.model_redshift = self.extra_args["model_redshift"]
        self.apply_gt_redshift = self.model_redshift and \
            self.extra_args["apply_gt_redshift"]
        self.redshift_unsupervision = self.model_redshift and \
            self.extra_args["redshift_unsupervision"]
        self.redshift_semi_supervision = self.model_redshift and \
            self.extra_args["redshift_semi_supervision"]
        if self.pretrain_infer: assert self.apply_gt_redshift
        #else: assert self.redshift_semi_supervision

        # infer all coords using original model
        self.trans_sample_method = self.extra_args["trans_sample_method"]
        self.recon_img = False
        self.recon_img_pretrain = False
        self.recon_img_valid_spectra = False
        self.recon_HSI = "recon_HSI" in tasks
        self.recon_synthetic_band = "recon_synthetic_band" in tasks
        self.recon_spectra_pixels_only = self.extra_args["train_spectra_pixels_only"]
        if "recon_img" in tasks:
            if self.pretrain_infer:
                self.recon_img_pretrain = self.extra_args["codebook_pretrain_pixel_supervision"]
            else: # self.main_infer
                if self.recon_spectra_pixels_only:
                    self.recon_img_valid_spectra = True
                else: self.recon_img = True

        self.plot_pixel_distrib = "plot_pixel_distrib" in tasks
        self.plot_embed_map = "plot_embed_map" in tasks and self.qtz
        self.plot_latent_embed = "plot_latent_embed" in tasks and self.qtz

        self.save_qtz_weights = "save_qtz_weights" in tasks and \
            ((self.qtz_latent and self.qtz_strategy == "soft") or self.qtz_spectra)
        self.save_qtz_w_main = self.save_qtz_weights and self.main_infer
        self.save_qtz_w_pre = self.save_qtz_weights and self.pretrain_infer

        self.save_scaler = "save_scaler" in tasks and self.generate_scaler and \
            self.qtz and self.main_infer
        self.save_redshift = "save_redshift" in tasks and self.model_redshift and self.qtz
        self.save_redshift_pre = self.save_redshift and self.pretrain_infer
        self.save_redshift_main = self.save_redshift and self.main_infer

        # infer all coords using modified model (recon codebook spectra)
        #   either we have the codebook spectra for all coords
        #   or we recon codebook spectra individually for each coord (when generate redshift)
        self.recon_codebook_spectra = "recon_codebook_spectra" in tasks and self.qtz
        self.recon_codebook_spectra_individ = "recon_codebook_spectra_individ" in tasks and self.qtz and self.model_redshift
        assert not (self.recon_codebook_spectra and self.recon_codebook_spectra_individ)

        # infer selected coords using partial model
        self.log_pixel_value = "log_pixel_value" in tasks # log spectra pixel value
        self.recon_gt_spectra = "recon_gt_spectra" in tasks and self.space_dim == 3
        self.recon_dummy_spectra = "recon_dummy_spectra" in tasks and self.space_dim == 3

        # keep only tasks required to perform
        self.group_tasks = []

        if self.recon_HSI or self.recon_synthetic_band or \
           self.plot_embed_map or self.plot_latent_embed or \
           self.save_qtz_weights or self.save_redshift_main or self.save_scaler or \
           self.recon_img or self.recon_img_pretrain or self.recon_img_valid_spectra:
            self.group_tasks.append("infer_all_coords_full_model")

        if self.recon_dummy_spectra or self.recon_gt_spectra or \
           self.save_qtz_weights or self.save_redshift_pre:
            self.group_tasks.append("infer_selected_coords_partial_model")

        if self.recon_codebook_spectra or self.recon_codebook_spectra_individ:
            self.group_tasks.append("infer_hardcode_coords_modified_model")

        if self.plot_pixel_distrib:
            self.group_tasks.append("infer_no_model_run")

        # set all grouped tasks to False, only required tasks will be toggled afterwards
        self.infer_without_model_run = False
        self.infer_all_coords_full_model = False
        self.infer_selected_coords_partial_model = False
        self.infer_hardcode_coords_modified_model = False

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
        self.patch_uids = self.dataset.get_patch_uids()

        self.use_full_wave = True
        self.calculate_metrics = False
        self.perform_integration = self.recon_img or self.recon_img_pretrain or \
            self.recon_img_valid_spectra

        self.requested_fields = ["coords"]

        if self.pretrain_infer:
            self.wave_source = "spectra"
            # coords source udpated to spectra latents at start of inferrence
            self.coords_source = "spectra_latents"

            self.requested_fields.append("spectra_sup_data")
            if self.recon_img_pretrain:
                self.requested_fields.append("pixels")
            if self.apply_gt_redshift:
                self.requested_fields.append("spectra_sup_redshift")

            if self.infer_selected:
                self.dataset_length = min(
                    self.extra_args["pretrain_num_infer"], self.num_sup_spectra)
            else: self.dataset_length = self.num_sup_spectra

        else: # self.main_infer
            self.wave_source = "trans"
            self.coords_source = "fits"

            if self.recon_img:
                self.requested_fields.append("pixels")
            if self.save_redshift_main:
                self.requested_fields.append("redshift_data")
                # self.requested_fields.append("spectra_bin_map")
            if self.redshift_semi_supervision:
                self.requested_fields.extend(["spectra_id_map","spectra_bin_map"])

            if self.recon_img:
                self.metric_options = self.extra_args["metric_options"]
                self.num_metrics = len(self.metric_options)
                self.calculate_metrics = self.recon_img and self.metric_options is not None

                if self.calculate_metrics:
                    num_patches = self.dataset.get_num_patches()
                    self.metrics = np.zeros((self.num_metrics, 0, num_patches, self.num_bands))
                    self.metrics_zscale = np.zeros((
                        self.num_metrics, 0, num_patches, self.num_bands))
                    self.metric_fnames = [ join(self.metric_dir, f"{option}.npy")
                                           for option in self.metric_options ]
                    self.metric_fnames_z = [ join(self.metric_dir, f"{option}_zscale.npy")
                                             for option in self.metric_options ]

            if self.recon_spectra_pixels_only:
                self.dataset_length = self.num_val_spectra
            else:
                self.dataset_length = self.dataset.get_num_coords()

        self.batch_size = min(self.extra_args["infer_batch_size"], self.dataset_length)
        self.reset_dataloader()

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
        self.requested_fields = ["coords"]

        if self.pretrain_infer:
            self.requested_fields.extend([
                "spectra_sup_data",
                "spectra_sup_mask",
                "spectra_sup_redshift"
            ])
            self.wave_source = "spectra"
            self.coords_source = "spectra_train"
            # pretrain coords set using checkpoint
            if self.infer_selected:
                self.dataset_length = self.extra_args["pretrain_num_infer"]
            else: self.dataset_length = self.num_sup_spectra
        else: # self.main_infer
            self.wave_source = "trans"
            self.coords_source = "spectra_valid"
            self._set_dataset_coords_cur_val_coords()
            self.dataset_length = self.num_cur_val_coords

        if not self.extra_args["infer_spectra_individually"]:
            self.batch_size = min(
                self.dataset_length * self.extra_args["spectra_neighbour_size"]**2,
                self.extra_args["infer_batch_size"])
        else: self.batch_size = self.extra_args["spectra_neighbour_size"]**2
        self.reset_dataloader()

    def post_inferrence_selected_coords_partial_model(self):
        pass

    def pre_inferrence_hardcode_coords_modified_model(self):
        """ Codebook spectra reconstruction.
        """
        self.use_full_wave = True
        self.perform_integration = False
        self.requested_fields = ["coords"]

        if self.pretrain_infer:
            self.wave_source = "spectra"
        else: self.wave_source = "trans"

        if self.recon_codebook_spectra:
            self.coords_source = "codebook_latents"
            self.dataset_length = self.extra_args["qtz_num_embed"]

        elif self.recon_codebook_spectra_individ:
            if self.pretrain_infer:
                self.coords_source = "codebook_latents"
                self.requested_fields.extend([
                    "spectra_sup_data",
                    "spectra_sup_mask",
                    "spectra_sup_redshift"
                ])

                if self.infer_selected:
                    self.dataset_length = self.extra_args["pretrain_num_infer"]
                else:
                    self.dataset_length = self.num_sup_spectra
            else: # self.main_infer
                self.coords_source = "spectra_valid"
                self._set_dataset_coords_cur_val_coords()
                self.dataset_length = self.num_cur_val_coords

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
            self.recon_pixels = []
            self.recon_HSI_now = self.recon_HSI and model_id == self.num_models
            self.to_HDU_now = self.extra_args["to_HDU"] and model_id == self.num_models
            # self.recon_flat_trans_now = self.recon_flat_trans and model_id == self.num_models
        if self.save_scaler: self.scalers = []
        if self.plot_embed_map: self.embed_ids = []
        if self.plot_latent_embed: self.latents = []
        if self.save_redshift_main: self.redshift = []
        if self.save_qtz_weights: self.qtz_weights = []
        if self.recon_img_pretrain: self.recon_pixels = []
        if self.recon_synthetic_band: self.recon_synthetic_pixels = []

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
                "cutout_patch_uids": self.extra_args["recon_cutout_patch_uids"],
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

        elif self.recon_img_pretrain:
            assert(self.pretrain_infer)
            vals = torch.stack(self.recon_pixels).detach().cpu().numpy()
            fname = join(self.recon_dir, f"{model_id}.pth")
            np.save(fname, vals)
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            # log.info(f"Recon vals {vals}")
            gt_vals = self.dataset.get_supervision_spectra_pixels()[:,0]
            # log.info(f"GT vals {gt_vals}")
            ratio = gt_vals / vals
            log.info(f"gt/recon ratio: {ratio}")

        elif self.recon_img_valid_spectra:
            vals = torch.stack(self.recon_pixels).detach().cpu().numpy()
            fname = join(self.recon_dir, f"{model_id}.pth")
            np.save(fname, vals)
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            log.info(f"Recon vals {vals}")
            gt_vals = self.dataset.get_supervision_spectra_pixels()[:,0]
            log.info(f"GT vals {gt_vals}")

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
                "cutout_patch_uids": self.extra_args["recon_cutout_patch_uids"],
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
                "match_patch": True,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.embed_ids, **re_args)

        if self.save_redshift_main:
            # plot redshift img
            if self.extra_args["mark_spectra"]:
                # positions = self.dataset.get_spectra_img_coords() # [n,3] r/c/patch_id
                positions = self.cur_patch.get_spectra_img_coords()
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
                "match_patch": True,
                "zscale": False,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.redshift, **re_args)

            # log redshift for spectra pixels
            if self.extra_args["pretrain_codebook"]:
                redshift = torch.stack(self.redshift).detach().cpu().numpy()
                redshift = redshift[self.val_spectra_map]
                gt_redshift = self.gt_redshift.detach().cpu().numpy()
                np.set_printoptions(suppress=True)
                np.set_printoptions(precision=3)
                log.info(f"est redshift (full): {redshift}")
                log.info(f"GT. redshift (full): {gt_redshift}")

        if self.save_scaler:
            re_args = {
                "fname": f'infer_{model_id}',
                "dir": self.scaler_dir,
                "verbose": self.verbose,
                "num_bands": 1,
                "log_max": False,
                "to_HDU": False,
                "save_locally": False,
                "plot_func": plot_horizontally,
                "match_patch": False,
                "zscale": True,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.scalers, **re_args)

            if self.main_infer and self.extra_args["pretrain_codebook"]:
                scalers = torch.stack(self.scalers).detach().cpu().numpy()
                scalers = scalers[self.val_spectra_map]
                np.set_printoptions(suppress=True)
                np.set_printoptions(precision=3)
                log.info(f"scaler: {scalers}")

        if self.save_qtz_w_pre:
            weights = torch.stack(self.qtz_weights).detach().cpu().numpy()[:,0]
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            log.info(f"qtz weights (full): {weights}")

        elif self.save_qtz_w_main:
            re_args = {
                "fname": f'{model_id}',
                "dir": self.qtz_weights_dir,
                "verbose": self.verbose,
                "num_bands": self.extra_args["qtz_num_embed"],
                "log_max": False,
                "to_HDU": False,
                "save_locally": True,
                "match_patch": False,
                "zscale": False,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.qtz_weights, **re_args)

            weights = torch.stack(self.qtz_weights).detach().cpu().numpy()[:,0]
            weights = weights[self.val_spectra_map]
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            log.info(f"qtz weights (full): {weights}")

    def pre_checkpoint_selected_coords_partial_model(self, model_id):
        self.reset_data_iterator()
        self.gt_fluxes = []
        self.recon_fluxes = []
        self.spectra_wave = []
        self.spectra_masks = []
        if self.save_redshift_pre: self.redshift = []
        if self.save_qtz_weights: self.qtz_weights = []

    def run_checkpoint_selected_coords_partial_model(self, model_id, checkpoint):
        if self.pretrain_infer:
            self._set_dataset_coords_pretrain(checkpoint)
        self.infer_spectra(model_id, checkpoint)

    def post_checkpoint_selected_coords_partial_model(self, model_id):
        self.gt_fluxes = torch.stack(self.gt_fluxes).view(
            self.dataset_length, -1).detach().cpu().numpy()
        self.recon_fluxes = torch.stack(self.recon_fluxes).view(
            self.dataset_length, self.extra_args["spectra_neighbour_size"]**2, -1
        ).detach().cpu().numpy()
        self.spectra_wave = torch.stack(self.spectra_wave).view(
            self.dataset_length, -1).detach().cpu().numpy()
        self.spectra_masks = torch.stack(self.spectra_masks).bool().view(
            self.dataset_length, -1).detach().cpu().numpy()

        self.dataset.plot_spectrum(
            self.spectra_dir, model_id, self.extra_args["flux_norm_cho"],
            self.spectra_wave, self.gt_fluxes, self.recon_fluxes,
            mode=self.mode, save_spectra=True,
            clip=self.extra_args["plot_clipped_spectrum"],
            masks=self.spectra_masks,
            spectra_clipped=False
        )

        if self.log_pixel_value:
            self.dataset.log_spectra_pixel_values(self.recon_spectra)

        if self.save_redshift_pre:
            redshift = torch.stack(self.redshift).detach().cpu().numpy()
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            log.info(f"redshift (patrtial): {redshift}")

        if self.save_qtz_weights:
            weights = torch.stack(self.qtz_weights).detach().cpu().numpy()[:,0]
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            log.info(f"qtz weights (patrtial): {weights}")

    def pre_checkpoint_hardcode_coords_modified_model(self, model_id):
        self.reset_data_iterator()
        self.codebook_spectra = []
        self.spectra_wave_c = []
        self.spectra_masks_c = []

    def run_checkpoint_hardcode_coords_modified_model(self, model_id, checkpoint):
        if self.recon_codebook_spectra:
            self._set_dataset_coords_codebook(checkpoint)
        elif self.recon_codebook_spectra_individ and self.pretrain_infer:
            self._set_dataset_coords_pretrain(checkpoint)

        self.infer_codebook_spectra(model_id, checkpoint)

    def post_checkpoint_hardcode_coords_modified_model(self, model_id):
        # [(num_supervision_spectra,)num_embeds,nsmpl]
        self.codebook_spectra = torch.stack(self.codebook_spectra).detach().cpu().numpy()
        self.spectra_wave_c = torch.stack(self.spectra_wave_c).view(
            self.dataset_length, -1).detach().cpu().numpy()
        self.spectra_masks_c = torch.stack(self.spectra_masks_c).bool().view(
            self.dataset_length, -1).detach().cpu().numpy()

        print(self.codebook_spectra.shape, self.spectra_wave_c.shape, self.spectra_masks_c.shape)

        # if spectra is 2d, add dummy 1st dim to simplify code
        if self.recon_codebook_spectra:
            self.codebook_spectra = [self.codebook_spectra]
            prefix = ""
        else: prefix = "individ-"

        for i, (wave, masks, codebook_spectra) in enumerate(
                zip(self.spectra_wave_c, self.spectra_masks_c, self.codebook_spectra)
        ):
            print(wave.shape, masks.shape, codebook_spectra.shape)
            cur_dir = join(self.codebook_spectra_dir, f"spectra-{i}")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            fname = f"{prefix}{model_id}"
            wave = np.tile(wave[None,:], self.dataset_length).reshape(-1, self.dataset_length).T
            masks = np.tile(masks[None,:], self.dataset_length).reshape(-1, self.dataset_length).T
            self.dataset.plot_spectrum(
                cur_dir, fname, self.extra_args["flux_norm_cho"],
                wave, None, codebook_spectra,
                mode=self.mode,
                is_codebook=True,
                save_spectra_together=True,
                clip=self.extra_args["plot_clipped_spectrum"],
                masks=masks
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
                        qtz=self.qtz,
                        qtz_strategy=self.qtz_strategy,
                        apply_gt_redshift=self.apply_gt_redshift,
                        perform_integration=self.perform_integration,
                        trans_sample_method=self.trans_sample_method,
                        save_qtz_weights=self.save_qtz_weights,
                        save_scaler=self.save_scaler,
                        save_embed_ids=self.plot_embed_map,
                        save_latents=self.plot_latent_embed,
                        save_redshift=self.save_redshift_main,
                    )

                if self.recon_img or self.recon_img_pretrain:
                    # artifically generated transmission function (last channel)
                    if self.recon_synthetic_band:
                        self.recon_synthetic_pixels.extend(ret["intensity"][...,-1:])
                        ret["intensity"] = ret["intensity"][...,:-1]
                    self.recon_pixels.extend(ret["intensity"])

                if self.save_scaler:
                    self.scalers.extend(ret["scaler"])
                if self.plot_latent_embed:
                    self.latents.extend(ret["latents"])
                if self.plot_embed_map:
                    self.embed_ids.extend(ret["min_embed_ids"])
                if self.save_qtz_weights:
                    self.qtz_weights.extend(ret["qtz_weights"])
                if self.save_redshift_main:
                    self.redshift.extend(ret["redshift"])
                    if self.extra_args["pretrain_codebook"]:
                        self.gt_redshift = data["spectra_sup_redshift"]

            except StopIteration:
                log.info("all coords inferrence done")
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
                    ret = forward(
                        data,
                        self.spectra_infer_pipeline,
                        iterations,
                        self.space_dim,
                        qtz=self.qtz,
                        qtz_strategy=self.qtz_strategy,
                        apply_gt_redshift=self.apply_gt_redshift,
                        save_spectra=True,
                        save_redshift=self.save_redshift_pre,
                        save_qtz_weights=self.save_qtz_weights
                    )

                self.spectra_masks.extend(data["spectra_sup_mask"])
                self.spectra_wave.extend(data["spectra_sup_data"][:,0])
                self.gt_fluxes.extend(data["spectra_sup_data"][:,1])

                fluxes = ret["intensity"]
                if fluxes.ndim == 3: # bandwise
                    fluxes = fluxes.flatten(1,2) # [bsz,nsmpl]
                self.recon_fluxes.extend(fluxes)
                if self.save_redshift_pre:
                    self.redshift.extend(ret["redshift"])
                if self.save_qtz_weights:
                    self.qtz_weights.extend(ret["qtz_weights"])

            except StopIteration:
                log.info("spectra forward done")
                break

    def infer_codebook_spectra(self, model_id, checkpoint):
        """ Reconstruct codebook spectra.
            The logic is identical between normal inferrence and pretrain inferrence.
        """
        iterations = checkpoint["epoch_trained"]
        model_state = checkpoint["model_state_dict"]
        load_model_weights(self.codebook_spectra_infer_pipeline, model_state)
        self.codebook_spectra_infer_pipeline.eval()

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                with torch.no_grad():
                    ret = forward(
                        data,
                        self.codebook_spectra_infer_pipeline,
                        iterations,
                        self.space_dim,
                        qtz=self.qtz,
                        qtz_strategy=self.qtz_strategy,
                        apply_gt_redshift=self.apply_gt_redshift,
                        save_codebook_spectra=self.recon_codebook_spectra_individ
                    )

                    if self.recon_codebook_spectra:
                        spectra = ret["intensity"]
                    elif self.recon_codebook_spectra_individ:
                        spectra = ret["codebook_spectra"]
                    else: assert 0

                self.codebook_spectra.extend(spectra)

            except StopIteration:
                log.info("codebook spectra forward done")
                break

    #############
    # Helpers
    #############

    def _configure_dataset(self):
        """ Configure dataset (batched fields and len) for inferrence.
        """
        if self.space_dim == 3:
            self.requested_fields.extend(["wave_data"])

        self.dataset.set_mode(self.mode)
        self.dataset.set_length(self.dataset_length)
        self.dataset.set_fields(self.requested_fields)
        self.dataset.set_wave_source(self.wave_source)
        self.dataset.set_coords_source(self.coords_source)
        self.dataset.toggle_wave_sampling(not self.use_full_wave)
        self.dataset.toggle_integration(self.perform_integration)

        # select the same random set of spectra to recon
        if self.pretrain_infer and self.infer_selected:
            ids = select_inferrence_ids(
                self.num_sup_spectra,
                self.extra_args["pretrain_num_infer"]
            )
            self.dataset.set_hardcode_data("selected_ids", ids)

    def _set_dataset_coords_pretrain(self, checkpoint):
        """ Set spectra latent vars as input coords (for pretrain infer only).
        """
        latents = checkpoint["latents"].weight
        self.dataset.set_hardcode_data(self.coords_source, latents)

    def _set_dataset_coords_codebook(self, checkpoint):
        """ Set codebook weights as input coords (for codebook spectra recon only).
        """
        codebook_latents = load_layer_weights(
            checkpoint['model_state_dict'], lambda n: "grid" not in n and "codebook" in n)
        codebook_latents = codebook_latents[:,None] # [num_embd, 1, latent_dim]
        codebook_latents = codebook_latents.detach().cpu().numpy()
        self.dataset.set_hardcode_data(self.coords_source, codebook_latents)

    def _set_dataset_coords_cur_val_coords(self):
        # cur_patch_spectra_ids = self.dataset.get_validation_spectra_ids(self.cur_patch_uid)
        # cur_val_coords = self.dataset.get_validation_spectra_norm_world_coords(cur_patch_spectra_ids)
        cur_val_coords = self.dataset.get_coords()[self.val_spectra_map]
        # print(cur_val_coords)
        self.dataset.set_hardcode_data(self.coords_source, cur_val_coords)
        self.num_cur_val_coords = len(cur_val_coords)

    # def calculate_recon_spectra_pixel_values(self):
    #     for patch_uid in self.patch_uids:
    #         # calculate spectrum pixel recon value
    #         if args.plot_spectrum:
    #             print("recon spectrum pixel", recon[args.spectrum_pos])
