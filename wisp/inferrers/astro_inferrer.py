
import os
import torch
import pickle
import numpy as np
import logging as log
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from os.path import exists, join
from functools import partial, lru_cache

from wisp.inferrers import BaseInferrer
from wisp.loss import spectra_supervision_loss
from wisp.datasets.data_utils import get_bound_id
from wisp.datasets.data_utils import get_neighbourhood_center_pixel_id

from wisp.utils.plot import plot_horizontally, plot_embed_map, \
    plot_latent_embed, annotated_heat, plot_simple, plot_multiple, \
    plot_precision_recall_all, plot_precision_recall_single
from wisp.utils.common import add_to_device, forward, select_inferrence_ids, \
    sort_alphanumeric, get_bool_classify_redshift, init_redshift_bins, to_numpy, \
    load_model_weights, load_pretrained_model_weights, load_layer_weights, load_embed, \
    get_loss, get_bin_ids, get_optimal_wrong_bin_ids, log_data, get_bool_has_redshift_latents, create_batch_ids


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
    def __init__(self, pipelines, dataset, device, mode, **extra_args):
        """ @Param
               pipelines: a dictionary of pipelines for different inferrence tasks.
               dataset: use the same dataset object for all tasks.
               mode: toggle between `pretrain_infer`,`main_infer`,and `test`.
        """
        super().__init__(pipelines, dataset, device, mode, **extra_args)

        self.summarize_inferrence_tasks()
        self.set_path()

        self.init_data()
        self.init_model()
        self.select_models()
        self.set_inferrence_funcs()

    #############
    # Initializations
    #############

    def set_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose: log.info(f"logging to {self.log_dir}")
        prefix = {"codebook_pretrain_infer":"pretrain",
                  "redshift_pretrain_infer":"pretrain",
                  "main_infer":"val", "test":"test"}[self.mode]

        for cur_path, cur_pname, in zip(
                ["model_dir","recon_dir","recon_synthetic_dir","metric_dir",
                 "zoomed_recon_dir","scaler_dir","pixel_distrib_dir","qtz_weights_dir",
                 "embed_map_dir","latent_dir","redshift_dir","latent_embed_dir","spectra_dir",
                 "codebook_coeff_dir","codebook_latents_dir",
                 "codebook_spectra_dir", "codebook_spectra_individ_dir"],
                ["models","recons","recon_synthetic","metrics",
                 "zoomed_recon","scaler","pixel_distrib","qtz_weights",
                 "embed_map","latents","redshift","latent_embed",f"{prefix}_spectra",
                 f"{prefix}_codebook_coeff",f"{prefix}_codebook_latents",
                 f"{prefix}_codebook_spectra",f"{prefix}_codebook_spectra_individ"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

    def init_data(self):
        if self.mode == "codebook_pretrain_infer" or self.mode == "redshift_pretrain_infer":
            self.batch_size = self.extra_args["pretrain_infer_batch_size"]

            if self.mode == "redshift_pretrain_infer":
                self.dataset.set_spectra_source("val")
                if self.extra_args["sample_from_codebook_pretrain_spectra"]:
                    self.num_spectra = self.extra_args["redshift_pretrain_num_spectra"]
                else: self.num_spectra = self.dataset.get_num_validation_spectra()
            else:
                self.dataset.set_spectra_source("sup")
                self.num_spectra = self.dataset.get_num_supervision_spectra()

        elif self.mode == "main_infer":
            self.batch_size = self.extra_args["infer_batch_size"]
            self.dataset.set_spectra_source("val")
            self.num_spectra = self.dataset.get_num_validation_spectra()

        elif self.mode == "test":
            self.batch_size = self.extra_args["infer_batch_size"]
            self.dataset.set_spectra_source("test")
            self.num_spectra = self.dataset.get_num_test_spectra()

        else: raise ValueError()

    def init_model(self):
        self.full_pipeline = self.pipelines["full"]
        log.info(self.full_pipeline)
        if self.recon_spectra or self.recon_spectra_all_bins or self.recon_redshift:
            self.spectra_infer_pipeline = self.pipelines["spectra_infer"]
        if self.recon_codebook_spectra or self.recon_codebook_spectra_individ:
            self.codebook_spectra_infer_pipeline = self.pipelines["codebook_spectra_infer"]

    def select_models(self):
        self.model_fnames = os.listdir(self.model_dir)
        self.selected_model_fnames = sort_alphanumeric(self.model_fnames)
        if self.infer_last_model_only:
            self.selected_model_fnames = self.selected_model_fnames[-1:]
        self.num_models = len(self.selected_model_fnames)
        if self.verbose: log.info(f"selected {self.num_models} models")

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

    def summarize_inferrence_tasks(self):
        """ Group similar inferrence tasks (tasks using same dataset and same model) together.
        """
        tasks = set(self.extra_args["tasks"])

        self.test = self.mode == "test"
        self.main_infer = self.mode == "main_infer"
        self.codebook_pretrain_infer = self.mode == "codebook_pretrain_infer"
        self.redshift_pretrain_infer = self.mode == "redshift_pretrain_infer"
        self.pretrain_infer = self.codebook_pretrain_infer or self.redshift_pretrain_infer
        assert sum([self.test,self.main_infer,self.pretrain_infer]) == 1
        # assert not self.redshift_pretrain_infer or not self.infer_selected

        # quantization setups
        self.qtz_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.qtz_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        assert not (self.qtz_latent and self.qtz_spectra)
        self.qtz = self.qtz_latent or self.qtz_spectra
        self.qtz_n_embd = self.extra_args["qtz_num_embed"]
        self.qtz_strategy = self.extra_args["quantization_strategy"]
        self.generate_scaler = self.qtz and self.extra_args["decode_scaler"]

        # redshift setups
        self.model_redshift = self.extra_args["model_redshift"]
        self.apply_gt_redshift = self.model_redshift and \
            self.extra_args["apply_gt_redshift"]
        self.redshift_unsupervision = self.model_redshift and \
            self.extra_args["redshift_unsupervision"]
        self.redshift_semi_supervision = self.model_redshift and \
            self.extra_args["redshift_semi_supervision"]

        self.classify_redshift = get_bool_classify_redshift(**self.extra_args)
        self.neg_sup_wrong_redshift = \
            self.extra_args["negative_supervise_wrong_redshift"]
        self.calculate_binwise_spectra_loss = \
            self.extra_args["calculate_binwise_spectra_loss"]
        self.use_binwise_spectra_loss_as_redshift_logits = \
            self.extra_args["use_binwise_spectra_loss_as_redshift_logits"]
        self.optimize_codebook_latents_for_each_redshift_bin = \
            self.extra_args["optimize_codebook_latents_for_each_redshift_bin"]
        self.has_redshift_latents = get_bool_has_redshift_latents(**self.extra_args)

        assert not self.codebook_pretrain_infer or (
            self.apply_gt_redshift or self.neg_sup_wrong_redshift)
        assert not self.neg_sup_wrong_redshift or (
            self.mode == "codebook_pretrain_infer" and self.classify_redshift and \
            self.calculate_binwise_spectra_loss)
        assert not self.calculate_binwise_spectra_loss or \
            self.extra_args["spectra_batch_reduction_order"] == "qtz_first"
        assert not self.use_binwise_spectra_loss_as_redshift_logits or \
            (self.classify_redshift and self.calculate_binwise_spectra_loss)
        assert not self.optimize_codebook_latents_for_each_redshift_bin or \
            self.calculate_binwise_spectra_loss
        assert not self.optimize_codebook_latents_for_each_redshift_bin or \
            (self.classify_redshift and not self.use_binwise_spectra_loss_as_redshift_logits), \
            "For the brute force method, we keep spectra under all bins without averaging. \
            During inferrence however, we can calculate logits for visualization purposes."

        if self.classify_redshift:
            redshift_bins = init_redshift_bins(
                self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"],
                self.extra_args["redshift_bin_width"]
            )
            self.num_redshift_bins = len(redshift_bins)

        self.save_redshift = "save_redshift" in tasks and self.model_redshift and self.qtz
        self.save_redshift_pre = self.save_redshift and self.pretrain_infer
        self.save_redshift_main = self.save_redshift and self.main_infer
        self.save_redshift_test = self.save_redshift and self.test

        # i) infer all coords using original model
        self.recon_img_all_pixels = False
        self.recon_img_sup_spectra = False  # -
        self.recon_img_val_spectra = False  #  |- recon spectra pixel values
        self.recon_img_test_spectra = False # -
        self.recon_HSI = "recon_HSI" in tasks
        self.recon_synthetic_band = "recon_synthetic_band" in tasks

        if "recon_img" in tasks:
            if self.pretrain_infer:
                self.recon_img_sup_spectra = \
                    self.extra_args["pretrain_pixel_supervision"]
            elif self.main_infer:
                if self.recon_spectra_pixels_only:
                    self.recon_img_val_spectra = True
                else: self.recon_img_all_pixels = True
            elif self.test:
                self.recon_img_test_spectra = True
            else:
                raise ValueError()
        self.recon_img = self.recon_img_all_pixels or self.recon_img_sup_spectra or \
            self.recon_img_val_spectra or self.recon_img_test_spectra or \
            self.recon_HSI or self.recon_synthetic_band

        self.plot_embed_map = "plot_embed_map" in tasks and self.qtz
        self.plot_latent_embed = "plot_latent_embed" in tasks and self.qtz
        self.save_qtz_weights = "save_qtz_weights" in tasks and \
            ((self.qtz_latent and self.qtz_strategy == "soft") or self.qtz_spectra)
        self.save_qtz_w_main = self.save_qtz_weights and self.main_infer
        self.save_qtz_w_pre = self.save_qtz_weights and self.pretrain_infer
        self.save_scaler = "save_scaler" in tasks and self.generate_scaler and \
            self.qtz and self.main_infer

        # ii) infer selected coords using partial model
        self.recon_spectra = "recon_spectra" in tasks
        # save spectra pixel values, doing same job as
        #   `recon_img_sup_spectra` during pretran infer &
        #   `recon_img_val_spectra` during main train infer
        self.save_pixel_values = "save_pixel_values" in tasks
        self.save_optimal_bin_ids = "save_optimal_bin_ids" in tasks
        self.plot_codebook_coeff = "plot_codebook_coeff" in tasks
        self.plot_redshift_logits = "plot_redshift_logits" in tasks
        self.save_codebook_latents = "save_codebook_latents" in tasks
        self.recon_spectra_all_bins = "recon_spectra_all_bins" in tasks
        self.plot_binwise_spectra_loss = "plot_binwise_spectra_loss" in tasks
        self.plot_codebook_coeff_all_bins = "plot_codebook_coeff_all_bins" in tasks

        self.plot_gt_bin_spectra = not self.recon_spectra_all_bins and \
            self.extra_args["plot_spectrum_under_gt_bin"]
        self.plot_optimal_wrong_bin_spectra = not self.recon_spectra_all_bins and \
            self.extra_args["plot_spectrum_under_optimal_wrong_bin"]
        self.plot_optimal_wrong_bin_codebook_coeff = not self.recon_spectra_all_bins and \
            self.extra_args["plot_codebook_coeff_under_optimal_wrong_bin"]

        assert not self.plot_codebook_coeff or self.qtz_spectra
        assert not self.save_codebook_latents or self.qtz_spectra
        assert not self.plot_redshift_logits or self.classify_redshift
        assert not self.save_optimal_bin_ids or self.calculate_binwise_spectra_loss
        assert not self.plot_gt_bin_spectra or self.calculate_binwise_spectra_loss
        assert not self.plot_binwise_spectra_loss or self.calculate_binwise_spectra_loss
        assert not self.plot_optimal_wrong_bin_spectra or self.calculate_binwise_spectra_loss
        assert not self.plot_optimal_wrong_bin_codebook_coeff or self.calculate_binwise_spectra_loss
        assert not self.plot_codebook_coeff_all_bins or (
            self.qtz_spectra and self.calculate_binwise_spectra_loss)
        assert sum([self.recon_spectra, self.recon_spectra_all_bins]) <= 1

        self.recon_redshift = self.save_redshift or self.save_optimal_bin_ids or \
            self.plot_redshift_logits or self.plot_binwise_spectra_loss or \
            self.plot_codebook_coeff or self.plot_codebook_coeff_all_bins

        # iii) infer all coords using modified model (recon codebook spectra)
        #   either we have the codebook spectra for all coords
        #   or we recon codebook spectra individually for each coord (when generate redshift)
        self.recon_codebook_spectra = "recon_codebook_spectra" in tasks and self.qtz
        self.recon_codebook_spectra_individ = "recon_codebook_spectra_individ" in tasks \
            and self.qtz and self.model_redshift
        assert not (self.recon_codebook_spectra and self.recon_codebook_spectra_individ)

        # iv) infer w/o no modeling running
        self.plot_img_residual = "plot_img_residual" in tasks
        self.integrate_gt_spectra = "integrate_gt_spectra" in tasks
        self.plot_gt_pixel_distrib = "plot_gt_pixel_distrib" in tasks

        # *) keep only tasks required
        self.group_tasks = []

        if self.recon_img or self.save_qtz_weights or \
           self.save_scaler or self.plot_embed_map or self.plot_latent_embed:
            self.group_tasks.append("infer_all_coords_full_model")

        if self.recon_redshift or self.recon_spectra or self.recon_spectra_all_bins:
           self.group_tasks.append("infer_selected_coords_partial_model")

        if self.recon_codebook_spectra or self.recon_codebook_spectra_individ:
            self.group_tasks.append("infer_hardcode_coords_modified_model")

        if self.plot_img_residual or self.integrate_gt_spectra or self.plot_gt_pixel_distrib:
            self.group_tasks.append("infer_no_model_run")

        self.infer_selected = self.extra_args["infer_selected"]
        self.infer_outlier_only = self.extra_args["infer_outlier_only"]
        assert sum([self.infer_selected, self.infer_outlier_only]) <= 1
        assert not self.infer_outlier_only or (
            self.save_redshift_pre and self.calculate_binwise_spectra_loss), \
            "we need to infer redshift to find outlier!"

        # set all grouped tasks to False, only required tasks will be toggled afterwards
        self.infer_without_model_run = False
        self.infer_all_coords_full_model = False
        self.infer_selected_coords_partial_model = False
        self.infer_hardcode_coords_modified_model = False

        if self.verbose:
            log.info(f"inferrence group tasks: {self.group_tasks}.")

    #############
    # Inferrence
    #############

    def pre_inferrence_all_coords_full_model(self):
        self.patch_uids = self.dataset.get_patch_uids()

        self.use_all_wave = True
        self.calculate_metrics = False
        self.perform_integration = True #self.recon_img

        self.requested_fields = []

        if self.pretrain_infer:
            assert (not self.redshift_pretrain_infer or self.infer_selected), \
                "we shall only infer selected spectra during codebook pretrain infer."

            self.coords_source = None
            self.wave_source = "spectra"
            self.use_all_wave = self.extra_args["pretrain_infer_use_all_wave"]
            if not self.use_all_wave:
                self.num_wave_samples = self.extra_args["pretrain_infer_num_wave"]
                self.wave_sample_method = self.extra_args["pretrain_infer_wave_sample_method"]

            self.requested_fields.extend(["idx","spectra_source_data","spectra_masks"])
            if self.recon_img: # _sup_spectra
                self.requested_fields.append("spectra_pixels")
            if self.apply_gt_redshift:
                self.requested_fields.append("spectra_redshift")

            if self.infer_selected:
                n = len(self._select_inferrence_ids())
                self.dataset_length = min(n, self.num_spectra)
                self.requested_fields.append("selected_ids")
            else: self.dataset_length = self.num_spectra

        elif self.main_infer:
            self.requested_fields.append("coords")
            self.wave_source = "trans"

            if self.recon_img_val_spectra:
                self.coords_source = "spectra_coords"
                self._set_coords_from_spectra_source(
                    self.dataset.get_validation_spectra_coords(), w_neighbour=False)
                self.requested_fields.append("spectra_pixels")
            elif self.recon_img_all_pixels:
                self.coords_source = "fits"
                self.dataset_length = self.dataset.get_num_coords()
                self.requested_fields.append("pixels")
                self.configure_img_metrics()
            else: raise ValueError()

            if self.save_redshift:
                if self.recon_spectra_pixels_only:
                    self.requested_fields.append("spectra_redshift")
                else:
                    # self.requested_fields.append("redshift_data")
                    self.requested_fields.extend([
                        "spectra_id_map","spectra_bin_map","redshift_data"])
            # if self.redshift_semi_supervision:
            #     self.requested_fields.extend(["spectra_id_map","spectra_bin_map"])

        elif self.test:
            assert 0
            # todo: replace with patch-wise test spectra
            self.wave_source = "trans"
            self.coords_source = "spectra_coords"
            self._set_coords_from_spectra_source(
                self.dataset.get_test_spectra_coords(), w_neighbour=False)
            if self.recon_img:
                self.requested_fields.append("spectra_pixels")
            if self.save_redshift:
                self.requested_fields.append("spectra_redshift")
        else:
            raise ValueError()

        self.batch_size = min(self.batch_size, self.dataset_length)
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
        self.use_all_wave = True
        self.perform_integration = False

        self.requested_fields = []

        if self.pretrain_infer:
            self.coords_source = None
            self.wave_source = "spectra"
            self.use_all_wave = self.extra_args["pretrain_infer_use_all_wave"]
            if not self.use_all_wave:
                self.num_wave_samples = self.extra_args["pretrain_infer_num_wave"]
                self.wave_sample_method = self.extra_args["pretrain_infer_wave_sample_method"]

            self.requested_fields.extend([
                "idx","spectra_source_data","spectra_masks","spectra_redshift"])
            if self.plot_gt_bin_spectra or self.plot_optimal_wrong_bin_spectra or \
               self.recon_spectra_all_bins:
                self.requested_fields.append("gt_redshift_bin_ids")
            if self.neg_sup_wrong_redshift or self.plot_optimal_wrong_bin_spectra:
                self.requested_fields.append("gt_redshift_bin_masks")

            if self.infer_selected:
                n = len(self._select_inferrence_ids())
                self.dataset_length = min(n, self.num_spectra)
                self.requested_fields.append("selected_ids")
            else: self.dataset_length = self.num_spectra

        elif self.main_infer:
            self.requested_fields.append("coords")

            self.wave_source = "trans"
            self.coords_source = "spectra_valid"
            self._set_coords_from_spectra_source(
                self.dataset.get_validation_spectra_coords())

            if self.save_redshift:
                assert(self.extra_args["spectra_neighbour_size"] == 1)
                self.requested_fields.append("spectra_redshift")

        elif self.test:
            assert 0
            # todo: replace with patch-wise test spectra
            self.wave_source = "trans"
            self.coords_source = "spectra_test"
            self._set_coords_from_spectra_source(
                self.dataset.get_test_spectra_coords())
        else:
            raise ValueError()

        self.configure_spectra_metrics()

        if not self.extra_args["infer_spectra_individually"]:
            self.batch_size = min(
                self.dataset_length * self.neighbour_size**2, self.batch_size)
        else: self.batch_size = self.neighbour_size**2
        self.reset_dataloader()

    def post_inferrence_selected_coords_partial_model(self):
        """ Log and save spectra metrics.
            @Input
              self.metrics: [[ dict[metric_name:val] ]
        """
        if len(self.metrics) != 0:
            metric_options = self.metrics[0][0].keys()
            self.metrics = np.array([
                [
                    [ v for k,v in cur_spectra_metrics.items() ]
                    for cur_spectra_metrics in cur_model_metrics
                ] for cur_model_metrics in self.metrics
            ]) # [n_models,n_spectra,n_metrics]

            for i, metric_option in enumerate(metric_options):
                fname = join(self.metric_dir, f"spectra_{metric_option}.npy")
                np.save(fname, self.metrics[...,i])
                cur_metrics = self.metrics[-1,:,i].T
                avg = np.mean(cur_metrics)
                log.info(f"avg_{metric_option}: {np.round(avg, 3)}")
                log.info(f"{metric_option}: {np.round(cur_metrics, 3)}")

    def pre_inferrence_hardcode_coords_modified_model(self):
        """ Codebook spectra reconstruction.
        """
        self.use_all_wave = True
        self.perform_integration = False

        self.requested_fields = []

        if self.recon_codebook_spectra:
            self.wave_source = "full_spectra"
        elif self.recon_codebook_spectra_individ:
            if self.pretrain_infer:
                self.wave_source = "spectra"
                self.use_all_wave = self.extra_args["pretrain_infer_use_all_wave"]
                self.num_wave_samples = self.extra_args["pretrain_infer_num_wave"]
                self.wave_sample_method = self.extra_args["pretrain_infer_wave_sample_method"]
            else:
                self.wave_source = "trans"
        else:
            raise ValueError()

        if self.recon_codebook_spectra:
            assert not self.infer_selected
            self.requested_fields.append("coords")
            self.coords_source = "codebook_latents"
            self.dataset_length = self.qtz_n_embd
            # if self.extra_args["plot_clipped_spectrum"]:
            #     self.requested_fields.append("spectra_masks")

        elif self.recon_codebook_spectra_individ:
            if self.pretrain_infer:
                self.coords_source = None
                self.requested_fields.extend([
                    "idx","spectra_source_data","spectra_masks","spectra_redshift"])

                if self.infer_selected:
                    n = len(self._select_inferrence_ids())
                    self.dataset_length = min(n, self.num_spectra)
                    self.requested_fields.append("selected_ids")
                else:
                    input("plot codebook spectra for all spectra, press Enter to confirm...")
                    self.dataset_length = self.num_spectra

            elif self.main_infer:
                self.coords_source = "spectra_valid"
                self._set_coords_from_spectra_source(
                    self.dataset.get_validation_spectra_coords()
                )
            elif self.test:
                assert 0
                # todo: replace with patch-wise test spectra
                self.coords_source = "spectra_test"
                self._set_coords_from_spectra_source(
                    self.dataset.get_test_spectra_coords())
            else:
                raise ValueError()
        else:
            raise ValueError()

        self.batch_size = min(self.batch_size, self.dataset_length)
        self.reset_dataloader()

    def post_inferrence_hardcode_coords_modified_model(self):
        pass

    def inferrence_no_model_run(self):
        if self.plot_img_residual:
            gt_fname = self.cur_patch.get_gt_img_fname() + ".npy"
            # `model_id` requires manual setup
            model_id = 0
            recon_fname = join(self.recon_dir, f"{self.cur_patch_uid}_model-{model_id}.npy")
            out_fname = join(self.recon_dir, f"{self.cur_patch_uid}_model-{model_id}_residual.png")
            gt = np.load(gt_fname)
            recon = np.load(recon_fname)
            residual = gt - recon

            kwargs = {"resid_lo": self.kwargs["img_resid_lo"],
                      "resid_hi": self.kwargs["img_resid_hi"] }
            plot_horizontally(residual, out_fname, plot_option="plot_heat_map", **kwargs)

        if self.plot_gt_pixel_distrib:
            assert(exists(self.recon_dir))
            for fname in os.listdir(self.recon_dir):
                if not "npy" in fname: continue
                in_fname = join(self.recon_dir, fname)
                out_fname = join(self.pixel_distrib_dir, fname[:-4] + ".png")
                pixels = np.load(in_fname)
                plot_horizontally(pixels, out_fname, plot_option="plot_distrib")

        if self.integrate_gt_spectra:
            self.gt_pixels = self.dataset.get_validation_spectra_pixels().numpy()
            valid_spectra = self.dataset.get_validation_spectra()[:,:2] # [bsz,2,nsmpl]
            valid_spectra_masks = self.dataset.get_validation_spectra_masks() # [bsz,nsmpl]
            func = self.dataset.get_transmission_interpolation_function()

            self.recon_pixels = []
            for (spectra, mask) in zip(valid_spectra, valid_spectra_masks):
                wave = spectra[0]
                (id_lo, id_hi) = get_bound_id(self.dataset.get_trans_wave_range(), wave)
                wave = wave[id_lo:id_hi+1]
                interp_trans = func(wave)
                nsmpl = np.sum(interp_trans != 0, axis=-1)
                self.recon_pixels.append( np.einsum("j,kj->k", wave, interp_trans) / nsmpl )

            log_data(self, "recon_pixels", gt_field="gt_pixels", log_ratio=True)

    #############
    # Infer with checkpoint
    #############

    def pre_checkpoint_all_coords_full_model(self, model_id):
        self.reset_data_iterator()

        if self.recon_img:
            self.gt_pixels = []
            self.recon_pixels = []
            # self.recon_HSI_now = self.recon_HSI and model_id == self.num_models
            # self.to_HDU_now = self.extra_args["to_HDU"] and model_id == self.num_models
            # self.recon_flat_trans_now = self.recon_flat_trans and model_id == self.num_models
        if self.save_scaler: self.scalers = []
        if self.plot_embed_map: self.embed_ids = []
        if self.plot_latent_embed: self.latents = []
        if self.save_qtz_weights: self.qtz_weights = []
        if self.recon_synthetic_band: self.recon_synthetic_pixels = []
        if self.save_redshift:
            if self.classify_redshift:
                self.argmax_redshift = []
                self.weighted_redshift = []
            else: self.redshift = []
            self.gt_redshift = []

    def run_checkpoint_all_coords_full_model(self, model_id, checkpoint):
        if self.pretrain_infer:
            # self._set_coords_from_checkpoint(checkpoint)
            self.full_pipeline.set_latents(checkpoint["model_state_dict"]["nef.latents"])
            if not self.apply_gt_redshift and self.split_latent:
                self.full_pipeline.set_redshift_latents(
                    checkpoint["model_state_dict"]["nef.redshift_latents"])

        self.infer_all_coords(model_id, checkpoint)

        if self.plot_latent_embed:
            self.embed = load_embed(checkpoint["model_state_dict"])

    def post_checkpoint_all_coords_full_model(self, model_id):
        if self.recon_img:
            if self.recon_img_all_pixels:
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
                    "to_HDU": False, #self.to_HDU_now,
                    "calculate_metrics": self.calculate_metrics,
                    "recon_synthetic_band": False,
                    "zoom": self.extra_args["recon_zoomed"],
                    "cutout_patch_uids": self.extra_args["recon_cutout_patch_uids"],
                    "cutout_sizes": self.extra_args["recon_cutout_sizes"],
                    "cutout_start_pos": self.extra_args["recon_cutout_start_pos"],
                    "zoomed_recon_dir": self.zoomed_recon_dir,
                    "zoomed_recon_fname": model_id,
                    "plot_residual": self.plot_residual_map
                }
                cur_metrics, cur_metrics_zscale = self.dataset.restore_evaluate_tiles(
                    self.recon_pixels, **re_args)

                if self.calculate_metrics:
                    # add metrics for current checkpoint
                    self.metrics = np.concatenate((self.metrics, cur_metrics[:,None]), axis=1)
                    self.metrics_zscale = np.concatenate((
                        self.metrics_zscale, cur_metrics_zscale[:,None]), axis=1)
            else:
                fname = join(self.recon_dir, f"model-{model_id}.pth")
                log_data(self, "recon_pixels", fname=fname, gt_field="gt_pixels")

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
            if self.pretrain_infer:
                self.embed_ids = torch.stack(self.embed_ids).detach().cpu().numpy()
                log.info(f"embed ids: {self.embed_ids}")
            else:
                coords = []
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
            # if self.recon_spectra_pixels_only:
            #     if self.classify_redshift:
            #         log_data(self, "argmax_redshift", gt_field="gt_redshift")
            #         log_data(self, "weighted_redshift")
            #     else:
            #         log_data(self, "redshift", gt_field="gt_redshift")
            # else:
            #     self._plot_redshift_map(model_id)
            #     if len(self.gt_redshift) > 0:
            #         log_data(self,
            #             "redshift", gt_field="gt_redshift", mask=self.val_spectra_map)

            if not self.recon_spectra_pixels_only:
                self._plot_redshift_map(model_id)
                mask=self.val_spectra_map
            else: mask = None

            if self.classify_redshift:
                log_data(self, "argmax_redshift", gt_field="gt_redshift", mask=mask)
                log_data(self, "weighted_redshift", mask=mask)
            else: log_data(self, "redshift", gt_field="gt_redshift", mask=mask)

        elif self.save_redshift_test:
            if self.classify_redshift:
                log_data(self, "argmax_redshift", gt_field="gt_redshift")
                log_data(self, "weighted_redshift")
            else:
                log_data(self, "redshift", gt_field="gt_redshift")

        if self.save_scaler:
            re_args = {
                "fname": f'infer_model-{model_id}',
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

            if self.main_infer:
                log_data(self, "scalers", mask=self.val_spectra_map)

        if self.save_qtz_w_pre:
            log_data(self, "qtz_weights")

        elif self.save_qtz_w_main:
            re_args = {
                "fname": f'model-{model_id}',
                "dir": self.qtz_weights_dir,
                "verbose": self.verbose,
                "num_bands": self.qtz_n_embd,
                "log_max": False,
                "to_HDU": False,
                "save_locally": True,
                "match_patch": False,
                "zscale": False,
                "calculate_metrics": False,
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.qtz_weights, **re_args)
            log_data(self, "qtz_weights", mask=self.val_spectra_map)

        log.info("== All coords inferrence done for current checkpoint.")

    def pre_checkpoint_selected_coords_partial_model(self, model_id):
        self.reset_data_iterator()

        if self.recon_spectra or self.recon_spectra_all_bins:
            self.gt_fluxes = []
            self.spectra_wave = []
            self.spectra_masks = []

        if self.recon_spectra:
            self.recon_fluxes = []
            if self.plot_gt_bin_spectra:
                self.gt_bin_fluxes = []
                self.gt_bin_spectra_losses = []
            if self.plot_optimal_wrong_bin_spectra:
                self.optimal_wrong_bin_fluxes = []
                self.optimal_wrong_bin_spectra_losses = []

        if self.recon_spectra_all_bins:
            self.recon_fluxes_all = []

        if self.plot_codebook_coeff:
            self.gt_redshift_cl = []
            self.codebook_coeff = []
            if self.plot_optimal_wrong_bin_codebook_coeff:
                self.optimal_wrong_bin_ids_cc = []

        if self.plot_codebook_coeff_all_bins:
            self.codebook_coeff_all_bins = []

        if self.plot_redshift_logits:
            self.gt_redshift_l = []
            self.redshift_logits = []

        if self.plot_binwise_spectra_loss:
            self.gt_redshift_bl = []
            self.binwise_loss = []

        if self.save_optimal_bin_ids:
            self.gt_bin_ids = []
            self.optimal_bin_ids = []
            self.optimal_wrong_bin_ids = []

        if self.save_redshift:
            if self.classify_redshift:
                self.argmax_redshift = []
                self.weighted_redshift = []
            else: self.redshift = []
            self.gt_redshift = []

        if self.save_qtz_weights: self.qtz_weights = []
        if self.save_pixel_values: self.spectra_trans = []
        if self.save_codebook_latents: self.codebook_latents = []

    def run_checkpoint_selected_coords_partial_model(self, model_id, checkpoint):
        if self.pretrain_infer:
            self.spectra_infer_pipeline.set_latents(
                checkpoint["model_state_dict"]["nef.latents"])
            if self.has_redshift_latents:
                self.spectra_infer_pipeline.set_redshift_latents(
                    checkpoint["model_state_dict"]["nef.redshift_latents"])

        self.infer_spectra(model_id, checkpoint)

    def post_checkpoint_selected_coords_partial_model(self, model_id):
        if self.pretrain_infer:
            num_spectra = self.dataset_length

            if self.recon_spectra or self.recon_spectra_all_bins:
                self.gt_fluxes = torch.stack(self.gt_fluxes).view(
                    num_spectra, -1).detach().cpu().numpy()
                self.gt_wave = torch.stack(self.spectra_wave).view(
                    num_spectra, -1).detach().cpu().numpy()
                self.gt_masks = torch.stack(self.spectra_masks).bool().view(
                    num_spectra, -1).detach().cpu().numpy()

                self.recon_wave = self.gt_wave
                self.recon_masks = self.gt_masks

                if self.recon_spectra:
                    self.recon_fluxes = torch.stack(self.recon_fluxes).view(
                        self.dataset_length, 1, -1).detach().cpu().numpy()
                elif self.recon_spectra_all_bins:
                    self.recon_fluxes_all = torch.stack(self.recon_fluxes_all).view(
                        self.dataset_length, self.num_redshift_bins, -1).detach().cpu().numpy()

                if self.plot_gt_bin_spectra:
                    self.gt_bin_fluxes = torch.stack(self.gt_bin_fluxes).view(
                        self.dataset_length, 1, -1).detach().cpu().numpy()
                    self.gt_bin_spectra_losses = torch.stack(
                        self.gt_bin_spectra_losses).detach().cpu().numpy()
                if self.plot_optimal_wrong_bin_spectra:
                    self.optimal_wrong_bin_fluxes = torch.stack(
                        self.optimal_wrong_bin_fluxes
                    ).view(self.dataset_length, 1, -1).detach().cpu().numpy()
                    self.optimal_wrong_bin_spectra_losses = torch.stack(
                        self.optimal_wrong_bin_spectra_losses).detach().cpu().numpy()
        else:
            if self.main_infer:
                if self.recon_spectra_pixels_only:
                    # todo: adapt to patch-wise inferrence
                    num_spectra = self.dataset.get_num_validation_spectra()
                    val_spectra = self.dataset.get_validation_spectra()
                    self.gt_wave = val_spectra[:,0]
                    self.gt_fluxes = val_spectra[:,1]
                    self.gt_masks = self.dataset.get_validation_spectra_masks()
                else:
                    num_spectra = self.cur_patch.get_num_spectra()
                    self.gt_wave = self.cur_patch.get_spectra_pixel_wave()
                    self.gt_masks = self.cur_patch.get_spectra_pixel_masks()
                    self.gt_fluxes = self.cur_patch.get_spectra_pixel_fluxes()
            elif self.test:
                assert 0
                # todo: replace with patch-wise test spectra
                num_spectra = self.dataset.get_num_test_spectra()
                test_spectra = self.dataset.get_test_spectra()
                self.gt_wave = test_spectra[:,0]
                self.gt_fluxes = test_spectra[:,1]
                self.gt_masks = self.dataset.get_test_spectra_masks()
            else:
                raise ValueError()

            self.recon_wave = np.tile(
                self.dataset.get_full_wave(), num_spectra).reshape(num_spectra, -1)
            self.recon_masks = np.tile(
                self.dataset.get_full_wave_masks(), num_spectra).reshape(num_spectra, -1)
            self.recon_fluxes = torch.stack(self.recon_fluxes).view(
                num_spectra, self.neighbour_size**2, -1).detach().cpu().numpy()

        if self.save_redshift_pre:
            if self.classify_redshift:
                outlier_ids = self._log_redshift_residual_outlier(model_id)
                fname = join(self.redshift_dir, f"model-{model_id}_max_redshift.txt")
                log_data(self, "argmax_redshift", gt_field="gt_redshift",
                         fname=fname, log_to_console=False)
                fname = join(self.redshift_dir, f"model-{model_id}_avg_redshift.txt")
                log_data(self, "weighted_redshift", fname=fname, log_to_console=False)
            else:
                log_data(self, "redshift", gt_field="gt_redshift", log_to_console=False)

        if self.save_redshift_main:
            if self.classify_redshift:
                log_data(self, "argmax_redshift", gt_field="gt_redshift")
                log_data(self, "weighted_redshift")
            else:
                log_data(self, "redshift", gt_field="gt_redshift")

        if self.recon_spectra:
            if self.infer_outlier_only:
                self._recon_spectra(
                    num_spectra, model_id, suffix="-outlier", ids=outlier_ids)
            else: self._recon_spectra(num_spectra, model_id)

        if self.recon_spectra_all_bins:
            self._recon_spectra_all_bins(num_spectra, model_id)

        if self.plot_codebook_coeff_all_bins:
            self._plot_codebook_coeff_all_bins(num_spectra, model_id)

        if self.plot_codebook_coeff:
            if self.infer_outlier_only:
                self._plot_codebook_coefficients(
                    model_id, suffix="-outlier", ids=outlier_ids)
            else: self._plot_codebook_coefficients(model_id)

        if self.plot_redshift_logits:
            if self.infer_outlier_only:
                self._plot_redshift_logits(
                    model_id, suffix="-outlier", ids=outlier_ids)
            else: self._plot_redshift_logits(model_id)

        if self.plot_binwise_spectra_loss:
            if self.infer_outlier_only:
                self._plot_binwise_spectra_loss(
                    model_id, suffix="outlier", ids=outlier_ids)
            else: self._plot_binwise_spectra_loss(model_id)

        if self.save_optimal_bin_ids:
            if self.infer_outlier_only:
                self._save_optimal_bin_ids(
                    model_id, suffix="-outlier", ids=outlier_ids)
            else: self._save_optimal_bin_ids(model_id)

        if self.save_qtz_weights:
            fname = join(self.qtz_weights_dir, f"model-{model_id}.npy")
            log_data(self, "qtz_weights", fname=fname, log_to_console=False)

        if self.save_codebook_latents:
            self._save_codebook_latents(model_id)

        # if self.save_pixel_values:
        #     self.recon_pixels = self.trans_obj.integrate(recon_fluxes)
        #     if self.pretrain_infer:
        #         self.gt_pixels = self.dataset.get_supervision_spectra_pixels().numpy()
        #     else: self.gt_pixels = self.dataset.get_supervision_validation_pixels().numpy()
        #     self.gt_pixels = self.gt_pixels[:,0]
        #     log_data(self,
        #         "recon_pixels", gt_field="gt_pixels", log_ratio=self.log_pixel_ratio)

        log.info("== Spectra inferrence done for current checkpoint.")

    def pre_checkpoint_hardcode_coords_modified_model(self, model_id):
        self.reset_data_iterator()
        self.codebook_spectra = []
        if self.recon_codebook_spectra:
            self.spectra_wave_c = []
            self.spectra_masks_c = []
        if self.recon_codebook_spectra_individ and self.pretrain_infer:
            self.spectra_wave_ci = []
            self.spectra_masks_ci = []

    def run_checkpoint_hardcode_coords_modified_model(self, model_id, checkpoint):
        if self.recon_codebook_spectra:
            self._set_coords_from_checkpoint(checkpoint)
        elif self.recon_codebook_spectra_individ and self.pretrain_infer:
            self.codebook_spectra_infer_pipeline.set_latents(
                checkpoint["model_state_dict"]["nef.latents"])
            if not self.apply_gt_redshift and self.split_latent:
                self.codebook_spectra_infer_pipeline.set_redshift_latents(
                    checkpoint["model_state_dict"]["nef.redshift_latents"])

        self.infer_codebook_spectra(model_id, checkpoint)

    def post_checkpoint_hardcode_coords_modified_model(self, model_id):
        # [(num_supervision_spectra,)num_embeds,nsmpl]
        self.codebook_spectra = torch.stack(
            self.codebook_spectra).detach().cpu().numpy()

        if self.recon_codebook_spectra:
            spectra_wave = torch.stack(self.spectra_wave_c).view(
                self.dataset_length, -1).detach().cpu().numpy()
            spectra_masks = torch.stack(self.spectra_masks_c).bool().view(
                self.dataset_length, -1).detach().cpu().numpy()

            # if spectra is 2d, add dummy 1st dim to simplify code
            enum = zip([spectra_wave], [self.codebook_spectra], [spectra_masks])
            dir = self.codebook_spectra_dir
            fname = f"model-{model_id}_" + \
                str(self.extra_args["codebook_spectra_plot_wave_lo"]) + "_" + \
                str(self.extra_args["codebook_spectra_plot_wave_hi"])

        elif self.recon_codebook_spectra_individ:
            if self.pretrain_infer:
                spectra_wave = torch.stack(self.spectra_wave_ci).view(
                    self.dataset_length, -1).detach().cpu().numpy()
                spectra_masks = torch.stack(self.spectra_masks_ci).bool().view(
                    self.dataset_length, -1).detach().cpu().numpy()

                enum = zip(spectra_wave, self.codebook_spectra, spectra_masks)
                dir = self.codebook_spectra_individ_dir
                fname = f"model-{model_id}"
            else:
                self.codebook_spectra = self.codebook_spectra.reshape(
                    -1, self.neighbour_size**2, self.extra_args["qtz_num_embed"],
                    self.codebook_spectra.shape[-1]
                ).transpose(0,2,1,3) # [bsz,num_embed,n_neighbour,nsmpl]
                num_spectra = len(self.codebook_spectra)

                spectra_wave = np.tile(
                    self.dataset.get_full_wave(), num_spectra).reshape(num_spectra, -1)
                spectra_masks = np.tile(
                    self.dataset.get_full_wave_masks(), num_spectra).reshape(num_spectra, -1)

                enum = zip(spectra_wave, self.codebook_spectra, spectra_masks)
                dir = self.codebook_spectra_individ_dir
                fname = f"model-{model_id}"
        else:
            raise ValueError()

        for i, obj in enumerate(enum):
            (wave, codebook_spectra, masks) = obj
            if self.recon_codebook_spectra_individ:
                wave = np.tile(wave, self.qtz_n_embd).reshape(self.qtz_n_embd, -1)
                if masks is not None:
                    masks = np.tile(masks, self.qtz_n_embd).reshape(self.qtz_n_embd, -1)

            cur_dir = join(dir, f"spectra-{i}")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            self.dataset.plot_spectrum(
                cur_dir, fname, self.extra_args["flux_norm_cho"],
                None, None, wave, codebook_spectra,
                is_codebook=True,
                save_spectra_together=True,
                recon_masks=masks,
                clip=self.extra_args["plot_clipped_spectrum"]
            )

        log.info("== Codebook inferrence done for current checkpoint.")

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

        num_batches = int(np.ceil(self.dataset_length / self.batch_size))
        log.info(f"infer all coords, totally {num_batches} batches")

        for i in tqdm(range(num_batches)):
        # for i in range(num_batches):
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
                    index_latent=self.index_latent,
                    apply_gt_redshift=self.apply_gt_redshift,
                    perform_integration=self.perform_integration,
                    trans_sample_method=self.trans_sample_method,
                    save_scaler=self.save_scaler,
                    save_redshift=self.save_redshift,
                    save_embed_ids=self.plot_embed_map,
                    save_latents=self.plot_latent_embed,
                    save_qtz_weights=self.save_qtz_weights,
                    # save_redshift_logits=self.classify_redshift
                )

            if self.recon_img:
                # artifically generated transmission function (last channel)
                if self.recon_synthetic_band:
                    self.recon_synthetic_pixels.extend(ret["intensity"][...,-1:])
                    ret["intensity"] = ret["intensity"][...,:-1]
                if self.recon_img_all_pixels:
                    self.gt_pixels.extend(data["pixels"])
                elif self.recon_img_sup_spectra:
                    self.gt_pixels.extend(data["spectra_pixels"])
                elif self.recon_img_val_spectra:
                    self.gt_pixels.extend(data["spectra_pixels"])
                elif self.recon_img_test_spectra:
                    self.gt_pixels.extend(data["spectra_pixels"])
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
                self.gt_redshift.extend(data["spectra_semi_sup_redshift"])

                if self.classify_redshift:
                    ids = torch.argmax(ret["redshift_logits"], dim=-1)
                    argmax_redshift = ret["redshift"][ids]
                    self.argmax_redshift.extend(argmax_redshift)
                    weighted_redshift = torch.sum(
                        ret["redshift"] * ret["redshift_logits"], dim=-1)
                    self.weighted_redshift.extend(weighted_redshift)
                else:
                    self.redshift.extend(ret["redshift"])

            elif self.save_redshift_test:
                self.redshift.extend(ret["redshift"])
                self.gt_redshift.extend(data["spectra_redshift"])

        log.info("== All coords forward done.")

    def infer_spectra(self, model_id, checkpoint):
        iterations = checkpoint["epoch_trained"]
        model_state = checkpoint["model_state_dict"]

        load_model_weights(self.spectra_infer_pipeline, model_state)

        ## debug, test apply_gt_spectra in redshift_pretrain_infer
        # shared_layers = set(self.spectra_infer_pipeline.state_dict().keys())
        # to_remove = []
        # for k in shared_layers:
        #     if "redshift" in k: to_remove.append(k)
        # for k in to_remove: shared_layers.remove(k)
        # load_pretrained_model_weights(
        #     self.spectra_infer_pipeline, model_state, shared_layer_names=shared_layers
        # )
        ## ends here

        self.spectra_infer_pipeline.eval()

        while True:
            try:
                data = self.next_batch()
                add_to_device(data, self.extra_args["gpu_data"], self.device)

                if self.recon_spectra_all_bins:
                    self.spectra_infer_pipeline.set_batch_reduction_order("qtz_first")

                # if self.classify_redshift and \
                #    self.extra_args["redshift_classification_method"] == "bayesian_weighted_avg":
                #     self.spectra_infer_pipeline.set_bayesian_redshift_logits_calculation(
                #         get_loss(self.extra_args["spectra_loss_cho"], self.cuda),
                #         data["spectra_masks"], data["spectra_source_data"])

                if self.calculate_binwise_spectra_loss:
                    self.spectra_infer_pipeline.set_batch_reduction_order("qtz_first")
                    loss_func = self._get_spectra_loss_func(data)
                else: loss_func = None

                with torch.no_grad():
                    ret = forward(
                        data,
                        self.spectra_infer_pipeline,
                        iterations,
                        self.space_dim,
                        spectra_loss_func=loss_func,
                        qtz=self.qtz,
                        qtz_strategy=self.qtz_strategy,
                        index_latent=self.index_latent,
                        split_latent=self.split_latent,
                        apply_gt_redshift=self.apply_gt_redshift,
                        calculate_binwise_spectra_loss=self.calculate_binwise_spectra_loss,
                        save_redshift=self.save_redshift,
                        save_spectra=self.recon_spectra,
                        save_qtz_weights=self.save_qtz_weights,
                        save_optimal_bin_ids=self.save_optimal_bin_ids,
                        save_gt_bin_spectra=self.plot_gt_bin_spectra,
                        save_codebook_logits=self.plot_codebook_coeff or \
                                             self.plot_codebook_coeff_all_bins,
                        save_redshift_logits=self.plot_redshift_logits,
                        save_codebook_latents=self.save_codebook_latents,
                        save_spectra_all_bins=self.recon_spectra_all_bins or \
                                              self.plot_optimal_wrong_bin_spectra,
                        init_redshift_prob= None if "init_redshift_prob" not in data else \
                                                data["init_redshift_prob"])

                if self.recon_spectra or self.recon_spectra_all_bins:
                    self.gt_fluxes.extend(data["spectra_source_data"][:,1])
                    self.spectra_wave.extend(data["spectra_source_data"][:,0])
                    self.spectra_masks.extend(data["spectra_masks"])

                if self.recon_spectra:
                    fluxes = ret["intensity"]
                    if fluxes.ndim == 3: # bandwise
                        fluxes = fluxes.flatten(1,2) # [bsz,nsmpl]
                    self.recon_fluxes.extend(fluxes)
                    if self.plot_gt_bin_spectra:
                        self.gt_bin_fluxes.extend(ret["gt_bin_spectra"])
                        gt_bin_losses = self._get_gt_bin_spectra_losses(ret, data)
                        self.gt_bin_spectra_losses.extend(gt_bin_losses)
                    if self.plot_optimal_wrong_bin_spectra:
                        fluxes, losses = self._get_optimal_wrong_bin_fluxes(ret, data)
                        self.optimal_wrong_bin_fluxes.extend(fluxes)
                        self.optimal_wrong_bin_spectra_losses.extend(losses)

                if self.recon_spectra_all_bins:
                    self.recon_fluxes_all.extend(
                        ret["spectra_all_bins"].permute(1,0,2)
                    ) # [bsz,num_bins,nsmpl]

                if self.save_qtz_weights:
                    self.qtz_weights.extend(ret["qtz_weights"])

                if self.save_codebook_latents:
                    self.codebook_latents.extend(ret["codebook_latents"])

                if self.save_optimal_bin_ids:
                    self.gt_bin_ids.extend(data["gt_redshift_bin_ids"][1])
                    self.optimal_bin_ids.extend(ret["optimal_bin_ids"])
                    ids, _ = get_optimal_wrong_bin_ids(ret, data)
                    self.optimal_wrong_bin_ids.extend(ids)

                if self.plot_codebook_coeff:
                    self.gt_redshift_cl.extend(data["spectra_redshift"])
                    self.codebook_coeff.extend(ret["codebook_logits"])
                    if self.plot_optimal_wrong_bin_codebook_coeff:
                        ids, _ = get_optimal_wrong_bin_ids(ret, data)
                        self.optimal_wrong_bin_ids_cc.extend(ids)

                if self.plot_codebook_coeff_all_bins:
                    self.codebook_coeff_all_bins.extend(ret["codebook_logits"])

                if self.plot_redshift_logits:
                    self.gt_redshift_l.extend(data["spectra_redshift"])
                    self.redshift_logits.extend(ret["redshift_logits"])

                if self.plot_binwise_spectra_loss:
                    self.gt_redshift_bl.extend(data["spectra_redshift"])
                    self.binwise_loss.extend(ret["spectra_binwise_loss"])

                if self.save_redshift_pre:
                    self.gt_redshift.extend(data["spectra_redshift"])
                    if self.classify_redshift:
                        ids = torch.argmax(ret["redshift_logits"], dim=-1)
                        argmax_redshift = ret["redshift"][ids]
                        self.argmax_redshift.extend(argmax_redshift)
                        weighted_redshift = torch.sum(
                            ret["redshift"] * ret["redshift_logits"], dim=-1)
                        self.weighted_redshift.extend(weighted_redshift)
                    else:
                        self.redshift.extend(ret["redshift"])

                elif self.save_redshift_main:
                    self.gt_redshift.extend(data["spectra_semi_sup_redshift"])
                    if self.classify_redshift:
                        ids = torch.argmax(ret["redshift_logits"], dim=-1)
                        argmax_redshift = ret["redshift"][ids]
                        self.argmax_redshift.extend(argmax_redshift)
                        weighted_redshift = torch.sum(
                            ret["redshift"] * ret["redshift_logits"], dim=-1)
                        self.weighted_redshift.extend(weighted_redshift)
                    else:
                        self.redshift.extend(ret["redshift"])

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

        if self.recon_codebook_spectra_individ:
            self.codebook_spectra_infer_pipeline.set_batch_reduction_order("bin_avg_first")

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
                        index_latent=self.index_latent,
                        split_latent=self.split_latent,
                        apply_gt_redshift=self.apply_gt_redshift and \
                                          self.recon_codebook_spectra_individ,
                        save_spectra=self.recon_codebook_spectra,
                        save_codebook_spectra=self.recon_codebook_spectra_individ
                    )

                if self.recon_codebook_spectra:
                    spectra = ret["spectra"]
                elif self.recon_codebook_spectra_individ:
                    spectra = ret["codebook_spectra"]
                else: raise ValueError()
                self.codebook_spectra.extend(spectra)

                if self.recon_codebook_spectra:
                    self.spectra_wave_c.extend(data["wave"])
                    self.spectra_masks_c.extend(data["spectra_masks"])

                elif self.recon_codebook_spectra_individ:
                    if self.pretrain_infer:
                        self.spectra_wave_ci.extend(data["wave"])
                        self.spectra_masks_ci.extend(data["spectra_masks"])
                        # self.redshift.extend(data["spectra_redshift"])

                else: raise ValueError()

            except StopIteration:
                log.info("codebook spectra forward done")
                break

    ##########
    # Helpers
    ##########

    def _configure_dataset(self):
        """ Configure dataset (batched fields and len) for inferrence.
        """
        if self.space_dim == 3:
            self.requested_fields.extend(["wave_data"])

        self.dataset.set_mode(self.mode)
        self.dataset.set_length(self.dataset_length)
        self.dataset.set_fields(self.requested_fields)
        self.dataset.set_wave_source(self.wave_source)
        if self.coords_source is not None:
            self.dataset.set_coords_source(self.coords_source)
        self.dataset.toggle_integration(self.perform_integration)
        self.dataset.toggle_selected_inferrence(self.infer_selected)

        self.dataset.toggle_wave_sampling(not self.use_all_wave)
        if not self.use_all_wave:
            self.dataset.set_num_wave_samples(self.num_wave_samples)
            self.dataset.set_wave_sample_method(self.wave_sample_method)

        # select the same random set of spectra to recon
        if self.infer_selected:
            ids = self._select_inferrence_ids()
            self.dataset.set_hardcode_data("selected_ids", ids)

        if self.classify_redshift:
            self.dataset.set_num_redshift_bins(self.num_redshift_bins)

    @lru_cache
    def _select_inferrence_ids(self):
        if self.extra_args["spectra_inferrence_id_fname"] is not None and \
           exists(self.extra_args["spectra_inferrence_id_fname"]):
            ids = np.load(self.extra_args["spectra_inferrence_id_fname"])
            self.num_spectra = len(ids)
            log.info(f"infer with given ids, totally {self.num_spectra}")
        else:
            ids = select_inferrence_ids(
                self.num_spectra, self.extra_args["pretrain_num_infer_upper_bound"])
        return ids

    def _get_spectra_loss_func(self, data):
        if self.extra_args["spectra_loss_cho"] == "emd":
            loss = spectra_supervision_emd_loss
        else:
            loss = partial(spectra_supervision_loss,
                           get_loss(self.extra_args["spectra_loss_cho"], self.cuda))
        return loss

    def _set_coords_from_checkpoint(self, checkpoint):
        """ Set dataset coords using saved model checkpoint.
        """
        if self.coords_source == "codebook_latents":
            # latent code in codebook
            codebook_latents = load_layer_weights(
                checkpoint["model_state_dict"], lambda n: "grid" not in n and "codebook" in n)
            codebook_latents = codebook_latents[:,None] # [num_embd, 1, latent_dim]
            codebook_latents = codebook_latents.detach().cpu().numpy()
            self.dataset.set_hardcode_data(self.coords_source, codebook_latents)
        else:
            raise ValueError()

    def _set_coords_from_spectra_source(self, coords, w_neighbour=True):
        """ Set dataset coords using spectra coords.
            @Param
              coords: spectra coords [bsz,n_neighbours,2/3]
              w_neighbour: if True, we use all neighbouring coords as well
                           ow, we keep only the center coords
        """
        if w_neighbour:
            coords = coords.view(-1,1,coords.shape[-1])
        else:
            pixel_id = get_neighbourhood_center_pixel_id(self.neighbour_size)
            coords = coords[:,pixel_id:pixel_id+1] # [bsz,1,2/3]
        self.dataset.set_hardcode_data(self.coords_source, coords)
        self.dataset_length = len(coords)

    ######################
    # Data saving helpers
    ######################

    def _log_redshift_residual_outlier(self, model_id):
        """ Given redshift classification strategy, save and log residual of
              bins whose estimations are identified as outliers.
        """
        gt_redshift = torch.stack(self.gt_redshift).detach().cpu().numpy()
        argmax_redshift = torch.stack(self.argmax_redshift).detach().cpu().numpy()
        self.redshift_residual = argmax_redshift - gt_redshift
        fname = join(self.redshift_dir, f"model-{model_id}_redshift_residual.txt")
        log_data(self, "redshift_residual", fname=fname, log_to_console=False)

        ids = np.arange(len(self.redshift_residual))
        outlier = ids[np.abs(self.redshift_residual) > self.extra_args["redshift_bin_width"]]
        outlier_gt = gt_redshift[outlier]
        outlier_est = argmax_redshift[outlier]
        to_save = np.array(list(outlier) + list(outlier_gt) + list(outlier_est)).reshape(3,-1)
        log.info(f"outlier spectra: {outlier}")
        log.info(f"gt_redshift: {outlier_gt}")
        log.info(f"argmax_redshift: {outlier_est}")
        fname = join(self.redshift_dir, f"model-{model_id}_redshift_outlier.txt")
        with open(fname, "w") as f: f.write(f"{to_save}")

        fname = join(self.redshift_dir, f"model-{model_id}_redshift_outlier_ids")
        np.save(fname, outlier)
        return outlier

    def _save_optimal_bin_ids(self, model_id, suffix="", ids=None):
        """ Log and save the redshift bin ids of the optimum spectra.
        """
        self.gt_bin_ids = torch.stack(self.gt_bin_ids).detach().cpu().numpy()
        self.optimal_bin_ids = torch.stack(self.optimal_bin_ids).detach().cpu().numpy()
        self.optimal_wrong_bin_ids = torch.stack(self.optimal_wrong_bin_ids).detach().cpu().numpy()
        if ids is not None:
            self.gt_bin_ids = self.gt_bin_ids[ids]
            self.optimal_bin_ids = self.optimal_bin_ids[ids]
            self.optimal_wrong_bin_ids = self.optimal_wrong_bin_ids[ids]

        fname = join(self.codebook_latents_dir, f"model-{model_id}_logits")
        log_data(self, "optimal_bin_ids", gt_field="gt_bin_ids",
                 fname=fname, log_to_console=True)
        log_data(self, "optimal_wrong_bin_ids", fname=fname, log_to_console=True)

    def _save_codebook_latents(self, model_id):
        codebook_latents = torch.stack(self.codebook_latents).detach().cpu().numpy()
        fname = join(self.codebook_latents_dir, f"model-{model_id}_logits")
        np.save(fname, codebook_latents)

    def _plot_codebook_coefficients(self, model_id, suffix="", ids=None):
        """ Plot coefficient of each code in the codebook.
        """
        codebook_coeff = torch.stack(self.codebook_coeff).detach().cpu().numpy()
        if self.plot_optimal_wrong_bin_codebook_coeff:
            optimal_wrong_bin_ids = torch.stack(
                self.optimal_wrong_bin_ids_cc).detach().cpu().numpy()

        if ids is not None:
            codebook_coeff = codebook_coeff[ids]
            if self.plot_optimal_wrong_bin_codebook_coeff:
                optimal_wrong_bin_ids = optimal_wrong_bin_ids[ids]

        fname = join(self.codebook_coeff_dir, f"model-{model_id}_logits{suffix}")
        np.save(fname, codebook_coeff)

        y, y2 = codebook_coeff, None
        if self.optimize_codebook_latents_for_each_redshift_bin:
            # if each bin has its own set of codebook coeff, we plot that for gt bin only
            gt_redshift = torch.stack(self.gt_redshift_cl).detach().cpu().numpy()
            if ids is not None: gt_redshift = gt_redshift[ids]
            gt_bin_ids = get_bin_ids(
                self.extra_args["redshift_lo"],
                self.extra_args["redshift_bin_width"],
                gt_redshift, add_batched_dim=True
            )
            y = codebook_coeff[gt_bin_ids[0], gt_bin_ids[1]]
            if self.plot_optimal_wrong_bin_codebook_coeff:
                optimal_wrong_bin_ids = create_batch_ids(optimal_wrong_bin_ids)
                optimal_wrong_bin_codebook_coeff = codebook_coeff[
                    optimal_wrong_bin_ids[0], optimal_wrong_bin_ids[1]]
                y2 = optimal_wrong_bin_codebook_coeff

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            y, fname, y2=y2, hist=True)

    def _plot_redshift_logits(self, model_id, suffix="", ids=None):
        """ Plot logits for each redshift bin.
        """
        gt_redshift = torch.stack(self.gt_redshift_l).detach().cpu().numpy()
        redshift_logits = torch.stack(self.redshift_logits).detach().cpu().numpy()
        if ids is not None:
            gt_redshift = gt_redshift[ids]
            redshift_logits = redshift_logits[ids]

        bin_centers = init_redshift_bins(
            self.extra_args["redshift_lo"], self.extra_args["redshift_hi"],
            self.extra_args["redshift_bin_width"])

        # n, nbins = redshift_logits.shape
        # gt_bin_ids = np.array([
        #     get_bin_id(self.extra_args["redshift_lo"],
        #                self.extra_args["redshift_bin_width"], val
        #     ) for val in gt_redshift
        # ])[None,:]
        # indices = np.arange(n)[None,:]
        # gt_bin_ids = np.concatenate((indices, gt_bin_ids), axis=0)
        # gt_logits = np.zeros(redshift_logits.shape)
        # gt_logits[gt_bin_ids[0,:], gt_bin_ids[1,:]] = 0.1

        if self.calculate_binwise_spectra_loss:
            sub_dir = join(
                self.redshift_dir, "beta-"+str(self.extra_args["binwise_loss_beta"])+suffix)
        else: sub_dir = join(self.redshift_dir, suffix)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)
        fname = join(sub_dir, f"model-{model_id}_logits")
        np.save(fname, np.concatenate((bin_centers[None,:], redshift_logits), axis=0))

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            redshift_logits, fname, x=bin_centers,
            vertical_xs=gt_redshift
        ) #,y2=gt_logits)

        if self.extra_args["plot_redshift_precision_recall"]:
            plot_precision_recall_all(
                redshift_logits, gt_redshift, self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"], self.extra_args["redshift_bin_width"],
                self.extra_args["num_spectrum_per_row"], f"{fname}_precision_recall.png")

        if self.extra_args["plot_redshift_precision_recall_together"]:
            plot_precision_recall_single(
                redshift_logits, gt_redshift, self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"], self.extra_args["redshift_bin_width"],
                f"{fname}_precision_recall.png")

        log.info("redshift logits plotting done")

    def _plot_binwise_spectra_loss(self, model_id, suffix="", ids=None):
        """ Plot reconstruction loss for spectra corresponding to each redshift bin.
        """
        losses = torch.stack(self.binwise_loss).detach().cpu().numpy()
        gt_redshift = torch.stack(self.gt_redshift_bl).detach().cpu().numpy()
        if ids is not None:
            losses = losses[ids]
            gt_redshift = gt_redshift[ids]

        bin_centers = init_redshift_bins(
            self.extra_args["redshift_lo"], self.extra_args["redshift_hi"],
            self.extra_args["redshift_bin_width"])

        sub_dir = join(self.redshift_dir, suffix)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)
        fname = join(sub_dir, f"model-{model_id}_losses")
        np.save(fname, np.concatenate((bin_centers[None,:], losses), axis=0))

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            losses, fname, x=bin_centers,vertical_xs=gt_redshift)

    def _plot_redshift_map(self, model_id):
        if self.extra_args["mark_spectra"]:
            positions = self.cur_patch.get_spectra_img_coords()
            markers = np.array(self.extra_args["spectra_markers"])
        else:
            positions, markers = [], []
        plot_annotated_heat_map = partial(annotated_heat, positions, markers)

        re_args = {
            "fname": f"model-{model_id}",
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
        # plot redshift img
        _, _ = self.dataset.restore_evaluate_tiles(self.redshift, **re_args)

    def _recon_spectra(self, num_spectra, model_id, suffix="", ids=None):
        """ Plot spectrum in multiple figures, each figure contains several spectrum.
        """
        titles = np.char.mod("%d", np.arange(num_spectra))

        if ids is not None:
            titles = titles[ids]
            self.gt_wave = self.gt_wave[ids]
            self.gt_masks = self.gt_masks[ids]
            self.gt_fluxes = self.gt_fluxes[ids]
            self.recon_wave = self.recon_wave[ids]
            self.recon_masks = self.recon_masks[ids]
            self.recon_fluxes = self.recon_fluxes[ids]
            if self.plot_gt_bin_spectra:
                self.gt_bin_fluxes = self.gt_bin_fluxes[ids]
                self.gt_bin_spectra_losses = self.gt_bin_spectra_losses[ids]
            if self.plot_optimal_wrong_bin_spectra:
                self.optimal_wrong_bin_fluxes = self.optimal_wrong_bin_fluxes[ids]
                self.optimal_wrong_bin_spectra_losses = self.optimal_wrong_bin_spectra_losses[ids]
            num_spectra = len(ids)

        n_spectrum_per_fig = self.extra_args["num_spectrum_per_fig"]
        n_figs = int(np.ceil(num_spectra / n_spectrum_per_fig))

        cur_checkpoint_metrics = []
        for i in range(n_figs):
            fname = f"model-{model_id}-plot{i}{suffix}"
            lo = i * n_spectrum_per_fig
            hi = min(lo + n_spectrum_per_fig, num_spectra)

            if self.plot_gt_bin_spectra:
                recon_fluxes2 = self.gt_bin_fluxes[lo:hi]
                recon_losses2 = self.gt_bin_spectra_losses[lo:hi]
            else: recon_fluxes2, recon_losses2 = None, None
            if self.plot_optimal_wrong_bin_spectra:
                recon_fluxes3 = self.optimal_wrong_bin_fluxes[lo:hi]
                recon_losses3 = self.optimal_wrong_bin_spectra_losses[lo:hi]
            else: recon_fluxes3, recon_losses3 = None, None

            if self.infer_selected:
                n = len(self._select_inferrence_ids())
                spectra_dir = join(self.spectra_dir, f"selected-{n}")
                Path(spectra_dir).mkdir(parents=True, exist_ok=True)
            else: spectra_dir = self.spectra_dir

            cur_metrics = self.dataset.plot_spectrum(
                spectra_dir, fname,
                self.extra_args["flux_norm_cho"],
                self.gt_wave[lo:hi], self.gt_fluxes[lo:hi],
                self.recon_wave[lo:hi], self.recon_fluxes[lo:hi],
                recon_fluxes2=recon_fluxes2, recon_losses2=recon_losses2,
                recon_fluxes3=recon_fluxes3, recon_losses3=recon_losses3,
                clip=self.extra_args["plot_clipped_spectrum"],
                gt_masks=self.gt_masks[lo:hi],
                recon_masks=self.recon_masks[lo:hi],
                calculate_metrics=not self.infer_outlier_only,
                titles=titles
            )
            if cur_metrics is not None:
                cur_checkpoint_metrics.extend(cur_metrics)

        if len(cur_checkpoint_metrics) != 0:
            self.metrics.append(cur_checkpoint_metrics)

        log.info("spectrum plotting done")

    def _recon_spectra_all_bins(self, num_spectra, model_id):
        """ Plot spectrum under all redshift for each spectra
        """
        recon_fluxes_all = self.recon_fluxes_all.transpose(1,0,2) # [num_bins,bsz,nsmpl]
        assert num_spectra == recon_fluxes_all.shape[1]

        num_bins = recon_fluxes_all.shape[0]
        n_spectrum_per_fig = self.extra_args["num_spectrum_per_fig"]
        n_figs_each = int(np.ceil(num_bins / n_spectrum_per_fig))
        redshift_bins = init_redshift_bins(
            self.extra_args["redshift_lo"], self.extra_args["redshift_hi"],
            self.extra_args["redshift_bin_width"]).numpy()

        def calculate_binwise_loss(gt_fluxes, recon_fluxes, masks, i):
            mask = torch.FloatTensor(masks[i]).to('cuda:0')
            gt_fluxes = torch.FloatTensor(gt_fluxes[i]).to('cuda:0')
            recon_fluxes = torch.FloatTensor(recon_fluxes[:,i]).to('cuda:0')
            losses = [F.mse_loss(recon*mask, gt_fluxes*mask, reduction="sum").item()
                      for recon in recon_fluxes]
            return np.array(losses)

        def calculate_loss(i):
            redshift_logits = torch.stack(self.redshift_logits).detach().cpu().numpy()[i]
            # plt.plot(redshift_logits);plt.savefig('tmp.png');plt.close()
            recon_fluxes = redshift_logits @ recon_fluxes_all[:,i]
            # plt.plot(recon_fluxes*self.recon_masks[i]);
            # plt.plot(self.gt_fluxes[i]*self.recon_masks[i]);
            # plt.savefig('tmp_.png'); plt.close()
            loss = F.mse_loss(torch.FloatTensor(recon_fluxes*self.recon_masks[i]),
                              torch.FloatTensor(self.gt_fluxes[i]*self.recon_masks[i]),
                              reduction="sum")
            return loss.item()

        def change_shape(data, m):
            return np.tile(data, m).reshape(m, -1)

        spectra_ids = np.arange(self.dataset_length)
        if self.infer_selected:
            n = len(self._select_inferrence_ids())
            spectra_dir = join(self.spectra_dir, f"selected-{n}")
        else: spectra_dir = self.spectra_dir

        for i in spectra_ids:
            cur_dir = join(spectra_dir, f"{i}-all-bins")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            titles = redshift_bins

            # calculate spectra loss under each redshift bin
            if self.calculate_binwise_spectra_loss:
                # loss = calculate_loss(i); print(loss) #;assert 0
                losses = calculate_binwise_loss(
                    self.gt_fluxes, recon_fluxes_all, self.recon_masks, i)
                fname = join(cur_dir, f"bin_wise_spectra_loss-model-{model_id}-spectra{i}")
                np.save(fname, losses)

                titles = np.concatenate((titles[:,None], losses[:,None]), axis=-1)
                titles = [
                    f"{title[0]:.{3}f}: {title[1]:.{3}f}" for title in titles
                ]

            for j in range(n_figs_each):
                fname = f"model-{model_id}-plot{j}-all_bins"
                lo = j * n_spectrum_per_fig
                hi = min(lo + n_spectrum_per_fig, num_bins)
                m = hi - lo

                _ = self.dataset.plot_spectrum(
                    cur_dir, fname,
                    self.extra_args["flux_norm_cho"],
                    change_shape(self.gt_wave[i], m),
                    change_shape(self.gt_fluxes[i], m),
                    change_shape(self.recon_wave[i], m),
                    recon_fluxes_all[lo:hi,i],
                    # save_spectra_together=True,
                    clip=self.extra_args["plot_clipped_spectrum"],
                    gt_masks=change_shape(self.gt_masks[i], m),
                    recon_masks=change_shape(self.recon_masks[i], m),
                    calculate_metrics=False,
                    titles=titles[lo:hi]
                )

        log.info("all bin spectrum plotting done")

    def _plot_codebook_coeff_all_bins(self, num_spectra, model_id):
        """ Plot spectrum under all redshift for each spectra
        """
        codebook_coeff = torch.stack(
            self.codebook_coeff_all_bins
        ).permute(1,0,2).detach().cpu().numpy() # [num_bins,bsz,nsmpl]
        assert num_spectra == codebook_coeff.shape[1]

        num_bins = codebook_coeff.shape[0]
        n_spectrum_per_fig = self.extra_args["num_spectrum_per_fig"]
        n_figs_each = int(np.ceil(num_bins / n_spectrum_per_fig))
        redshift_bins = init_redshift_bins(
            self.extra_args["redshift_lo"], self.extra_args["redshift_hi"],
            self.extra_args["redshift_bin_width"]).numpy()

        spectra_ids = np.arange(self.dataset_length)
        if self.infer_selected:
            n = len(self._select_inferrence_ids())
            codebook_coeff_dir = join(self.codebook_coeff_dir, f"selected-{n}")
        else: codebook_coeff_dir = self.codebook_coeff_dir


        for i in spectra_ids:
            cur_dir = join(codebook_coeff_dir, f"{i}-all-bins")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            titles = redshift_bins

            for j in range(n_figs_each):
                fname = join(cur_dir, f"model-{model_id}-plot{j}-all_bins")
                lo = j * n_spectrum_per_fig
                hi = min(lo + n_spectrum_per_fig, num_bins)
                m = hi - lo

                plot_multiple(
                    self.extra_args["num_spectrum_per_fig"],
                    self.extra_args["num_spectrum_per_row"],
                    codebook_coeff[lo:hi,i], fname, hist=True,
                    titles=titles[lo:hi]
                )

        log.info("all bin codebook coeff plotting done")

    def _get_gt_bin_spectra_losses(self, ret, data):
        all_bin_losses = ret["spectra_binwise_loss"] # [bsz,nbins]
        bsz = len(all_bin_losses)
        gt_bin_losses = all_bin_losses[data["gt_redshift_bin_masks"]]
        return gt_bin_losses

    def _get_optimal_wrong_bin_fluxes(self, ret, data):
        ids, optimal_wrong_bin_losses = get_optimal_wrong_bin_ids(ret, data)
        ids = create_batch_ids(ids.detach().cpu().numpy())
        fluxes = ret["spectra_all_bins"] # [nbins,bsz,nsmpl]
        fluxes = fluxes[ids[1],ids[0]]
        return fluxes, optimal_wrong_bin_losses

    def configure_img_metrics(self):
        self.metric_options = self.extra_args["metric_options"]
        self.num_metrics = len(self.metric_options)
        self.calculate_metrics = self.recon_img_all_pixels \
            and self.metric_options is not None

        if self.calculate_metrics:
            num_patches = self.dataset.get_num_patches()
            self.metrics = np.zeros((self.num_metrics, 0, num_patches, self.num_bands))
            self.metrics_zscale = np.zeros((
                self.num_metrics, 0, num_patches, self.num_bands))
            self.metric_fnames = [ join(self.metric_dir, f"img_{option}.npy")
                                   for option in self.metric_options ]
            self.metric_fnames_z = [ join(self.metric_dir, f"img_{option}_zscale.npy")
                                     for option in self.metric_options ]

    def configure_spectra_metrics(self):
        self.metric_options = self.extra_args["spectra_metric_options"]
        self.num_metrics = len(self.metric_options)
        self.metrics = []
