# Configuration file reference

AnimalPoseTracker reads configuration from three layers. Every value has one
home, and the layers only decide "default vs project vs this command":

1. **Package defaults** -- shipped with the package and never edited by hand:
   `animalposetracker/cfg/training.yaml` (training) and
   `animalposetracker/cfg/runtime_defaults.yaml` (everything else).
2. **The project's files** -- carried by each project and edited freely:
   - `project.yaml` -- identity (name, worker, model type/scale, date, sources).
   - `configs/dataset.yaml` -- the data and its shape (classes, keypoints,
     skeleton, flip table, OKS sigmas). The shape's single home.
   - `configs/model.yaml` -- the model's architecture graph.
   - `configs/runtime.yaml` -- the project's runtime parameters, sectioned by
     flow: `common` / `train` / `predict` / `export`. A snapshot: it carries
     every registered key with the value the project uses.
3. **This run** -- command-line flags (or the GUI's own controls) beat
   everything; the same key name is used at every layer.

The `infer` pipeline is deliberately independent: it reads no project files,
only command-line arguments and the artifact's sidecar metadata, so an
exported model can be taken to any machine as-is.

Resolution and its rules live in `animalposetracker/config/` (registry,
defaults, resolver, writer); nothing else reads or writes the project YAMLs.

## Adding or changing a key

1. Register it in `animalposetracker/config/schema.py` (name, scope, group,
   type, apply level, GUI visibility, one-line note) and list the configure
   forms it belongs to in `views` (`train` / `evaluate` / `infer` / `export`)
   -- the GUI's parameter trees are generated from that annotation, so a key
   with `views` shows up on those forms with no other change.
2. Give it a default in `cfg/training.yaml` or `cfg/runtime_defaults.yaml`.
3. Run the tests (`tests/test_config_schema.py` checks the two stay aligned)
   and refresh this document:
   `python -m animalposetracker.config.docs --write docs/pytorch/pytorch_config.md`.

New projects receive the key automatically; existing test-stage projects are
rebuilt rather than migrated (see the refactor plan, 2026-10-14).

## Reference

<!-- BEGIN GENERATED REFERENCE -->

This reference is generated from the key registry (`animalposetracker/config/schema.py`) and the defaults files (`cfg/training.yaml`, `cfg/runtime_defaults.yaml`). Do not edit it by hand -- run `python -m animalposetracker.config.docs --write docs/pytorch/pytorch_config.md` after changing the registry.

### Scope `train`

**`train.shared.runtime`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `model` | str | — | build | hidden | Model graph path (derived from the project layout, not persisted) |
| `data` | str | — | build | hidden | Dataset path (derived from the project layout, not persisted) |
| `num_workers` | int | 8 | build | form | Data loading worker count |
| `time_hours` | float | `null` | build | advanced | Training time cap in hours; overrides epochs when set |
| `cache` | bool / str | False | build | form | Data cache: false / true (ram) / ram / disk |
| `fraction` | float | 1.0 | build | form | Fraction of the training set to use |
| `profile` | bool | False | build | advanced | Per-stage timing profile |
| `plots` | bool | True | build | form | Save training curve plots |
| `validation_enabled` | bool | True | build | form | Run validation during training |
| `seed` | int | 0 | build | form | Random seed |
| `deterministic` | bool | True | build | advanced | Deterministic algorithms |
| `save` | bool | True | build | form | Save checkpoints |
| `save_period` | int | 0 | build | advanced | Save an epoch checkpoint every N epochs; 0 = off |
| `pretrained` | bool | True | build | form | Start from pretrained weights |
| `pretrained_weights` | str | `null` | build | advanced | Explicit pretrained weights path |
| `verbose` | bool | True | build | advanced | Verbose logging |
| `resume` | bool / str | False | build | form | Resume from the latest checkpoint |
| `resume_from` | str | `null` | build | hidden | Resume from a specific full checkpoint |
| `channels_last` | bool | `null` | build | advanced | Explicit channels_last switch; null = auto |

**`train.models.training`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `gradient_accumulation_steps` | int | `null` / 1 (RTPose / ViTPose) | build | advanced | Gradient accumulation steps |
| `gradient_clip_norm` | float | 10.0 / 1.0 (RTPose / ViTPose) | build | advanced | Gradient clipping norm; null = off |
| `lr_milestones` | list | `null` / `[460, 490]` (RTPose / ViTPose) | build | advanced | MultiStep learning-rate milestones |
| `validation_interval` | int | 1 / 10 (RTPose / ViTPose) | build | advanced | Run validation every N epochs |
| `log_interval` | int | 1 / 50 (RTPose / ViTPose) | build | advanced | Log every N batches |
| `epochs` | int | 1000 / 500 (RTPose / ViTPose) | build | form | Training epochs |
| `batch_size` | int | 16 / 64 (RTPose / ViTPose) | build | form | Training batch size ('auto' on the training CLI probes the largest batch that fits 60% of the GPU memory) |
| `validation_batch_size` | int | `null` / 32 (RTPose / ViTPose) | build | advanced | Validation batch size |
| `image_size` | int | 640 / 256 (RTPose / ViTPose) | build | form | Model input size (a model property; follows the model recipe) |
| `optimizer` | str | `auto` / `AdamW` (RTPose / ViTPose) | build | form | Optimizer (auto, or a concrete name) |
| `learning_rate` | float | 0.01 / 0.0005 (RTPose / ViTPose) | build | form | Initial learning rate |
| `final_lr_factor` | float | 0.01 / — (RTPose / ViTPose) | build | form | Final learning-rate factor |
| `momentum` | float | 0.937 / 0.9 (RTPose / ViTPose) | build | form | SGD momentum / Adam beta1 |
| `weight_decay` | float | 0.0005 / 0.1 (RTPose / ViTPose) | build | form | Weight decay |
| `warmup_epochs` | float | 3.0 / — (RTPose / ViTPose) | build | form | Warmup epochs |
| `warmup_iters` | int | — / 500 (RTPose / ViTPose) | build | advanced | Warmup iterations (alternative to warmup_epochs) |
| `warmup_start_factor` | float | — / 0.001 (RTPose / ViTPose) | build | advanced | Warmup start learning-rate factor |
| `warmup_momentum` | float | 0.8 / — (RTPose / ViTPose) | build | advanced | Warmup start momentum |
| `warmup_bias_lr` | float | 0.1 / — (RTPose / ViTPose) | build | advanced | Warmup learning rate for the bias group |
| `nominal_batch_size` | int | 64 / — (RTPose / ViTPose) | build | advanced | Nominal batch size (used to scale the learning rate) |
| `cosine_schedule` | bool | False / — (RTPose / ViTPose) | build | form | Cosine learning-rate schedule |
| `amp` | bool | True / False (RTPose / ViTPose) | build | form | Mixed-precision training |
| `ema` | bool | True / False (RTPose / ViTPose) | build | form | Exponential moving average |
| `close_mosaic` | int | 10 / — (RTPose / ViTPose) | build | form | Disable mosaic for the last N epochs |
| `layer_decay_rate` | float | `null` / — (RTPose / ViTPose) | build | advanced | ViT layer decay rate |
| `seed` | int | — / 21 (RTPose / ViTPose) | build | advanced | Recipe-level random seed (beats shared.runtime.seed) |

**`train.shared.training`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `lr_gamma` | float | 0.1 | build | advanced | Learning-rate decay at milestones |
| `best_metric` | str | `COCO/AP` | build | advanced | Best-checkpoint metric (COCO/AP or YOLO/AP) |
| `best_metric_mode` | str | `max` | build | advanced | Selection direction: max / min |
| `early_stopping_patience` | int | 300 | build | form | Early-stopping patience in epochs |
| `single_cls` | bool | False | build | advanced | Train multi-class data as a single class |
| `multi_scale` | bool | False | build | advanced | Multi-scale jittering |
| `freeze` | int / list | `null` | build | advanced | Freeze the first n layers, or explicit layer indices |

**`train.models.loss`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `box_loss_weight` | float | 7.5 / — (RTPose / ViTPose) | build | form | Box regression loss weight |
| `class_loss_weight` | float | 0.5 / — (RTPose / ViTPose) | build | form | Classification loss weight |
| `distribution_loss_weight` | float | 1.5 / — (RTPose / ViTPose) | build | form | DFL distribution loss weight |
| `keypoint_loss_weight` | float | 12.0 / — (RTPose / ViTPose) | build | form | Keypoint loss weight |
| `visibility_loss_weight` | float | 1.0 / — (RTPose / ViTPose) | build | form | Visibility loss weight |
| `label_sigma` | float | — / 6.0 (RTPose / ViTPose) | build | advanced | SimCC label Gaussian sigma |
| `kl_beta` | float | — / 1.0 (RTPose / ViTPose) | build | advanced | SimCC KL loss beta |

**`train.models.augmentation`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `hsv_h` | float | 0.015 / — (RTPose / ViTPose) | build | advanced | Hue augmentation gain |
| `hsv_s` | float | 0.7 / — (RTPose / ViTPose) | build | advanced | Saturation augmentation gain |
| `hsv_v` | float | 0.4 / — (RTPose / ViTPose) | build | advanced | Value augmentation gain |
| `degrees` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | Rotation augmentation (+/- degrees) |
| `translate` | float | 0.1 / — (RTPose / ViTPose) | build | advanced | Translation augmentation fraction |
| `scale` | float | 0.5 / — (RTPose / ViTPose) | build | advanced | Scale augmentation gain |
| `shear` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | Shear augmentation (+/- degrees) |
| `perspective` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | Perspective augmentation |
| `flipud` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | Vertical flip probability |
| `fliplr` | float | 0.5 / — (RTPose / ViTPose) | build | advanced | Horizontal flip probability |
| `bgr` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | BGR channel-swap probability |
| `mosaic` | float | 1.0 / — (RTPose / ViTPose) | build | advanced | Mosaic probability |
| `mixup` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | MixUp probability |
| `horizontal_flip_probability` | float | — / 0.5 (RTPose / ViTPose) | build | advanced | Horizontal flip probability |
| `half_body_probability` | float | — / 0.3 (RTPose / ViTPose) | build | advanced | Half-body crop probability |
| `half_body_upper_probability` | float | — / 0.7 (RTPose / ViTPose) | build | advanced | Probability that a half-body crop keeps the upper body |
| `half_body_scale_factor` | float | — / 1.5 (RTPose / ViTPose) | build | advanced | Half-body crop scale factor |
| `bbox_shift_probability` | float | — / 0.3 (RTPose / ViTPose) | build | advanced | Box shift probability |
| `bbox_shift_factor` | float | — / 0.16 (RTPose / ViTPose) | build | advanced | Box shift factor |
| `bbox_scale_factor` | float | — / 0.5 (RTPose / ViTPose) | build | advanced | Box scale factor |
| `bbox_rotation_probability` | float | — / 0.6 (RTPose / ViTPose) | build | advanced | Box rotation probability |
| `bbox_rotation_factor` | float | — / 80.0 (RTPose / ViTPose) | build | advanced | Box rotation factor (degrees) |

**`train.models.validation`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `confidence` | float | 0.001 / — (RTPose / ViTPose) | build | advanced | Validation confidence threshold |
| `iou` | float | 0.7 / — (RTPose / ViTPose) | build | advanced | Validation NMS IoU |
| `max_det` | int | 300 / — (RTPose / ViTPose) | build | advanced | Validation max detections per image |
| `agnostic_nms` | bool | False / — (RTPose / ViTPose) | build | advanced | Validation class-agnostic NMS |
| `oks_nms_threshold` | float | — / 0.9 (RTPose / ViTPose) | build | advanced | AnimalViTPose validation OKS-NMS threshold |
| `keypoint_score_threshold` | float | — / 0.2 (RTPose / ViTPose) | build | advanced | AnimalViTPose validation keypoint filter threshold |
| `flip_tta` | bool | — / False (RTPose / ViTPose) | build | advanced | AnimalViTPose validation flip TTA |

**`train.metrics`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `oks_area_mode` | str | `annotation` | build | advanced | OKS area basis: annotation / bbox |
| `pck_threshold` | float | 0.05 | build | advanced | PCK threshold (as a fraction of the normalized distance) |
| `auc_norm_factor` | float | 30.0 | build | advanced | AUC normalization factor |
| `auc_thresholds` | int | 20 | build | advanced | AUC threshold sample count |
| `coco_max_detections` | int | 20 | build | advanced | COCO evaluation max detections per image |

### Scope `common`

**`common`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `device` | str | `auto` | build | form | Device (auto, or a concrete device name) |
| `project` | str | `results` | build | form | Run output root |
| `name` | str | `train` | build | form | Run name (output goes under project/name) |
| `exist_ok` | bool | False | build | form | Allow writing into an existing output directory |
| `selected_weights` | str | `null` | build | hidden | Explicitly selected weights or artifact path for val/predict/export (the GUI's model picker) |

### Scope `predict`

**`predict`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `source` | str | `null` | build | form | Default input source (image / video / directory / glob) |
| `confidence` | float | 0.25 | live | form | Prediction confidence threshold |
| `iou` | float | 0.7 | live | form | Prediction NMS IoU |
| `max_det` | int | 300 | live | form | Max detections per image |
| `agnostic_nms` | bool | False | live | form | Class-agnostic NMS |
| `keypoint_confidence` | float | 0.25 | live | form | Keypoint draw/filter threshold |
| `dnn` | bool | False | build | advanced | Infer through the OpenCV DNN backend |
| `show_labels` | bool | True | live | form | Draw class labels |
| `show_conf` | bool | True | live | form | Draw confidence |
| `show_boxes` | bool | True | live | form | Draw detection boxes |
| `show_keypoints` | bool | True | live | form | Draw keypoints |
| `show_skeletons` | bool | False | live | form | Draw skeleton lines |
| `line_width` | int | 2 | live | form | Box/skeleton line width |
| `point_radius` | int | 4 | live | form | Keypoint radius |

**`predict.tracking`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `enabled` | bool | False | build | form | Cross-frame tracking switch |
| `algorithm` | str | `bytetrack` | build | form | Tracking algorithm (six options, e.g. bytetrack) |

**`predict.pose_filter`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `method` | str | `none` | run_end | form | Temporal smoothing method: none / median / savgol |
| `window_length` | int | 5 | run_end | form | Smoothing window length in frames |
| `polyorder` | int | 2 | run_end | form | Savitzky-Golay polynomial order |
| `max_gap` | int | 0 | run_end | form | Maximum missing-frame gap to interpolate |
| `confidence_threshold` | float | `null` | run_end | form | Filter confidence threshold; null = inherit keypoint_confidence |

### Scope `export`

**`export`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `format` | str | `null` | build | form | Export format (onnx / engine / rknn / ...) |
| `half` | bool | False | build | form | FP16 export |
| `dynamic` | bool | False | build | advanced | Dynamic axes / dynamic batch |
| `simplify` | bool | True | build | advanced | ONNX graph simplification (onnxslim) |
| `opset` | int | 17 | build | advanced | ONNX opset version |
| `workspace` | float | 4.0 | build | advanced | TensorRT workspace (GiB) |
| `rknn_target` | str | `rk3588` | build | advanced | Rockchip NPU target (e.g. rk3588) |
| `optimize` | bool | False | build | advanced | TorchScript mobile optimization |
| `int8` | bool | False | build | advanced | INT8 quantization |
| `keras` | bool | False | build | hidden | Keras export (errors when unsupported) |
| `nms` | bool | False | build | hidden | Embed NMS into the graph (errors when unsupported) |

<!-- END GENERATED REFERENCE -->
