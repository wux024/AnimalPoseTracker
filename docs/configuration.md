# Configuration

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
   type, apply level, GUI visibility, one-line note).
2. Give it a default in `cfg/training.yaml` or `cfg/runtime_defaults.yaml`.
3. Run the tests (`tests/test_config_schema.py` checks the two stay aligned)
   and refresh this document:
   `python -m animalposetracker.config.docs --write docs/configuration.md`.

New projects receive the key automatically; existing test-stage projects are
rebuilt rather than migrated (see the refactor plan, 2026-10-14).

## Reference

<!-- BEGIN GENERATED REFERENCE -->

This reference is generated from the key registry (`animalposetracker/config/schema.py`) and the defaults files (`cfg/training.yaml`, `cfg/runtime_defaults.yaml`). Do not edit it by hand -- run `python -m animalposetracker.config.docs --write docs/configuration.md` after changing the registry.

### Scope `train`

**`train.shared.runtime`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `model` | str | — | build | hidden | 模型图路径（由项目结构推导，不落盘） |
| `data` | str | — | build | hidden | 数据集路径（由项目结构推导，不落盘） |
| `num_workers` | int | 8 | build | form | 数据加载线程数 |
| `time_hours` | float | `null` | build | advanced | 训练时间上限（小时），给出后覆盖 epochs |
| `cache` | bool / str | False | build | form | 数据缓存：false / true(ram) / ram / disk |
| `fraction` | float | 1.0 | build | form | 训练集使用比例 |
| `profile` | bool | False | build | advanced | 逐阶段耗时剖析 |
| `plots` | bool | True | build | form | 保存训练曲线图 |
| `validation_enabled` | bool | True | build | form | 训练时是否跑验证 |
| `seed` | int | 0 | build | form | 随机种子 |
| `deterministic` | bool | True | build | advanced | 确定性算法 |
| `save` | bool | True | build | form | 保存 checkpoint |
| `save_period` | int | 0 | build | advanced | 每 N 轮另存 epoch checkpoint；0 = 关闭 |
| `pretrained` | bool | True | build | form | 使用预训练权重 |
| `pretrained_weights` | str | `null` | build | advanced | 显式预训练权重路径 |
| `verbose` | bool | True | build | advanced | 详细日志 |
| `resume` | bool / str | False | build | form | 从 last checkpoint 继续训练 |
| `resume_from` | str | `null` | build | hidden | 从指定完整 checkpoint 恢复 |
| `channels_last` | bool | `null` | build | advanced | channels_last 显式开关；null = 自动 |

**`train.models.training`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `gradient_accumulation_steps` | int | `null` / 1 (RTPose / ViTPose) | build | advanced | 梯度累积步数 |
| `gradient_clip_norm` | float | 10.0 / 1.0 (RTPose / ViTPose) | build | advanced | 梯度裁剪范数；null = 关闭 |
| `lr_milestones` | list | `null` / `[460, 490]` (RTPose / ViTPose) | build | advanced | MultiStep 学习率里程碑 |
| `validation_interval` | int | 1 / 10 (RTPose / ViTPose) | build | advanced | 每 N 轮验证一次 |
| `log_interval` | int | 1 / 50 (RTPose / ViTPose) | build | advanced | 每 N 个 batch 打日志 |
| `epochs` | int | 1000 / 500 (RTPose / ViTPose) | build | form | 训练轮数 |
| `batch_size` | int | 16 / 64 (RTPose / ViTPose) | build | form | 训练批大小 |
| `validation_batch_size` | int | `null` / 32 (RTPose / ViTPose) | build | advanced | 验证批大小 |
| `image_size` | int | 640 / 256 (RTPose / ViTPose) | build | form | 模型输入尺寸（模型属性：跟随模型配方） |
| `optimizer` | str | `auto` / `AdamW` (RTPose / ViTPose) | build | form | 优化器（auto 或具体名字） |
| `learning_rate` | float | 0.01 / 0.0005 (RTPose / ViTPose) | build | form | 初始学习率 |
| `final_lr_factor` | float | 0.01 / — (RTPose / ViTPose) | build | form | 末端学习率因子 |
| `momentum` | float | 0.937 / 0.9 (RTPose / ViTPose) | build | form | SGD 动量 / Adam beta1 |
| `weight_decay` | float | 0.0005 / 0.1 (RTPose / ViTPose) | build | form | 权重衰减 |
| `warmup_epochs` | float | 3.0 / — (RTPose / ViTPose) | build | form | warmup 轮数 |
| `warmup_iters` | int | — / 500 (RTPose / ViTPose) | build | advanced | warmup 步数（与 warmup_epochs 二选一） |
| `warmup_start_factor` | float | — / 0.001 (RTPose / ViTPose) | build | advanced | warmup 起始学习率因子 |
| `warmup_momentum` | float | 0.8 / — (RTPose / ViTPose) | build | advanced | warmup 起始动量 |
| `warmup_bias_lr` | float | 0.1 / — (RTPose / ViTPose) | build | advanced | bias 组 warmup 学习率 |
| `nominal_batch_size` | int | 64 / — (RTPose / ViTPose) | build | advanced | 名义批大小（缩放学习率用） |
| `cosine_schedule` | bool | False / — (RTPose / ViTPose) | build | form | 余弦学习率调度 |
| `amp` | bool | True / False (RTPose / ViTPose) | build | form | 混合精度训练 |
| `ema` | bool | True / False (RTPose / ViTPose) | build | form | 指数滑动平均 |
| `close_mosaic` | int | 10 / — (RTPose / ViTPose) | build | form | 末尾 N 轮关闭 mosaic |
| `layer_decay_rate` | float | `null` / — (RTPose / ViTPose) | build | advanced | ViT 层衰减率 |
| `seed` | int | — / 21 (RTPose / ViTPose) | build | advanced | 模型配方级随机种子（优先于 shared.runtime.seed） |

**`train.shared.training`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `lr_gamma` | float | 0.1 | build | advanced | 里程碑处的学习率衰减 |
| `best_metric` | str | `coco/AP` | build | advanced | 最优模型选择指标（coco/AP 或 ultralytics/AP） |
| `best_metric_mode` | str | `max` | build | advanced | 选择方向：max / min |
| `early_stopping_patience` | int | 300 | build | form | 早停耐心轮数 |
| `single_cls` | bool | False | build | advanced | 多类数据按单类训练 |
| `multi_scale` | bool | False | build | advanced | 多尺度抖动 |
| `freeze` | int / list | `null` | build | advanced | 冻结前 n 层或指定层索引 |

**`train.models.loss`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `box_loss_weight` | float | 7.5 / — (RTPose / ViTPose) | build | form | 框回归损失权重 |
| `class_loss_weight` | float | 0.5 / — (RTPose / ViTPose) | build | form | 分类损失权重 |
| `distribution_loss_weight` | float | 1.5 / — (RTPose / ViTPose) | build | form | DFL 分布损失权重 |
| `keypoint_loss_weight` | float | 12.0 / — (RTPose / ViTPose) | build | form | 关键点损失权重 |
| `visibility_loss_weight` | float | 1.0 / — (RTPose / ViTPose) | build | form | 可见性损失权重 |
| `label_sigma` | float | — / 6.0 (RTPose / ViTPose) | build | advanced | SimCC 标签高斯 sigma |
| `kl_beta` | float | — / 1.0 (RTPose / ViTPose) | build | advanced | SimCC KL 损失 beta |

**`train.models.augmentation`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `hsv_h` | float | 0.015 / — (RTPose / ViTPose) | build | advanced | 色调增强幅度 |
| `hsv_s` | float | 0.7 / — (RTPose / ViTPose) | build | advanced | 饱和度增强幅度 |
| `hsv_v` | float | 0.4 / — (RTPose / ViTPose) | build | advanced | 明度增强幅度 |
| `degrees` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | 旋转增强（±度） |
| `translate` | float | 0.1 / — (RTPose / ViTPose) | build | advanced | 平移增强比例 |
| `scale` | float | 0.5 / — (RTPose / ViTPose) | build | advanced | 缩放增强增益 |
| `shear` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | 剪切增强（±度） |
| `perspective` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | 透视增强 |
| `flipud` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | 上下翻转概率 |
| `fliplr` | float | 0.5 / — (RTPose / ViTPose) | build | advanced | 左右翻转概率 |
| `bgr` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | BGR 通道交换概率 |
| `mosaic` | float | 1.0 / — (RTPose / ViTPose) | build | advanced | mosaic 概率 |
| `mixup` | float | 0.0 / — (RTPose / ViTPose) | build | advanced | mixup 概率 |
| `horizontal_flip_probability` | float | — / 0.5 (RTPose / ViTPose) | build | advanced | 水平翻转概率 |
| `half_body_probability` | float | — / 0.3 (RTPose / ViTPose) | build | advanced | 半身裁剪概率 |
| `half_body_upper_probability` | float | — / 0.7 (RTPose / ViTPose) | build | advanced | 半身裁剪取上半身概率 |
| `half_body_scale_factor` | float | — / 1.5 (RTPose / ViTPose) | build | advanced | 半身裁剪缩放因子 |
| `bbox_shift_probability` | float | — / 0.3 (RTPose / ViTPose) | build | advanced | 框平移概率 |
| `bbox_shift_factor` | float | — / 0.16 (RTPose / ViTPose) | build | advanced | 框平移幅度 |
| `bbox_scale_factor` | float | — / 0.5 (RTPose / ViTPose) | build | advanced | 框缩放幅度 |
| `bbox_rotation_probability` | float | — / 0.6 (RTPose / ViTPose) | build | advanced | 框旋转概率 |
| `bbox_rotation_factor` | float | — / 80.0 (RTPose / ViTPose) | build | advanced | 框旋转幅度（度） |

**`train.models.validation`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `confidence` | float | 0.001 / — (RTPose / ViTPose) | build | advanced | 验证置信度阈值 |
| `iou` | float | 0.7 / — (RTPose / ViTPose) | build | advanced | 验证 NMS IoU |
| `max_det` | int | 300 / — (RTPose / ViTPose) | build | advanced | 验证每图最大检测数 |
| `agnostic_nms` | bool | False / — (RTPose / ViTPose) | build | advanced | 验证类无关 NMS |
| `oks_nms_threshold` | float | — / 0.9 (RTPose / ViTPose) | build | advanced | ViTPose 验证 OKS-NMS 阈值 |
| `keypoint_score_threshold` | float | — / 0.2 (RTPose / ViTPose) | build | advanced | ViTPose 验证关键点过滤阈值 |
| `flip_tta` | bool | — / False (RTPose / ViTPose) | build | advanced | ViTPose 验证翻转 TTA |

**`train.metrics`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `oks_area_mode` | str | `annotation` | build | advanced | OKS 面积口径：annotation / bbox |
| `pck_threshold` | float | 0.05 | build | advanced | PCK 判定阈值（占归一化距离比例） |
| `auc_norm_factor` | float | 30.0 | build | advanced | AUC 归一化因子 |
| `auc_thresholds` | int | 20 | build | advanced | AUC 阈值采样数 |
| `coco_max_detections` | int | 20 | build | advanced | COCO 评估每图最大检测数 |

### Scope `common`

**`common`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `device` | str | `auto` | build | form | 运行设备（auto 或具体设备名） |
| `project` | str | `runs` | build | form | 运行输出根目录 |
| `name` | str | `train` | build | form | 运行名（输出在 project/name 下） |
| `exist_ok` | bool | False | build | form | 允许写进已存在的输出目录 |
| `selected_weights` | str | `null` | build | hidden | 为评估/预测/导出显式选择的权重或产物路径（GUI 模型选择） |

### Scope `predict`

**`predict`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `source` | str | `null` | build | form | 默认输入源（图片/视频/目录/glob） |
| `confidence` | float | 0.25 | live | form | 预测置信度阈值 |
| `iou` | float | 0.7 | live | form | 预测 NMS IoU |
| `max_det` | int | 300 | live | form | 每图最大检测数 |
| `agnostic_nms` | bool | False | live | form | 类无关 NMS |
| `keypoint_confidence` | float | 0.25 | live | form | 关键点绘制/过滤阈值 |
| `dnn` | bool | False | build | advanced | 用 OpenCV DNN 后端推理 |
| `show_labels` | bool | True | live | form | 绘制类标签 |
| `show_conf` | bool | True | live | form | 绘制置信度 |
| `show_boxes` | bool | True | live | form | 绘制检测框 |
| `show_keypoints` | bool | True | live | form | 绘制关键点 |
| `show_skeletons` | bool | False | live | form | 绘制骨架连线 |
| `line_width` | int | 2 | live | form | 框/骨架线宽 |
| `point_radius` | int | 4 | live | form | 关键点半径 |

**`predict.tracking`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `enabled` | bool | False | build | form | 跨帧跟踪开关 |
| `algorithm` | str | `bytetrack` | build | form | 跟踪算法（bytetrack 等 6 种） |

**`predict.pose_filter`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `method` | str | `none` | run_end | form | 时序平滑方法：none / median / savgol |
| `window_length` | int | 5 | run_end | form | 平滑窗口帧数 |
| `polyorder` | int | 2 | run_end | form | Savitzky-Golay 多项式阶数 |
| `max_gap` | int | 0 | run_end | form | 最大插值缺失帧数 |
| `confidence_threshold` | float | `null` | run_end | form | 滤波可信阈值；null = 继承 keypoint_confidence |

### Scope `export`

**`export`**

| key | type | default | apply | visibility | notes |
|---|---|---|---|---|---|
| `format` | str | `null` | build | form | 导出格式（onnx / engine / rknn ...） |
| `half` | bool | False | build | form | FP16 导出 |
| `dynamic` | bool | False | build | advanced | 动态轴/动态 batch |
| `simplify` | bool | True | build | advanced | ONNX 图简化（onnxslim） |
| `opset` | int | 17 | build | advanced | ONNX opset 版本 |
| `workspace` | float | 4.0 | build | advanced | TensorRT workspace（GiB） |
| `rknn_target` | str | `rk3588` | build | advanced | Rockchip NPU 目标（如 rk3588） |
| `optimize` | bool | False | build | advanced | TorchScript 移动端优化 |
| `int8` | bool | False | build | advanced | INT8 量化 |
| `keras` | bool | False | build | hidden | Keras 导出（不支持则报错） |
| `nms` | bool | False | build | hidden | 图中嵌入 NMS（不支持则报错） |

<!-- END GENERATED REFERENCE -->
