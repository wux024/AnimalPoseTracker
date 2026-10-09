# 自研训练框架重构草案

- owner: Codex
- date: 2026-10-08
- status: 实施中

## 已确认的要求

- 作者明确要求：模型训练不能依赖 YOLO 训练框架。
- 对齐基线固定为 [`wux024/ultralytics@188d5bdb`](https://github.com/wux024/ultralytics/tree/188d5bdb204263ced5f061aeabd4d1606a150911)，行为参考该提交的 pose validator、metrics 和 loss；本项目在本地实现对应逻辑，不在训练运行时导入 Ultralytics。

## 当前实现

- 新增自有训练配置、PyTorch 训练循环、参数分组和 `optimizer=auto` 选择、学习率计划、warmup、`nbs` 梯度累积、AMP、提前停止和 EMA 检查点恢复。
- 本地 `YOLOPoseHead` 损失已按 `animalrtpose` 分支语义对齐：任务对齐分配、CIoU、DFL、OKS 关键点损失、可见性 BCE 和配置权重。运行时不导入该分支。
- 现有拆分目录和关键点文本标注由本地数据集加载器读取。训练侧已接入 2×2 Mosaic、HSV、旋转/平移/缩放/剪切/透视、MixUp 与翻转；裁剪变换会同步更新框、关键点和可见性。数据集未提供 `flip_idx` 时会禁用左右/上下翻转并告警。
- `Tri-Mouse` 预设包含 `flip_idx=[0,2,1,3,...,11]`；mouse smoke 配置现已带上该映射，训练数据加载器会读取它并进行左右关键点重排。
- 设置 `annotation_format: coco` 后，加载器会按 split 自动发现 `annotations/{train,val,test}.json` 或 `person_keypoints_{split}2017.json`；也可以用 `train_annotations`/`val_annotations` 显式覆盖路径。
- `cache` 支持 RAM 和磁盘缓存，`fraction` 仅抽取训练集，`close_mosaic` 会按训练 epoch 关闭 Mosaic。`time` 支持小时数上限；`profile` 记录数据等待、前向/损失、反向/优化器和 epoch 用时。
- `multi_scale` 对训练批次按模型 stride 随机缩放；`freeze` 支持冻结前 N 层或显式层索引并保持冻结层 BN 统计不变；`single_cls` 会把数据标签和模型输出头都映射为单类。pose 任务设置非零 `copy_paste` 会明确报错，因为该参考参数只定义了分割复制粘贴。
- AnimalRTPose 与 AnimalViTPose 的关键点 AP 统一由 COCOeval 的 `keypoints` evaluator 计算；不再把框 AP 纳入姿态验证分数。两者输出 `coco/AP`、`coco/AP50`、`coco/AP75` 和 `coco/AR`，并用 `coco/AP` 选择最佳检查点。AnimalRTPose 仍保留 `fitness` 和旧 pose mAP 字段作为兼容别名。
- `animalposetracker/cfg/training.yaml` 只定义训练侧默认值。两个模型完全相同的参数只放在 `training.shared`，同名但取值不同以及模型独有的训练、损失、增强、验证参数留在各自 profile。训练 profile 根据 `project.yaml` 中唯一的 `model_type` 选择，不在 `other.yaml` 重复保存模型选择；模型图、AnimalViTPose 规模、SimCC 输出几何和输入归一化属于 `configs/model.yaml`；数据路径与关键点元数据属于 `configs/dataset.yaml`。项目管理层只在内存中展平选中的训练 profile，GUI 源码不需要改动。
- `project.yaml` 保留标注 GUI 直接读取的类别/关键点名称、数量和骨架元数据；`dataset.yaml` 提供数据加载器需要的对应字段；`model.yaml.kpt_shape` 是模型输出形状约束，必须与数据集一致。`kpt_oks_sigmas` 只保留在 `dataset.yaml`，模型选择只保留在 `project.yaml`。
- 项目评估通过 `animalposetracker.training.cli --validate-only` 执行，复用训练验证器并把 COCO/PCK/AUC/EPE 写入 `runs/val/metrics.json`。项目预测默认为 test split（缺失时使用 val）：AnimalRTPose 在整图上预测，AnimalViTPose 从 YOLO/COCO 标注框生成 top-down crop，并通过独立的 `InstanceBoxProvider` 接口预留检测器来源。预测 worker 不调用 GUI 的实时推理引擎。
- 项目导出通过 `animalposetracker.export_cli` 执行，不调用 YOLO CLI。TorchScript、ONNX、OpenVINO、TensorRT、CoreML、TensorFlow SavedModel/GraphDef/TFLite/Edge TPU/TF.js、PaddlePaddle、MNN、NCNN、IMX500、RKNN 有本地转换分派；目标 SDK/转换器按格式可选安装。导出的模型保存 JSON 元数据描述输入尺寸、归一化、关键点形状和输出张量。
- 数据集 YAML 只使用 `kpt_oks_sigmas` 记录逐关键点 OKS sigma，并显式传入 COCOeval 覆盖其人体 COCO-17 内置值。MMPose 数据集 metainfo 的 `sigmas` 在迁入本项目配置时写入 `kpt_oks_sigmas`。缺省时使用均匀 `1/K` 自定义关键点回退；自定义动物数据应配置逐关键点 sigma。`plots` 输出训练与验证曲线。
- GUI 的训练、恢复入口已调用 `python -m animalposetracker.training.cli`；训练状态写入 JSONL 并显示在状态栏，停止请求通过标记文件传递以保存检查点。
- GUI 保存的 `box/cls/dfl/pose/kobj`、`val` 和 `patience` 现在映射到自有损失权重、验证开关和早停；界面静态默认值也已与 `cfg/default.yaml` 对齐。
- 增加训练和导出依赖 extra 及子包发现，使新增训练包能随项目安装。README 不要求安装 Ultralytics。
- DDP 使用 PyTorch `DistributedDataParallel` 和分布式训练采样器；CPU/Gloo 两进程训练已在 mouse 子集上跑通。Linux 优先选 NCCL，Windows 使用 Gloo；当前工作环境是 CPU-only，GPU 路径未实测。分布式验证目前各 rank 都计算完整验证集，正确但重复计算。
- AnimalViTPose 已接入同一个 `Trainer`：按 COCO 或 YOLO 实例框生成 top-down crop，支持 MMPose Small/Base/Large/Huge。网络宽度、深度、drop-path、MAE 预训练地址、SimCC 输出几何和归一化参数由模型 YAML 管理；layer decay 与训练配方、损失、增强和验证参数由训练配置管理。COCO 和 YOLO 验证都使用 COCOeval；YOLO 标注在内存中转换为 COCO ground truth。验证输出 AP、PCK、AUC 和 EPE。AnimalRTPose 也输出 PCK、AUC、EPE，误差仅在 COCOeval 以 OKS 0.50 匹配到的实例上统计；未匹配实例由 COCO AP/AR 反映。MMPose 只作实现基线，训练运行时不导入 MMPose。
- `SimCCHead` 的 top-down 数据适配、损失与指标通过可插拔组件接入共享训练循环；没有复制一套 optimizer-step / checkpoint / resume 循环。`profile` 仍是本地训练阶段计时，不执行 ONNX/TensorRT 基准测试。
- 已用桌面 `mouse` 数据和 `animalrtpose-n.pt` 完成对齐后的 CPU 两轮试跑。加载器用 PyTorch 受限加载把已知旧 checkpoint 类名映射到本地网络类，不导入 YOLO 包；模型 465 个张量中迁移了 423 个，42 个尺寸不同的分类/关键点头张量保留新初始化。

## 设计边界

1. 训练、恢复、验证、项目数据集预测、模型导出和检查点读写全部由 AnimalPoseTracker 自己编排；这些入口不导入或调用 Ultralytics，不调用 `yolo` 命令，也不复用 GUI 实时推理引擎。
2. 使用项目自己的 `animalposetracker.nn` 构建网络，并以 PyTorch 实现梯度更新、优化器、学习率计划、混合精度和设备管理。
3. 将模型定义、数据集读取、模型专属目标编码与损失、共用训练循环、指标、检查点和 GUI 进程管理拆分成明确模块。
4. 项目 YAML、数据集拆分目录、标注文本和 `runs/train/weights/{best,last}.pt` 作为优先保留的兼容接口。复用标注文件格式只代表解析该格式，不依赖产生或训练这些标注的框架。
5. 旧项目首次打开时读取原有配置；新增训练配置要有明确的默认值和迁移逻辑，避免静默覆盖作者项目参数。旧权重可用于初始化的范围和旧优化器状态能否恢复，需要逐种检查点明确处理。

## 建议的代码边界

- `animalposetracker/training/config.py`：训练参数模型、类型转换、范围校验，以及旧项目参数到新配置的映射。
- `animalposetracker/training/data/`：读取现有图像与标注、验证样本、构造批次、图像变换。统一批次包含图像、目标边框、类别、关键点及可见性。
- `animalposetracker/training/losses/`：按网络头注册独立准则。多尺度回归头实现自己的候选分配、边框分布回归和关键点目标；SimCC 头实现关键点裁剪与 x/y 分布目标。共享训练循环不判断具体张量形状。
- `animalposetracker/training/engine.py`：训练和验证循环、优化器、学习率计划、梯度裁剪、混合精度、提前停止与恢复。
- `animalposetracker/training/checkpoint.py`：保存模型权重、优化器与计划状态、epoch、随机数状态和配置；同时提供只加载模型权重的入口，供推理使用。
- `animalposetracker/training/metrics.py` 与 `callbacks.py`：集中定义关键点指标、日志、检查点和进度事件。
- `animalposetracker/training/cli.py`：AnimalPoseTracker 自己的命令行训练工作进程。GUI 启动这个工作进程并接收可解析的状态事件，不直接依赖第三方训练命令。

## 建议的实施顺序

1. 先固定项目配置到训练参数、标注到批次、网络头到损失的接口，并解决 `pyproject.toml` 子包发现和训练依赖的可选安装方式。
2. 实现现有标注格式的数据读取和单实例/多实例批次；不先改 GUI 或标注工具的数据格式。
3. 完成共用训练循环与检查点，再按模型头接入损失。`AnimalRTPose` 和 `AnimalViTPose` 的两条模型专属路径均已接入共享循环；下一步按实际使用顺序补齐其余已有模型配置。
4. 添加自有验证、日志、暂停/停止信号和恢复入口，并保持当前项目的输出目录约定。自有验证与恢复流程已接入。
5. 将项目验证、AnimalRTPose/AnimalViTPose 数据集预测和模型导出入口切换到本地工作进程。已完成；外部图片/视频的 AnimalViTPose 预测可以向 worker 注入 `InstanceBoxProvider`。

## 实施时需要特别处理

- `AnimalViTPose` 是单只动物裁剪后的顶视姿态回归，与多目标检测式数据路径不同。它的裁剪、关键点坐标变换和训练/推理预处理必须使用同一几何规则。
- `AnimalViTPose` 已加入 `MODEL_YAML_PATHS`；项目配置固定使用一份 `configs/model.yaml`，项目规模保存在 `project.yaml` 并在训练时应用到这份模型图。
- 当前 YAML 使用与常见检测网络相似的图结构书写方式；运行解析器和图构建器属于项目本地实现。重构应明确这是项目自己的模型描述格式，并避免以第三方解析或训练运行时作为隐式依赖。
- 训练、验证、预测和导出复用 `other.yaml` 配置，但通过本地 Python worker 分别执行。AnimalViTPose 是 top-down 模型；数据集预测默认使用该 split 的标注框，外部图片/视频则需要向 worker 注入 `InstanceBoxProvider`。不得把整图静默当作一个实例 crop。
