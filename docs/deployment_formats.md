# Project prediction and deployment formats

`AnimalPoseTrackerProject.predict()` is a dataset-oriented pose workflow. With no explicit source it reads `test` from `configs/dataset.yaml`, falling back to `val`; `--split` can select `train`, `val`, or `test` in the worker. AnimalRTPose predicts detections on each full image. AnimalViTPose uses each instance box from the selected COCO or YOLO annotation split, performs the same padded top-down crop and normalization as training, then maps SimCC points back to the source image.

The project worker is separate from the live image/video/camera inferencer. It accepts local image/video files, directories, recursive globs, and `.txt` source manifests. It does not open camera indices, RTSP streams, or browser URLs. Those remain live-inference sources.

For AnimalViTPose, external images without labels need an instance box provider. The worker exposes `InstanceBoxProvider.boxes_for_image(image, image_path)`. `DetectorBoxProvider` adapts either a callable or an object with `predict_boxes(image)`. `run(argv, box_provider=...)` can receive that provider from Python; the project does not train or load a detector automatically.

## Artifact formats

`Project.export()` exports from a native `.pt`/`.pth` checkpoint. `.pt` remains the native checkpoint format; it is a prediction input, not an export target. Other targets are selected with `format` in `configs/other.yaml`.

| Format | Export target | Project prediction | Runtime/converter requirement |
|---|---|---|---|
| PyTorch checkpoint | Native `.pt`/`.pth` | Yes | `.[training]` |
| TorchScript | `torchscript` | Yes | PyTorch |
| ONNX | `onnx` | Yes | `onnxruntime`; export uses `.[export]` |
| OpenVINO | `openvino` | Yes | `openvino>=2024.0.0` |
| TensorRT | `engine` | Yes | TensorRT Python API, CUDA, and a CUDA device for export/predict |
| CoreML | `coreml` | Yes | `coremltools>=8.0`; CoreML runtime platform required |
| TensorFlow SavedModel | `saved_model` | Yes | `tensorflow`, `onnx2tf` |
| TensorFlow GraphDef | `pb` | Yes | `tensorflow`, `onnx2tf` |
| TensorFlow Lite | `tflite` | Yes | `tensorflow` or `tflite-runtime` |
| TensorFlow Edge TPU | `edgetpu` | Yes | TFLite plus Coral compiler/runtime and compatible Edge TPU hardware |
| TensorFlow.js | `tfjs` | No local runner | `tensorflowjs`; browser deployment format, as in the Ultralytics AutoBackend baseline |
| PaddlePaddle | `paddle` | Yes | `paddlepaddle`, `x2paddle` |
| MNN | `mnn` | Yes | `MNN>=2.9.6` |
| NCNN | `ncnn` | Yes | NCNN Python bindings and PNNX executable |
| Sony IMX500 | `imx` | Yes, through the generated ONNX graph | Linux, Model Compression Toolkit, Edge MDT target package, IMX500 converter and runtime extensions |
| Rockchip NPU | `rknn` | Yes | RKNN Toolkit 2 for export; `rknn-toolkit-lite2` on supported Rockchip hardware for prediction |
| NVIDIA Triton | Remote model URL | Yes | `tritonclient[http]` or `tritonclient[grpc]`; not a local export target |

Export and prediction use the same raw-output contract: model-specific preprocessing and pose decoding stay in the AnimalPoseTracker worker. Deployment libraries are loaded lazily, so only the selected format's packages are required. Format availability does not imply that every optional SDK is installed on the current machine; missing SDKs fail with a format-specific dependency message.

Optional Python groups are available for common software backends:

```powershell
pip install -e ".[export]"
pip install -e ".[export-openvino]"
pip install -e ".[export-coreml]"
pip install -e ".[export-tensorflow]"
pip install -e ".[export-paddle]"
pip install -e ".[export-mnn]"
pip install -e ".[export-ncnn]"
pip install -e ".[predict-triton]"
```

TensorRT, Edge TPU, IMX500, and RKNN require vendor/platform components beyond a portable Python extra. They are not installed automatically by AnimalPoseTracker. INT8 calibration is wired for TFLite/Edge TPU and IMX500; unsupported precision requests are rejected rather than silently ignored.

The worker preserves the existing `runs/predict/predictions.json` list schema. `prediction_metadata.json` records the checkpoint/artifact, format, model type, and selected dataset split.
