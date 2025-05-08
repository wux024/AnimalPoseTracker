#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ctypes
from typing import Optional, List, Union
import numpy as np
from pathlib import Path

try:
    import tensorrt as trt
    from cuda import cuda, cudart
    def check_cuda_err(err):
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Cuda Runtime Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))

    def cuda_call(call):
        err, res = call[0], call[1:]
        check_cuda_err(err)
        if len(res) == 1:
            res = res[0]
        return res


    class HostDeviceMem:
        """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
        def __init__(self, size: int, dtype: Optional[np.dtype] = None):
            dtype = dtype or np.dtype(np.uint8)
            nbytes = size * dtype.itemsize
            host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
            self._device = cuda_call(cudart.cudaMalloc(nbytes))
            self._nbytes = nbytes

        @property
        def host(self) -> np.ndarray:
            return self._host

        @host.setter
        def host(self, data: Union[np.ndarray, bytes]):
            if isinstance(data, np.ndarray):
                if data.size > self.host.size:
                    raise ValueError(
                        f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                    )
                np.copyto(self.host[:data.size], data.flat, casting='safe')
            else:
                assert self.host.dtype == np.uint8
                self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

        @property
        def device(self) -> int:
            return self._device

        @property
        def nbytes(self) -> int:
            return self._nbytes

        def __str__(self):
            return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

        def __repr__(self):
            return self.__str__()

        def free(self):
            cuda_call(cudart.cudaFree(self.device))
            cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    # If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
    def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for binding in tensor_names:
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
            size = trt.volume(shape)
            trt_type = engine.get_tensor_dtype(binding)

            # Allocate host and device buffers
            if trt.nptype(trt_type):
                dtype = np.dtype(trt.nptype(trt_type))
                bindingMemory = HostDeviceMem(size, dtype)
            else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
                size = int(size * trt_type.itemsize)
                bindingMemory = HostDeviceMem(size)

            # Append the device buffer to device bindings.
            bindings.append(int(bindingMemory.device))

            # Append to the appropriate list.
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(bindingMemory)
            else:
                outputs.append(bindingMemory)
        return inputs, outputs, bindings, stream


    # Frees the resources allocated in allocate_buffers
    def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
        for mem in inputs + outputs:
            mem.free()
        cuda_call(cudart.cudaStreamDestroy(stream))


    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
        nbytes = host_arr.size * host_arr.itemsize
        cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

    # Wrapper for cudaMemcpy which infers copy size and does error checking
    def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
        nbytes = host_arr.size * host_arr.itemsize
        cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


    def _do_inference_base(inputs, outputs, stream, execute_async_func):
        # Transfer input data to the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
        # Run inference.
        execute_async_func()
        # Transfer predictions back from the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
        # Synchronize the stream
        cuda_call(cudart.cudaStreamSynchronize(stream))
        # Return only the host outputs.
        return [out.host for out in outputs]


    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(context, engine, bindings, inputs, outputs, stream):
        def execute_async_func():
            context.execute_async_v3(stream_handle=stream)
        # Setup context tensor address.
        num_io = engine.num_io_tensors
        for i in range(num_io):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
        return _do_inference_base(inputs, outputs, stream, execute_async_func)
except ImportError:
    pass


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (np.squeeze(x, axis=-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (np.squeeze(x, axis=-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
        / (4 * np.sqrt(np.clip(a1 * b1 - c1 ** 2, 0, None) * np.clip(a2 * b2 - c2 ** 2, 0, None)) + eps)
        + eps
    ) * 0.5 * np.log(1)
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def _get_covariance_matrix(boxes):
    """
    Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (np.ndarray): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (tuple): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = np.concatenate((np.square(boxes[:, 2:4]) / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, 3, axis=-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = np.square(cos)
    sin2 = np.square(sin)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin



def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = x.clone()
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray): The input tensor with the bounding box coordinates in the xywh format

    Returns:
        y (np.ndarray): The bounding box coordinates in the xyltwh format
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (numpy.ndarray): The bounding boxes to clip.
        shape (tuple): The shape of the image.

    Returns:
        (numpy.ndarray): The clipped boxes.
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def save_one_box(xyxy, im, gain=1.02, pad=10, square=False, BGR=False):
    """
    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy (np.ndarray | list): A tensor or list representing the bounding box in xyxy format.
        im (np.ndarray): The input image.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box.
        square (bool, optional): If True, the bounding box will be transformed into a square.
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB.

    Returns:
        (np.ndarray): The cropped image.

    """
    if not isinstance(xyxy, np.numpy):  # may be list
        xyxy = np.stack(xyxy)
    b = xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    xyxy = clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    return crop
