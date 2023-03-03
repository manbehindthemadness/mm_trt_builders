"""
Testing TensorRT inference on resnet 50

https://dongle94.github.io/tensorrt/nvidia-tensorrt-cupy-with-yolov5/#cupy
"""
import os
import tensorrt as trt
import cupy as cp
import cv2
from PIL import Image

os.environ['DISPLAY'] = ':0'

logger = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(logger)

inputs = []
outputs = []
bindings = []

engine_path = 'fcn-resnet101.engine'

input_image = 'input.jpg'
with Image.open(input_image) as img:
    img.load()
image_width = img.width
image_height = img.height
img = cp.asarray(img)  # noqa


def preprocess(image):
    # Mean normalization
    mean = cp.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = cp.array([0.229, 0.224, 0.225]).astype('float32')
    data = (cp.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    img = cp.moveaxis(data, 2, 0)
    img = cp.ascontiguousarray(img)
    img = cp.expand_dims(img, axis=0)
    return img


def postprocess(data):
    num_classes = 21
    # create a color palette, selecting a color for each class
    palette = cp.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = cp.array([palette * i % 255 for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    data *= 255
    img = Image.fromarray(cp.asnumpy(data.astype('uint8')), mode='P')
    img.putpalette(cp.asnumpy(colors))  # noqa
    return img


with open(engine_path, 'rb') as f:
    engine_data = f.read()
engine = trt_runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()
context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))

for binding in engine:
    # Append the device buffer to device bindings.
    shape = engine.get_binding_shape(binding)
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # Allocate buffers
    cuda_mem = cp.zeros(shape=(1, 3, image_height, image_width), dtype=dtype)
    bindings.append(cuda_mem.data.ptr)

    if engine.binding_is_input(binding):
        inputs.append(cuda_mem)
    else:
        outputs.append(cuda_mem)

src = preprocess(img)
cp.copyto(inputs[0], src)


with cp.cuda.Stream(non_blocking=False) as stream:
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)

    # Synchronize the stream
    stream.synchronize()

    # Return the outputs.
    res = outputs[0][0]

pass
print(res.min(), res.max(), res.shape, len(res[0][0]))

data = cp.vstack(res[0])
result = postprocess(data)

import numpy as np

cv2.imshow('res', np.asarray(result))
cv2.waitKey(0)
