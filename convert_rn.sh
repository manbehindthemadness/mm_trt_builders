#!/bin/bash

rm -f ./*.onnx*
downsample=0.23
rez=2k
width=2560
height=1440
precision=32
target=rn50-trt8517-fp$precision-$rez-ds$downsample.engine

echo
echo target $target
echo
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp$precision.onnx


shapes=$(python rvm_onnx_infer.py --model "rvm_resnet50_fp$precision.onnx" --input-image "input.jpg" --precision float$precision --downsample-ratio $downsample --input-resize $width $height)

echo SHAPES "$shapes"

python -m onnxsim rvm_resnet50_fp${precision}.onnx rvm_resnet50_fp${precision}_sim.onnx --overwrite-input-shape $shapes

python rvm_onnx_modify_rn.py -i rvm_resnet50_fp${precision}_sim.onnx --downsample-ratio $downsample

# https://github.com/aadhithya/onnx-typecast.git  Convert all int64 values to int 32
python convert.py rvm_resnet50_fp${precision}_sim_modified.onnx rvm_resnet50_fp${precision}_sim_modified_fp${precision}.onnx 32

trtexec --onnx=rvm_resnet50_fp${precision}_sim_modified_fp${precision}.onnx  --workspace=8000 --saveEngine=$target --verbose
