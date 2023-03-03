#!/bin/bash

rm -f ./*.onnx*
downsample=$1
rez=$2
width=$3
height=$4
precision=$5

target="mnv3-trt8517-fp$precision-$rez-ds$downsample.engine"

echo
echo target $target
echo
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp$precision.onnx

shapes=$(python rvm_onnx_infer.py --model "rvm_mobilenetv3_fp$precision.onnx" --input-image "input.jpg" --precision float$precision --downsample-ratio $downsample --input-resize $width $height)

echo $shapes

python -m onnxsim rvm_mobilenetv3_fp${precision}.onnx rvm_mobilenetv3_fp${precision}_sim.onnx --overwrite-input-shape $shapes

python rvm_onnx_modify.py -i rvm_mobilenetv3_fp${precision}_sim.onnx --downsample-ratio $downsample

# https://github.com/aadhithya/onnx-typecast.git  Convert all int64 values to int 32
python convert.py rvm_mobilenetv3_fp${precision}_sim_modified.onnx rvm_mobilenetv3_fp${precision}_sim_modified_fp${precision}.onnx 32

trt_shapes=$(python translate_inputs.py --onnx-shapes "$shapes" --size $width $height)

echo $trt_shapes

trtexec --onnx=rvm_mobilenetv3_fp${precision}_sim_modified_fp${precision}.onnx --streams=2 --exposeDMA --workspace=$6 --saveEngine=$target --verbose # --useDLACore=0 --allowGPUFallback # --optShapes=$trt_shapes
