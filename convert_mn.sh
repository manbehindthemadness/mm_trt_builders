#!/bin/bash

rm -f ./*.onnx*
downsample=$1
rez=$2
width=$3
height=$4
precision=$5
workspace=$6
# trt_version="$(echo $( tr -d '.-' <<< $(echo $(echo $(echo $(dpkg -l | grep nvinfer-plugin-dev) | cut -d ' ' -f 3)) | cut -d '+' -f 1)))"
trt_version='test'

if [ -z "${downsample}"    ]; then echo "Error: ARG DOWNSAMPLE    not specified."; exit 1; fi \
    && if [ -z "${rez}"   ]; then echo "Error: ARG REZ   not specified."; exit 1; fi \
    && if [ -z "${height}"  ]; then echo "Error: ARG HEIGHT  not specified."; exit 1; fi \
    && if [ -z "${width}" ]; then echo "Error: ARG WIDTH not specified."; exit 1; fi \
    && if [ -z "${precision}" ]; then echo "Error: ARG PRECISION not specified."; exit 1; fi \
    && if [ -z "${workspace}" ]; then echo "Error: ARG WORKSPACE not specified."; exit 1; fi

target="mnv3-trt$trt_version-fp$precision-$rez-ds$downsample.engine"

echo target: "$target"
echo tensorrt version: "$trt_version"
echo downsample: "$downsample"
echo resolution: "$rez"
echo width: "$width"
echo height: "$height"
echo precision: "$precision"
echo workspace: "$workspace"
sleep 5
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp${precision}.onnx

shapes=$(python rvm_onnx_infer.py --model "rvm_mobilenetv3_fp${precision}.onnx" --input-image "input.jpg" --precision float${precision} --downsample-ratio ${downsample} --input-resize ${width} ${height})

echo ${shapes}

python -m onnxsim rvm_mobilenetv3_fp${precision}.onnx rvm_mobilenetv3_fp${precision}_sim.onnx --overwrite-input-shape ${shapes}

python rvm_onnx_modify.py -i rvm_mobilenetv3_fp${precision}_sim.onnx --downsample-ratio ${downsample}

# https://github.com/aadhithya/onnx-typecast.git  Convert all int64 values to int 32
python convert.py rvm_mobilenetv3_fp${precision}_sim_modified.onnx rvm_mobilenetv3_fp${precision}_sim_modified_fp${precision}.onnx 32

trt_shapes=$(python translate_inputs.py --onnx-shapes "${shapes}" --size ${width} ${height})

echo ${trt_shapes}

trtexec --onnx=rvm_mobilenetv3_fp${precision}_sim_modified_fp${precision}.onnx --streams=2 --exposeDMA --workspace=${workspace} --saveEngine=${target} --verbose # --useDLACore=0 --allowGPUFallback # --optShapes=$trt_shapes
