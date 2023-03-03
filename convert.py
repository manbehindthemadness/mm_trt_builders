"""
Global type converter.
"""
import onnx

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto, GraphProto
from onnx import numpy_helper as nph

import numpy as np
from collections import OrderedDict

from logger import log
import typer


DTYPE = np.int32
DNAME = 'int32'


def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def convert_params_to_int32(params_dict):
    converted_params = []
    for param in params_dict:
        data = params_dict[param]
        if data.data_type == TensorProto.INT64:
            data_cvt = nph.to_array(data).astype(DTYPE)
            data = nph.from_array(data_cvt, data.name)
        converted_params += [data]
    return converted_params


def convert_constant_nodes_to_int32(nodes):
    """
    convert_constant_nodes_to_int32 Convert Constant nodes to INT32. If a constant node has data type INT64, a new version of the
    node is created with INT32 data type and stored.
    Args:
        nodes (list): list of nodes
    Returns:
        list: list of new nodes all with INT32 constants.
    """
    new_nodes = []
    for node in nodes:
        if (
            node.op_type == "Constant"
            and node.attribute[0].t.data_type == TensorProto.INT64
        ):
            data = nph.to_array(node.attribute[0].t).astype(DTYPE)
            new_t = nph.from_array(data)
            new_node = h.make_node(
                "Constant",
                inputs=[],
                outputs=node.output,
                name=node.name,
                value=new_t,
            )
            new_nodes += [new_node]
        else:
            new_nodes += [node]

    return new_nodes


def convert_model_to_int32(model_path: str, out_path: str, precision: int):
    """
    convert_model_to_int32 Converts ONNX model with INT64 params to INT32 params.\n
    Args:\n
        model_path (str): path to original ONNX model.\n
        out_path (str): path to save converted model.
    """
    global DNAME
    global DTYPE

    if precision == 32:
        DTYPE = np.int32
        DNAME = 'int32'
    elif precision == 16:
        DTYPE = np.int16
        DNAME = 'int16'
    elif precision == 8:
        DTYPE = np.int8
        DNAME = 'int8'
    else:
        raise TypeError

    log.info(f"ONNX INT64 --> {DNAME.upper()} Converter")
    log.info(f"Loading Model: {model_path}")
    # * load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    # * get model opset version.
    opset_version = model.opset_import[0].version
    graph = model.graph
    # * The initializer holds all non-constant weights.
    init = graph.initializer
    # * collect model params in a dictionary.
    params_dict = make_param_dictionary(init)
    log.info(f"Converting {DNAME.upper()} model params to {DNAME}...")
    # * convert all INT64 aprams to INT32.
    converted_params = convert_params_to_int32(params_dict)
    log.info(f"Converting constant {DNAME.upper()} nodes to {DNAME}...")
    new_nodes = convert_constant_nodes_to_int32(graph.node)

    graph_name = f"{graph.name}-{DNAME}"
    log.info("Creating new graph...")
    # * create a new graph with converted params and new nodes.
    graph_int32 = h.make_graph(
        new_nodes,
        graph_name,
        graph.input,
        graph.output,
        initializer=converted_params,
    )
    log.info(f"Creating new {DNAME} model...")
    model_int32 = h.make_model(graph_int32, producer_name="onnx-typecast")
    model_int32.opset_import[0].version = opset_version
    ch.check_model(model_int32)
    log.info(f"Saving converted model as: {out_path}")
    onnx.save_model(model_int32, out_path)
    log.info(f"Done Done London. 🎉")
    return


if __name__ == "__main__":
    typer.run(convert_model_to_int32)
