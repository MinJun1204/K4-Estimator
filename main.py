import onnx
from onnx import shape_inference
from collections import defaultdict, deque
import numpy as np
import math
import sys

# ONNX dtype 문자열 → byte size
ONNX_TYPE_SIZE = {
    'FLOAT': 4,
    'FLOAT16': 2,
    'DOUBLE': 8,
    'INT32': 4,
    'INT64': 8,
    'UINT8': 1,
    'INT8': 1,
    'BOOL': 1,
    'UINT16': 2,
    'INT16': 2,
    'UINT32': 4,
    'UINT64': 8,
    'BFLOAT16': 2,
}

max_mreq = 0

def load_and_infer_model(path):
    model = onnx.load(path)
    return shape_inference.infer_shapes(model)

def build_tensor_maps(graph):
    tensor_producer = {}
    tensor_consumers = defaultdict(list)

    for node in graph.node:
        for output in node.output:
            tensor_producer[output] = node
        for input_name in node.input:
            tensor_consumers[input_name].append(node)

    return tensor_producer, tensor_consumers

def get_value_info_map(graph):
    tensor_info = {}

    # 1. from value_info, inputs, outputs (inferred ones)
    def extract(info):
        type_proto = info.type.tensor_type
        elem_type = onnx.TensorProto.DataType.Name(type_proto.elem_type)
        shape = [d.dim_value if d.HasField("dim_value") else '?' for d in type_proto.shape.dim]
        return (elem_type, shape)

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_info[value_info.name] = extract(value_info)

    # 2. from initializers (e.g., weights/biases)
    for initializer in graph.initializer:
        name = initializer.name
        elem_type = onnx.TensorProto.DataType.Name(initializer.data_type)
        shape = list(initializer.dims)
        tensor_info[name] = (elem_type, shape)

    return tensor_info

def get_attribute(node, attr_name, default=None):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
    return default

def resolve_dynamic_shapes(model, batch_size=1):
    for tensor in model.graph.input:
        shape = tensor.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:  # dynamic ('batch_size')인 경우
                dim.dim_value = batch_size
                dim.ClearField("dim_param")  # 'batch_size' 제거

    for tensor in model.graph.output:
        shape = tensor.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:  # dynamic ('batch_size')인 경우
                dim.dim_value = batch_size
                dim.ClearField("dim_param")  # 'batch_size' 제거

    for tensor in model.graph.value_info:
        shape = tensor.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:  # dynamic ('batch_size')인 경우
                dim.dim_value = batch_size
                dim.ClearField("dim_param")  # 'batch_size' 제거
    return model

def topological_sort(graph):
    input_count = {}
    consumers = defaultdict(list)
    node_by_id = {}

    for node in graph.node:
        node_id = id(node)
        node_by_id[node_id] = node
        input_count[node_id] = 0
        for input_tensor in node.input:
            if input_tensor:  # 빈 문자열 제외
                input_count[node_id] += 1
                consumers[input_tensor].append(node_id)

    initial_ready_tensors  = set(init.name for init in graph.initializer)
    initial_ready_tensors |= set(i.name for i in graph.input)

    queue = deque()
    for tensor in initial_ready_tensors:
        for node_id in consumers.get(tensor, []):
            input_count[node_id] -= 1
            if input_count[node_id] == 0:
                queue.append(node_id)

    sorted_nodes = []
    visited = set()

    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)

        node = node_by_id[node_id]
        sorted_nodes.append(node)

        for output in node.output:
            for consumer_id in consumers.get(output, []):
                input_count[consumer_id] -= 1
                if input_count[consumer_id] == 0:
                    queue.append(consumer_id)

    return sorted_nodes

def traverse_graph(model):
    global max_mreq

    graph = model.graph
    tensor_info_map = get_value_info_map(graph)

    sorted_nodes = topological_sort(graph)

    print("Topologically Sorted ONNX Graph:")
    for node in sorted_nodes:
        node_mreq = 0
        print(f"\n** Node: {node.name or '[Unnamed]'} | OpType: {node.op_type}] **")

        # if node.op_type == "MaxPool":
        #     kernel_shape = get_attribute(node, "kernel_shape")
        #     strides = get_attribute(node, "strides")
        #     pads = get_attribute(node, "pads")
        #     ceil_mode = get_attribute(node, "ceil_mode", False)
        #     input_shape = tensor_info_map[node.input[0]][1]
        #     output_shape = compute_maxpool2d_output_shape(input_shape, kernel_shape, strides, pads, ceil_mode)
        #     print(f"  Input shape: {input_shape}, Output shape: {output_shape}")
        # elif node.op_type == "Conv":

        print("  Inputs:")
        for input_tensor in node.input:
            dtype, shape = tensor_info_map.get(input_tensor, ("?", []))
            elem_size = ONNX_TYPE_SIZE.get(dtype, 0)
            num_elements = np.prod(shape) if shape else 0
            # num_elements = 1
            mreq = num_elements * elem_size
            node_mreq += mreq

            print(f"    {input_tensor}: dtype={dtype}, shape={shape}, mreq={mreq:,}")

        print("  Outputs:")
        for output_tensor in node.output:
            dtype, shape = tensor_info_map.get(output_tensor, ("?", []))
            elem_size = ONNX_TYPE_SIZE.get(dtype, 0)
            num_elements = np.prod(shape) if shape else 0
            # num_elements = 1
            mreq = num_elements * elem_size
            node_mreq += mreq

            print(output_tensor)

            print(f"    {output_tensor}: dtype={dtype}, shape={shape}, mreq={mreq:,}")
        
        print(f"  Node Mreq: {node_mreq:,} bytes")
        max_mreq = max(max_mreq, node_mreq)

def get_tensor_size_bytes(initializer):
    dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
    elem_size = ONNX_TYPE_SIZE.get(dtype, 0)
    num_elements = np.prod(initializer.dims) if initializer.dims else 0
    return dtype, num_elements * elem_size

def calculate_total_weight_memory(model_path):
    model = onnx.load(model_path)
    total_bytes = 0

    print("🔍 Weight memory usage breakdown:\n")

    for initializer in model.graph.initializer:
        name = initializer.name
        shape = list(initializer.dims)
        dtype, size_bytes = get_tensor_size_bytes(initializer)
        total_bytes += size_bytes
        print(f"  {name}: dtype={dtype}, shape={shape}, size={size_bytes:,} bytes")

    print("\n📦 Total weight memory usage: {:,} bytes ({:.2f} KB / {:.2f} MB)".format(
        total_bytes, total_bytes / 1024, total_bytes / (1024 * 1024)))

    return total_bytes

def compute_maxpool2d_output_shape(input_shape, kernel_shape, strides, pads, ceil_mode=False):
    """
    input_shape: [N, C, H, W]
    kernel_shape: [kH, kW]
    strides: [sH, sW]
    pads: [pad_top, pad_left, pad_bottom, pad_right]
    ceil_mode: bool
    """
    N, C, H, W = input_shape
    kH, kW = kernel_shape
    sH, sW = strides
    pad_top, pad_left, pad_bottom, pad_right = pads

    if ceil_mode:
        out_H = math.ceil((H + pad_top + pad_bottom - kH) / sH) + 1
        out_W = math.ceil((W + pad_left + pad_right - kW) / sW) + 1
    else:
        out_H = math.floor((H + pad_top + pad_bottom - kH) / sH) + 1
        out_W = math.floor((W + pad_left + pad_right - kW) / sW) + 1

    return [N, C, out_H, out_W]

if __name__ == "__main__":
    model_path = "models/" + sys.argv[1] + ".onnx"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    model = load_and_infer_model(model_path)
    model = resolve_dynamic_shapes(model, batch_size)
    traverse_graph(model)

    print(f"\nMaximum memory requirement for the model: {max_mreq:,} bytes ({max_mreq / 1024:.2f} KB / {max_mreq / (1024 * 1024):.2f} MB)")

    # calculate_total_weight_memory(model_path)