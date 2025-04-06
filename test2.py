import onnx
from onnx import shape_inference
from collections import defaultdict, deque
import numpy as np
import math

def load_and_infer_model(path):
    model = onnx.load(path)
    return shape_inference.infer_shapes(model)

if __name__ == "__main__":
    model_path = "models/model.onnx"
    model = load_and_infer_model(model_path)

    print(model.graph.value_info)