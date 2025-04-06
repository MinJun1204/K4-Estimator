import onnx
from onnx import shape_inference

def resolve_dynamic_shapes(model, batch_size=1):
    for tensor in model.graph.input:
        shape = tensor.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:  # dynamic ('batch_size')인 경우
                dim.dim_value = batch_size
                dim.ClearField("dim_param")  # 'batch_size' 제거
    return model

# 모델 경로
model_path = "models/resnet152-v1-7.onnx"

# 1. 모델 로드
model = onnx.load(model_path)
model = resolve_dynamic_shapes(model, batch_size=1)

# 2. Shape inference 적용
inferred_model = shape_inference.infer_shapes(model)

# 3. Value info 리스트 생성 (입력 + 출력 + 중간 텐서)
all_value_infos = list(inferred_model.graph.input) + \
                  list(inferred_model.graph.output) + \
                  list(inferred_model.graph.value_info)

# 4. Shape을 찾기 위한 헬퍼 함수
def get_shape(name):
    for value_info in all_value_infos:
        if value_info.name == name:
            shape = [dim.dim_value if dim.HasField("dim_value") else "?" 
                     for dim in value_info.type.tensor_type.shape.dim]
            return shape
    return "Shape not found"

# 5. 노드별로 입력과 출력 shape 출력
for node in inferred_model.graph.node:
    print(f"\nNode: {node.op_type}")
    for input_name in node.input:
        print(f"  Input: {input_name}, Shape: {get_shape(input_name)}")
    for output_name in node.output:
        print(f"  Output: {output_name}, Shape: {get_shape(output_name)}")
