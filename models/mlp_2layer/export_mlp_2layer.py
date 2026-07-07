"""Export a 2-layer MLP as an ONNX model for e2e NPU testing.

Structure (no bias; bias on layer 1 triggers an AIE routing error):
  layer 1: h0 = MatMul(x, w1); h1 = Relu(h0)
  layer 2: y  = MatMul(h1, w2)

Weights w1, w2 are runtime inputs (x, w1, w2), not baked initializers: two
constant weights fail to compile on the amd-aie backend ("memref.subview
non-zero base offset"). Run inside the dev container venv (python with onnx).
See USER_GUIDE.md 2-4 for the import -> compile -> run flow.
"""

import onnx
from onnx import TensorProto as T
from onnx import helper as h

D = 128


def vi(name):
    return h.make_tensor_value_info(name, T.FLOAT, [D, D])


graph = h.make_graph(
    [
        h.make_node("MatMul", ["x", "w1"], ["h0"]),
        h.make_node("Relu", ["h0"], ["h1"]),
        h.make_node("MatMul", ["h1", "w2"], ["y"]),
    ],
    "mlp_2layer",
    [vi("x"), vi("w1"), vi("w2")],  # weights are runtime inputs, no initializers
    [vi("y")],
)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 20)])
onnx.checker.check_model(model)
onnx.save(model, "mlp_2layer.onnx")
print("saved mlp_2layer.onnx")
