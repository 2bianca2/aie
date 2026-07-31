"""Export a 2-layer MLP as an ONNX model for e2e NPU testing.

Structure (no bias; bias on layer 1 triggers an AIE routing error):
  layer 1: h0 = MatMul(x, w1); h1 = Relu(h0)
  layer 2: y  = MatMul(h1, w2)

Weights w1, w2 are baked constants (initializers) -- the standard export form.
The amd-aie backend folds the resulting non-zero constant-pool base offset into
the DMA access pattern, so constant weights compile and run (no need to split
them out as runtime inputs). Run inside the dev container venv (python w/ onnx).
See USER_GUIDE.md 2-4 for the import -> compile -> run flow.
"""

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

D = 128


def vi(name):
    return h.make_tensor_value_info(name, T.FLOAT, [D, D])


rng = np.random.default_rng(0)
w1 = (rng.standard_normal((D, D)) * 0.1).astype(np.float32)
w2 = (rng.standard_normal((D, D)) * 0.1).astype(np.float32)

graph = h.make_graph(
    [
        h.make_node("MatMul", ["x", "w1"], ["h0"]),
        h.make_node("Relu", ["h0"], ["h1"]),
        h.make_node("MatMul", ["h1", "w2"], ["y"]),
    ],
    "mlp_2layer",
    [vi("x")],  # only x is a runtime input; weights are baked constants
    [vi("y")],
    [nh.from_array(w1, "w1"), nh.from_array(w2, "w2")],  # initializers
)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 20)])
onnx.checker.check_model(model)
onnx.save(model, "mlp_2layer.onnx")
print("saved mlp_2layer.onnx (weights baked as constants)")
