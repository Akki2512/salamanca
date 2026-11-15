"""Wrap a SONIC encoder exported by the current training code so it takes the released 1762-D input.

The released encoder input carries 11 root-height values after the joint command block that the
current export omits. The wrapper drops those 11 values and feeds the rest through unchanged, so
the benchmark's SONIC backend drives it exactly like the released model.
"""
import sys

import onnx
from onnx import TensorProto, compose, helper

RELEASE_DIM, EXPORT_DIM, CUT_START, CUT_END = 1762, 1751, 584, 595

src, dst = sys.argv[1], sys.argv[2]
enc = onnx.load(src)
assert enc.graph.input[0].type.tensor_type.shape.dim[-1].dim_value == EXPORT_DIM
name = enc.graph.input[0].name
opset = next(o.version for o in enc.opset_import if o.domain in ("", "ai.onnx"))

const = lambda n, v: helper.make_tensor(n, TensorProto.INT64, [len(v)], v)
pre = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("Slice", ["release_obs", "s0", "e0", "ax"], ["head"]),
            helper.make_node("Slice", ["release_obs", "s1", "e1", "ax"], ["tail"]),
            helper.make_node("Concat", ["head", "tail"], ["export_obs"], axis=1),
        ],
        "drop_root_height",
        [helper.make_tensor_value_info("release_obs", TensorProto.FLOAT, [1, RELEASE_DIM])],
        [helper.make_tensor_value_info("export_obs", TensorProto.FLOAT, [1, EXPORT_DIM])],
        [const("s0", [0]), const("e0", [CUT_START]), const("s1", [CUT_END]), const("e1", [RELEASE_DIM]), const("ax", [1])],
    ),
    opset_imports=[helper.make_opsetid("", opset)],
)
pre.ir_version = enc.ir_version
merged = compose.merge_models(pre, enc, io_map=[("export_obs", name)], prefix1="pad_")
merged.graph.input[0].name = "obs_dict"
for node in merged.graph.node:
    node.input[:] = ["obs_dict" if i == "pad_release_obs" else i for i in node.input]
onnx.checker.check_model(merged)
onnx.save(merged, dst)
print("wrote", dst)
