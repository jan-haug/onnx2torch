__all__ = [
    'OnnxLayerNorm',
]

import torch
import torch.nn.functional as F
from torch import nn
from typing import List

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_shape_from_value_info


class OnnxLayerNorm(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, normalized_shape: List[int], epsilon: float):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_data: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        # Identity replacement alternative
        # return input_data.clone()
        return F.layer_norm(
            input_data,
            normalized_shape=self.normalized_shape,
            weight=weight,
            bias=bias,
            eps=self.epsilon,
        )


@add_converter(operation_type='LayerNormalization', version=17)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    axis = node_attributes.get('axis', -1)
    epsilon = node_attributes.get('epsilon', 1e-5)
    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)
    normalized_shape = input_shape[axis:]

    torch_module = OnnxLayerNorm(normalized_shape=normalized_shape, epsilon=epsilon)

    onnx_mapping = OnnxMapping(
        inputs=node.input_values,
        outputs=(node.output_values[0],),
    )
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping,
    )
